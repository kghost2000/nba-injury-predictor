"""Daily batch prediction pipeline.

Reuses the exact feature engineering from models/injury_risk.py, loads the
trained LightGBM model, and writes predictions to the daily_predictions table.

Run: conda run -n nba python -m pipeline.daily_predictions
"""

import json
import logging
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.schema import get_engine
from models.injury_risk import (
    FEATURE_COLS,
    build_base_dataframe,
    engineer_features,
    load_raw_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "models" / "saved" / "injury_risk"
RECENCY_DAYS = 14  # only predict for players with a game in last N days


def load_models():
    """Load LightGBM model and scaler from saved pickles."""
    lgbm_path = SAVED_DIR / "lightgbm.pkl"
    lr_path = SAVED_DIR / "logistic_regression.pkl"

    if not lgbm_path.exists() or not lr_path.exists():
        raise FileNotFoundError(
            f"Model pickles not found in {SAVED_DIR}. "
            "Run: conda run -n nba python main.py --injury-risk"
        )

    with open(lgbm_path, "rb") as f:
        lgbm_data = pickle.load(f)

    with open(lr_path, "rb") as f:
        lr_data = pickle.load(f)

    scaler = lr_data["scaler"]
    feature_cols = lgbm_data["features"]

    return lgbm_data, scaler, feature_cols


def predict_lgbm(lgbm_data, scaler, X):
    """Scale features and generate LightGBM predictions."""
    X_scaled = scaler.transform(X)
    return lgbm_data["calibrated_model"].predict_proba(X_scaled)[:, 1]


def assign_tiers(risk_scores):
    """Assign risk tiers based on percentile rank.

    - Top 5% of daily scores -> 'high' (~25 players/day)
    - Top 15% -> 'medium'
    - Rest -> 'low'
    """
    percentiles = pd.Series(risk_scores).rank(pct=True).values * 100
    tiers = np.where(
        percentiles >= 95, "high",
        np.where(percentiles >= 85, "medium", "low")
    )
    return percentiles, tiers


def compute_top_factors(feature_vector, feature_cols, population_means, population_stds,
                        global_importances, top_n=5):
    """Compute per-prediction explanations using z-score * global importance.

    Returns list of dicts: [{feature, z_score, importance_score, direction, description}]
    """
    factors = []
    for i, col in enumerate(feature_cols):
        val = feature_vector[i]
        mean = population_means[i]
        std = population_stds[i]

        if std == 0 or np.isnan(std):
            z_score = 0.0
        else:
            z_score = (val - mean) / std

        importance = global_importances.get(col, 0.0)
        score = abs(z_score) * importance

        direction = "above" if z_score > 0 else "below"
        factors.append({
            "feature": col,
            "z_score": round(float(z_score), 2),
            "importance_score": round(float(score), 4),
            "direction": direction,
            "value": round(float(val), 3),
        })

    # Sort by importance_score descending, take top N
    factors.sort(key=lambda x: x["importance_score"], reverse=True)
    return factors[:top_n]


def get_global_importances(lgbm_data):
    """Extract global feature importances from the LightGBM model."""
    cal_model = lgbm_data["calibrated_model"]

    # CalibratedClassifierCV wraps the estimator — unwrap to get LGBMClassifier
    base = cal_model.estimator
    while hasattr(base, "estimator"):
        base = base.estimator

    importances = base.feature_importances_
    feature_cols = lgbm_data["features"]
    total = importances.sum()
    if total > 0:
        importances = importances / total  # normalize to sum to 1
    return dict(zip(feature_cols, importances))


def run(prediction_date=None):
    """Run the daily prediction pipeline."""
    engine = get_engine()

    if prediction_date is None:
        prediction_date = datetime.now().date()
    elif isinstance(prediction_date, str):
        prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d").date()

    logger.info(f"Running daily predictions for {prediction_date}")

    # 1. Load models
    logger.info("Loading models...")
    lgbm_data, scaler, feature_cols = load_models()
    threshold = lgbm_data.get("threshold", 0.07)
    logger.info(f"  Features: {len(feature_cols)}, threshold: {threshold}")

    # 2. Run full feature engineering pipeline (same code path as training)
    logger.info("Loading raw data and engineering features...")
    pgl_df, game_df, ps_df, ir_df = load_raw_data(engine)
    df = build_base_dataframe(pgl_df, game_df)
    df = engineer_features(df, ps_df, ir_df)

    # 3. Take each player's most recent game row
    df = df.sort_values(["player_id", "game_date"])
    latest = df.groupby("player_id").tail(1).copy()

    # 4. Filter to players with a game in last RECENCY_DAYS
    cutoff_date = pd.Timestamp(prediction_date) - timedelta(days=RECENCY_DAYS)
    latest = latest[latest["game_date"] >= cutoff_date]
    logger.info(f"  {len(latest)} players with a game in last {RECENCY_DAYS} days")

    if len(latest) == 0:
        logger.warning("No players to predict. Exiting.")
        return

    # 5. Build feature matrix
    X = latest[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    # 6. Predict
    logger.info("Generating predictions...")
    risk_scores = predict_lgbm(lgbm_data, scaler, X)
    percentiles, tiers = assign_tiers(risk_scores)

    # 7. Compute per-prediction explanations
    logger.info("Computing per-prediction explanations...")
    global_importances = get_global_importances(lgbm_data)

    # Population stats from the full engineered dataset (not just latest)
    pop_X = df[feature_cols].values.astype(float)
    pop_X = np.nan_to_num(pop_X, nan=0.0)
    population_means = np.nanmean(pop_X, axis=0)
    population_stds = np.nanstd(pop_X, axis=0)

    # Load player names
    player_names = pd.read_sql(
        "SELECT player_id, player_name FROM player", engine
    ).set_index("player_id")["player_name"]

    # 8. Write to database
    logger.info("Writing predictions to daily_predictions table...")
    inserted = 0
    with engine.begin() as conn:
        for i, (idx, row) in enumerate(latest.iterrows()):
            player_id = int(row["player_id"])
            team_id = int(row["team_id"]) if pd.notna(row.get("team_id")) else None
            player_name = player_names.get(player_id, f"Player {player_id}")

            top_factors = compute_top_factors(
                X[i], feature_cols, population_means, population_stds,
                global_importances
            )

            feature_dict = {col: round(float(X[i][j]), 4) for j, col in enumerate(feature_cols)}

            conn.execute(
                text(
                    "INSERT INTO daily_predictions "
                    "(player_id, player_name, team_id, prediction_date, "
                    "risk_score, risk_percentile, risk_tier, "
                    "top_factors, feature_vector, model_version) "
                    "VALUES "
                    "(:player_id, :player_name, :team_id, :prediction_date, "
                    ":risk_score, :risk_percentile, :risk_tier, "
                    ":top_factors, :feature_vector, :model_version) "
                    "ON DUPLICATE KEY UPDATE "
                    "risk_score = VALUES(risk_score), "
                    "risk_percentile = VALUES(risk_percentile), "
                    "risk_tier = VALUES(risk_tier), "
                    "top_factors = VALUES(top_factors), "
                    "feature_vector = VALUES(feature_vector), "
                    "player_name = VALUES(player_name), "
                    "team_id = VALUES(team_id), "
                    "updated_at = CURRENT_TIMESTAMP"
                ),
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "team_id": team_id,
                    "prediction_date": prediction_date,
                    "risk_score": round(float(risk_scores[i]), 6),
                    "risk_percentile": round(float(percentiles[i]), 2),
                    "risk_tier": str(tiers[i]),
                    "top_factors": json.dumps(top_factors),
                    "feature_vector": json.dumps(feature_dict),
                    "model_version": "lightgbm_v1",
                },
            )
            inserted += 1

    # Summary
    tier_counts = pd.Series(tiers).value_counts()
    logger.info(f"Inserted/updated {inserted} predictions for {prediction_date}")
    logger.info(f"  Tier distribution: {tier_counts.to_dict()}")
    logger.info(f"  Risk score range: [{risk_scores.min():.4f}, {risk_scores.max():.4f}]")
    logger.info(f"  Mean risk: {risk_scores.mean():.4f}")

    # Log top 10 highest risk players
    top_idx = np.argsort(risk_scores)[-10:][::-1]
    logger.info("  Top 10 highest risk:")
    for idx in top_idx:
        row = latest.iloc[idx]
        pid = int(row["player_id"])
        name = player_names.get(pid, f"Player {pid}")
        logger.info(
            f"    {name}: score={risk_scores[idx]:.4f}, "
            f"percentile={percentiles[idx]:.1f}%, tier={tiers[idx]}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run daily injury risk predictions")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Prediction date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()
    run(prediction_date=args.date)
