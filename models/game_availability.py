"""Game availability prediction: will an injured player play in the next game?

Builds a dataset by joining injury_reports → player → player_season → game →
player_game_log, engineers features using utils/injury_features.py and workload
history, then trains and evaluates Logistic Regression and XGBoost classifiers.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.schema import get_engine
from utils.injury_features import parse_injury_description

logger = logging.getLogger(__name__)

# Mapping from full team name (injury_reports) → team_id (NBA Stats API)
# Built from the `team` table in MySQL.
_TEAM_NAME_TO_ID = {
    "Atlanta Hawks": 1610612737,
    "Boston Celtics": 1610612738,
    "Brooklyn Nets": 1610612751,
    "Charlotte Hornets": 1610612766,
    "Chicago Bulls": 1610612741,
    "Cleveland Cavaliers": 1610612739,
    "Dallas Mavericks": 1610612742,
    "Denver Nuggets": 1610612743,
    "Detroit Pistons": 1610612765,
    "Golden State Warriors": 1610612744,
    "Houston Rockets": 1610612745,
    "Indiana Pacers": 1610612754,
    "LA Clippers": 1610612746,
    "Los Angeles Clippers": 1610612746,
    "Los Angeles Lakers": 1610612747,
    "Memphis Grizzlies": 1610612763,
    "Miami Heat": 1610612748,
    "Milwaukee Bucks": 1610612749,
    "Minnesota Timberwolves": 1610612750,
    "New Orleans Pelicans": 1610612740,
    "New York Knicks": 1610612752,
    "Oklahoma City Thunder": 1610612760,
    "Orlando Magic": 1610612753,
    "Philadelphia 76ers": 1610612755,
    "Phoenix Suns": 1610612756,
    "Portland Trail Blazers": 1610612757,
    "Sacramento Kings": 1610612758,
    "San Antonio Spurs": 1610612759,
    "Toronto Raptors": 1610612761,
    "Utah Jazz": 1610612762,
    "Washington Wizards": 1610612764,
}

# Map report_date → season_id (the start-year int used in the DB)
_SEASON_BOUNDS = {
    2021: ("2021-10-01", "2022-07-01"),
    2022: ("2022-10-01", "2023-07-01"),
    2023: ("2023-10-01", "2024-07-01"),
    2024: ("2024-10-01", "2025-07-01"),
    2025: ("2025-10-01", "2026-07-01"),
}

SAVED_DIR = Path(__file__).parent / "saved"


def _get_season_id(report_date):
    """Map a report_date to its season_id (int), or None if outside range."""
    date_str = str(report_date)[:10]
    for season_id, (start, end) in _SEASON_BOUNDS.items():
        if start <= date_str < end:
            return season_id
    return None


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(engine):
    """Build the prediction dataset using SQL-driven joins across core tables.

    For each injury report row:
    1. Match player via player.player_name
    2. Find team_id via player_season for that season
    3. Find next game the team plays after report_date via game table
    4. Check if player appears in player_game_log for that game → target
    5. Extract injury features from description
    6. Compute workload features from player_game_log history
    7. Pull age and height from player_season

    Returns a DataFrame with features and target column 'played'.
    """
    logger.info("Loading injury_reports...")
    injury_df = pd.read_sql(
        "SELECT id, player_name, team, injury_status, injury_description, report_date "
        "FROM injury_reports "
        "WHERE report_date IS NOT NULL "
        "ORDER BY report_date",
        engine,
        parse_dates=["report_date"],
    )
    logger.info(f"  Loaded {len(injury_df)} injury reports.")

    # Map team names to team_ids and assign season_id
    injury_df["team_id"] = injury_df["team"].map(_TEAM_NAME_TO_ID)
    injury_df = injury_df.dropna(subset=["team_id"])
    injury_df["team_id"] = injury_df["team_id"].astype(int)

    injury_df["season_id"] = injury_df["report_date"].apply(_get_season_id)
    injury_df = injury_df.dropna(subset=["season_id"])
    injury_df["season_id"] = injury_df["season_id"].astype(int)
    logger.info(f"  {len(injury_df)} reports after team/season filtering.")

    # Load player name → player_id mapping
    logger.info("Loading player lookup...")
    player_df = pd.read_sql("SELECT player_id, player_name FROM player", engine)
    name_to_pid = dict(zip(player_df["player_name"], player_df["player_id"]))

    # Load player_season for age + height
    logger.info("Loading player_season (age, height)...")
    ps_df = pd.read_sql(
        "SELECT player_id, season_id, team_id, age, player_height_inches FROM player_season",
        engine,
    )
    # Key: (player_id, season_id) → {age, height, team_id}
    ps_lookup = {}
    for _, row in ps_df.iterrows():
        ps_lookup[(row["player_id"], row["season_id"])] = {
            "age": row["age"],
            "player_height_inches": row["player_height_inches"],
            "ps_team_id": row["team_id"],
        }

    # Load game schedule: team_id → sorted list of (date, game_id)
    logger.info("Loading game schedule...")
    game_df = pd.read_sql(
        "SELECT game_id, team_id_home, team_id_away, date FROM game ORDER BY date",
        engine,
    )
    team_schedule = {}
    for _, g in game_df.iterrows():
        for tid in [g["team_id_home"], g["team_id_away"]]:
            team_schedule.setdefault(tid, []).append((g["date"], g["game_id"]))
    # Sort each team's schedule
    for tid in team_schedule:
        team_schedule[tid].sort()

    # Load player_game_log for target + workload features
    logger.info("Loading player_game_log...")
    pgl_df = pd.read_sql(
        "SELECT player_id, game_id, min FROM player_game_log",
        engine,
    )
    # Set of (player_id, game_id) for target lookup
    player_game_set = set(zip(pgl_df["player_id"], pgl_df["game_id"]))

    # Player history: player_id → list of (game_date, minutes), sorted by date
    logger.info("Building player history index...")
    # Need game dates for player_game_log rows
    game_date_map = dict(zip(game_df["game_id"], game_df["date"]))
    pgl_df["game_date"] = pgl_df["game_id"].map(game_date_map)
    pgl_df = pgl_df.dropna(subset=["game_date"])

    player_history = {}
    for pid, group in pgl_df.groupby("player_id"):
        sorted_group = group.sort_values("game_date")
        player_history[pid] = list(zip(sorted_group["game_date"], sorted_group["min"]))

    # Process each injury report
    logger.info("Building dataset rows (this may take a minute)...")
    rows = []
    skipped_no_player = 0
    skipped_no_game = 0

    for _, ir in injury_df.iterrows():
        team_id = ir["team_id"]
        report_date = ir["report_date"]
        player_name = ir["player_name"]
        season_id = ir["season_id"]

        # Resolve player_id
        player_id = name_to_pid.get(player_name)
        if player_id is None:
            skipped_no_player += 1
            continue

        # Find next game for this team after report_date
        schedule = team_schedule.get(team_id)
        if not schedule:
            skipped_no_game += 1
            continue

        report_d = report_date.date() if hasattr(report_date, 'date') else report_date
        next_game = None
        for game_date, game_id in schedule:
            if game_date > report_d:
                next_game = (game_date, game_id)
                break

        if next_game is None:
            skipped_no_game += 1
            continue

        next_game_date, next_game_id = next_game

        # Target: did the player play in that game?
        played = 1 if (player_id, next_game_id) in player_game_set else 0

        # Parse injury description features
        inj_features = parse_injury_description(ir["injury_description"])

        # Compute workload features from player history
        history = player_history.get(player_id, [])
        workload = _compute_workload_features(history, report_d)

        # Player attributes from player_season
        ps_info = ps_lookup.get((player_id, season_id), {})
        age = ps_info.get("age")
        height = ps_info.get("player_height_inches")

        row = {
            "injury_report_id": ir["id"],
            "player_name": player_name,
            "player_id": player_id,
            "team_id": team_id,
            "report_date": report_date,
            "next_game_date": next_game_date,
            "next_game_id": next_game_id,
            "season_id": season_id,
            "played": played,
            # Injury status
            "injury_status": ir["injury_status"],
            # Injury description features
            "body_region": inj_features["body_region"],
            "severity": inj_features["severity"],
            "is_non_injury": int(inj_features["is_non_injury"]),
            # Workload features
            **workload,
            # Player attributes
            "age": age,
            "player_height_inches": height,
        }
        rows.append(row)

    logger.info(
        f"  Built {len(rows)} dataset rows "
        f"(skipped: {skipped_no_player} no player match, "
        f"{skipped_no_game} no next game)."
    )

    df = pd.DataFrame(rows)

    # Exclude "Out" status — trivially predictive (ruled out = won't play by definition)
    before_out = len(df)
    df = df[df["injury_status"] != "Out"]
    logger.info(f"  Excluded {before_out - len(df)} 'Out' rows (trivially predictive).")

    # Exclude non-injury rows (G-League, rest, personal, etc.) — not predictable
    before_ni = len(df)
    df = df[df["is_non_injury"] == 0]
    logger.info(f"  Excluded {before_ni - len(df)} non-injury rows (G-League, rest, personal).")

    # Deduplicate: same player + same report_date → keep last (most recent report)
    df = df.drop_duplicates(subset=["player_id", "report_date"], keep="last")
    logger.info(f"  {len(df)} rows after deduplication.")

    return df


def _compute_workload_features(history, report_date):
    """Compute workload features from a player's game history prior to report_date.

    Args:
        history: list of (game_date, minutes) tuples, sorted by game_date
        report_date: date or datetime

    Returns:
        dict with workload feature values
    """
    from datetime import timedelta

    # Normalize to date for comparison
    if hasattr(report_date, 'date'):
        rd = report_date.date()
    else:
        rd = report_date

    # Filter to games before report_date
    prior = [(gd, mins) for gd, mins in history if gd < rd]

    if not prior:
        return {
            "minutes_last_game": 0.0,
            "avg_minutes_last_5": 0.0,
            "games_played_last_7d": 0,
            "games_played_last_14d": 0,
            "days_since_last_game": 30.0,
        }

    # Most recent game
    last_game_date, last_game_mins = prior[-1]
    days_since = (rd - last_game_date).days

    # Last 5 games avg minutes
    last_5 = prior[-5:]
    valid_mins = [m for _, m in last_5 if m is not None and m > 0]
    avg_min_5 = float(np.mean(valid_mins)) if valid_mins else 0.0

    # Games in last 7 and 14 days
    seven_days_ago = rd - timedelta(days=7)
    fourteen_days_ago = rd - timedelta(days=14)

    games_7d = sum(1 for gd, _ in prior if gd >= seven_days_ago)
    games_14d = sum(1 for gd, _ in prior if gd >= fourteen_days_ago)

    return {
        "minutes_last_game": float(last_game_mins or 0.0),
        "avg_minutes_last_5": avg_min_5,
        "games_played_last_7d": games_7d,
        "games_played_last_14d": games_14d,
        "days_since_last_game": float(days_since),
    }


# ---------------------------------------------------------------------------
# Feature engineering for modeling
# ---------------------------------------------------------------------------

def _prepare_features(df):
    """Convert raw dataset DataFrame into feature matrix X and target y.

    One-hot encodes injury_status and body_region. Fills missing values.
    """
    feature_df = df.copy()

    # One-hot encode injury_status
    status_dummies = pd.get_dummies(
        feature_df["injury_status"], prefix="status"
    ).astype(int)
    feature_df = pd.concat([feature_df, status_dummies], axis=1)

    # One-hot encode body_region
    region_dummies = pd.get_dummies(
        feature_df["body_region"].fillna("unknown"), prefix="region"
    ).astype(int)
    feature_df = pd.concat([feature_df, region_dummies], axis=1)

    # Fill missing values
    feature_df["severity"] = feature_df["severity"].fillna(0).astype(int)
    feature_df["age"] = feature_df["age"].fillna(feature_df["age"].median())
    feature_df["player_height_inches"] = feature_df["player_height_inches"].fillna(
        feature_df["player_height_inches"].median()
    )

    # Select feature columns (all injury_status and is_non_injury excluded)
    region_cols = sorted(c for c in feature_df.columns if c.startswith("region_"))
    numeric_cols = [
        "severity",
        "minutes_last_game",
        "avg_minutes_last_5",
        "games_played_last_7d",
        "games_played_last_14d",
        "days_since_last_game",
        "age",
        "player_height_inches",
    ]

    feature_cols = region_cols + numeric_cols
    X = feature_df[feature_cols].values.astype(float)
    y = feature_df["played"].values.astype(int)

    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate():
    """Main entry point: load data, train models, print metrics, save models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    engine = get_engine()

    # Build or load dataset
    cache_path = SAVED_DIR / "dataset_cache.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached dataset from {cache_path}")
        df = pd.read_pickle(cache_path)
    else:
        df = build_dataset(engine)
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        logger.info(f"Cached dataset to {cache_path}")

    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['played'].value_counts()}")
    logger.info(f"Seasons: {sorted(df['season_id'].unique())}")

    # Temporal split per plan:
    #   Train: 2021-22 through 2023-24 (season_id 2021, 2022, 2023)
    #   Validation: 2024-25 (season_id 2024)
    #   Test: 2025-26 (season_id 2025)
    train_seasons = {2021, 2022, 2023}
    val_season = 2024
    test_season = 2025

    train_df = df[df["season_id"].isin(train_seasons)]
    val_df = df[df["season_id"] == val_season]
    test_df = df[df["season_id"] == test_season]

    logger.info(f"\nTrain set: {len(train_df)} rows (seasons: {sorted(train_seasons)})")
    logger.info(f"Val set:   {len(val_df)} rows (season: {val_season})")
    logger.info(f"Test set:  {len(test_df)} rows (season: {test_season})")

    if len(train_df) == 0:
        logger.error("No training data. Aborting.")
        return

    # Prepare features — fit on train columns, align all splits
    X_train, y_train, train_cols = _prepare_features(train_df)

    all_feature_cols = sorted(set(train_cols))
    splits = {"Validation": val_df, "Test": test_df}
    aligned_splits = {}

    for name, split_df in splits.items():
        if len(split_df) == 0:
            logger.warning(f"  {name} set is empty, skipping.")
            continue
        X_split, y_split, split_cols = _prepare_features(split_df)
        all_feature_cols_union = sorted(set(train_cols) | set(split_cols))
        aligned_splits[name] = (
            _align_columns(split_cols, all_feature_cols_union, X_split),
            y_split,
        )
        # Also re-align training data if new columns appeared
        all_feature_cols = all_feature_cols_union

    X_train_aligned = _align_columns(train_cols, all_feature_cols, X_train)
    feature_names = all_feature_cols

    # --- Logistic Regression ---
    logger.info("\n" + "=" * 60)
    logger.info("Training Logistic Regression...")
    logger.info("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aligned)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    # --- XGBoost ---
    logger.info("\n" + "=" * 60)
    logger.info("Training XGBoost...")
    logger.info("=" * 60)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train_aligned, y_train)

    # Evaluate on each split
    for split_name, (X_eval, y_eval) in aligned_splits.items():
        print(f"\n{'=' * 60}")
        print(f"  {split_name} Set Results")
        print(f"{'=' * 60}")

        X_eval_scaled = scaler.transform(X_eval)

        lr_preds = lr.predict(X_eval_scaled)
        lr_probs = lr.predict_proba(X_eval_scaled)[:, 1]
        _print_metrics("Logistic Regression", y_eval, lr_preds, lr_probs)

        xgb_preds = xgb.predict(X_eval)
        xgb_probs = xgb.predict_proba(X_eval)[:, 1]
        _print_metrics("XGBoost", y_eval, xgb_preds, xgb_probs)

        _print_comparison(y_eval, lr_preds, lr_probs, xgb_preds, xgb_probs)

    # Feature importances (print once)
    _print_feature_importance_lr(lr, feature_names)
    _print_feature_importance_xgb(xgb, feature_names)

    # Save models
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SAVED_DIR / "logistic_regression.pkl", "wb") as f:
        pickle.dump({"model": lr, "scaler": scaler, "features": feature_names}, f)

    with open(SAVED_DIR / "xgboost.pkl", "wb") as f:
        pickle.dump({"model": xgb, "features": feature_names}, f)

    logger.info(f"\nModels saved to {SAVED_DIR}/")


def _align_columns(existing_cols, all_cols, X):
    """Align feature matrix to a common set of columns, filling missing with 0."""
    col_to_idx = {c: i for i, c in enumerate(existing_cols)}
    aligned = np.zeros((X.shape[0], len(all_cols)))
    for i, col in enumerate(all_cols):
        if col in col_to_idx:
            aligned[:, i] = X[:, col_to_idx[col]]
    return aligned


def _print_metrics(model_name, y_true, y_pred, y_prob):
    """Print classification metrics."""
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:>5d}  FP={cm[0][1]:>5d}")
    print(f"  FN={cm[1][0]:>5d}  TP={cm[1][1]:>5d}")


def _print_feature_importance_lr(model, feature_names):
    """Print Logistic Regression coefficients sorted by absolute value."""
    coefs = model.coef_[0]
    importance = sorted(
        zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True
    )
    print("\nFeature Importances (LR coefficients):")
    for name, coef in importance[:15]:
        print(f"  {name:<35s} {coef:+.4f}")


def _print_feature_importance_xgb(model, feature_names):
    """Print XGBoost feature importances (gain)."""
    importances = model.feature_importances_
    importance = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    print("\nFeature Importances (XGBoost gain):")
    for name, imp in importance[:15]:
        print(f"  {name:<35s} {imp:.4f}")


def _print_comparison(y_test, lr_preds, lr_probs, xgb_preds, xgb_probs):
    """Print side-by-side model comparison."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    print(f"\n{'Metric':<15s} {'Logistic Reg':>15s} {'XGBoost':>15s}")
    print("-" * 45)

    for metric_name, metric_fn in [
        ("Accuracy", accuracy_score),
        ("F1", f1_score),
    ]:
        lr_val = metric_fn(y_test, lr_preds)
        xgb_val = metric_fn(y_test, xgb_preds)
        print(f"{metric_name:<15s} {lr_val:>15.4f} {xgb_val:>15.4f}")

    lr_auc = roc_auc_score(y_test, lr_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    print(f"{'ROC-AUC':<15s} {lr_auc:>15.4f} {xgb_auc:>15.4f}")
