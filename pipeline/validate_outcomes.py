"""Validate predictions against actual outcomes.

For predictions from N days ago, check player_game_log for the 3-day window.
Compute ROC-AUC, PR-AUC, confusion matrix, and tier hit rates.
Store results in batch_metrics table.

Run: conda run -n nba python -m pipeline.validate_outcomes
"""

import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import text

from database.schema import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VALIDATION_LAG_DAYS = 3  # wait N days before validating
DEFAULT_THRESHOLD = 0.07  # optimized threshold from training


def get_predictions_to_validate(conn, prediction_date):
    """Fetch unverified predictions for a given date."""
    result = conn.execute(
        text(
            "SELECT id, player_id, player_name, team_id, prediction_date, "
            "risk_score, risk_percentile, risk_tier "
            "FROM daily_predictions "
            "WHERE prediction_date = :pred_date "
            "AND outcome_verified = 0"
        ),
        {"pred_date": prediction_date},
    )
    return result.fetchall()


def check_player_outcome(conn, player_id, team_id, prediction_date):
    """Check if a player missed a game in the 3-day window after prediction.

    Returns (games_in_window, games_missed, games_played, had_injury_report, injury_desc)
    """
    window_start = prediction_date + timedelta(days=1)
    window_end = prediction_date + timedelta(days=3)

    # Find team games in the window
    team_games = conn.execute(
        text(
            "SELECT game_id, date FROM game "
            "WHERE (team_id_home = :team_id OR team_id_away = :team_id) "
            "AND date >= :start AND date <= :end"
        ),
        {"team_id": team_id, "start": window_start, "end": window_end},
    ).fetchall()

    if not team_games:
        return 0, 0, 0, 0, None

    games_in_window = len(team_games)
    games_missed = 0
    games_played = 0

    for game in team_games:
        game_id = game[0]
        played = conn.execute(
            text(
                "SELECT COUNT(*) FROM player_game_log "
                "WHERE player_id = :pid AND game_id = :gid"
            ),
            {"pid": player_id, "gid": game_id},
        ).scalar()
        if played > 0:
            games_played += 1
        else:
            games_missed += 1

    # Check for injury report in window
    had_injury = 0
    injury_desc = None
    ir_result = conn.execute(
        text(
            "SELECT injury_description FROM injury_reports ir "
            "JOIN player p ON ir.player_name = p.player_name "
            "WHERE p.player_id = :pid "
            "AND ir.report_date >= :start AND ir.report_date <= :end "
            "AND ir.injury_status IN ('Out', 'Doubtful', 'Questionable') "
            "LIMIT 1"
        ),
        {"pid": player_id, "start": window_start, "end": window_end},
    ).fetchone()
    if ir_result:
        had_injury = 1
        injury_desc = ir_result[0]

    return games_in_window, games_missed, games_played, had_injury, injury_desc


def run(prediction_date=None, lag_days=VALIDATION_LAG_DAYS, threshold=DEFAULT_THRESHOLD):
    """Run outcome validation for predictions from lag_days ago."""
    engine = get_engine()
    today = datetime.now().date()

    if prediction_date is None:
        prediction_date = today - timedelta(days=lag_days)
    elif isinstance(prediction_date, str):
        prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d").date()

    logger.info(f"Validating predictions from {prediction_date} (lag={lag_days} days)")

    with engine.begin() as conn:
        predictions = get_predictions_to_validate(conn, prediction_date)

        if not predictions:
            logger.info("No unverified predictions found for this date.")
            return

        logger.info(f"  Found {len(predictions)} predictions to validate")

        # Validate each prediction
        y_true = []
        y_scores = []
        tiers = []
        validated_count = 0

        for pred in predictions:
            pred_id = pred[0]
            player_id = pred[1]
            team_id = pred[3]
            risk_score = pred[5]
            risk_tier = pred[7]

            if team_id is None:
                continue

            games_in_window, games_missed, games_played, had_injury, injury_desc = \
                check_player_outcome(conn, player_id, team_id, prediction_date)

            # Skip if no team game in window (can't evaluate)
            if games_in_window == 0:
                continue

            missed = 1 if games_missed > 0 else 0

            # Update daily_predictions with actual outcome
            conn.execute(
                text(
                    "UPDATE daily_predictions "
                    "SET outcome_verified = 1, missed_game_actual = :missed "
                    "WHERE id = :pred_id"
                ),
                {"missed": missed, "pred_id": pred_id},
            )

            # Insert detailed outcome
            conn.execute(
                text(
                    "INSERT INTO prediction_outcomes "
                    "(prediction_id, player_id, prediction_date, verification_date, "
                    "games_in_window, games_missed, games_played, "
                    "had_injury_report, injury_description) "
                    "VALUES "
                    "(:pred_id, :pid, :pred_date, :verif_date, "
                    ":giw, :gm, :gp, :hir, :idesc)"
                ),
                {
                    "pred_id": pred_id,
                    "pid": player_id,
                    "pred_date": prediction_date,
                    "verif_date": today,
                    "giw": games_in_window,
                    "gm": games_missed,
                    "gp": games_played,
                    "hir": had_injury,
                    "idesc": injury_desc,
                },
            )

            y_true.append(missed)
            y_scores.append(risk_score)
            tiers.append(risk_tier)
            validated_count += 1

        logger.info(f"  Validated {validated_count} predictions")

        if validated_count < 10:
            logger.warning("Too few validated predictions for meaningful metrics.")
            return

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        tiers = np.array(tiers)

        # Compute metrics
        positive_rate = y_true.mean()
        logger.info(f"  Positive rate: {positive_rate:.4f} ({y_true.sum()}/{len(y_true)})")

        # Need at least one positive and one negative for ranking metrics
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.warning("All outcomes identical; cannot compute ranking metrics.")
            roc_auc = None
            pr_auc = None
        else:
            roc_auc = roc_auc_score(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  PR-AUC: {pr_auc:.4f}")

        # Confusion matrix at threshold
        y_pred = (y_scores >= threshold).astype(int)
        if y_pred.sum() > 0 or y_true.sum() > 0:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            tn, fp, fn, tp = len(y_true), 0, 0, 0
            prec = rec = f1 = 0.0

        logger.info(f"  Threshold={threshold}: TP={tp} FP={fp} FN={fn} TN={tn}")
        logger.info(f"  Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

        # Tier hit rates
        high_mask = tiers == "high"
        medium_mask = tiers == "medium"
        high_count = int(high_mask.sum())
        medium_count = int(medium_mask.sum())
        high_hit_rate = float(y_true[high_mask].mean()) if high_count > 0 else None
        medium_hit_rate = float(y_true[medium_mask].mean()) if medium_count > 0 else None

        logger.info(f"  High tier: {high_count} players, hit rate={high_hit_rate}")
        logger.info(f"  Medium tier: {medium_count} players, hit rate={medium_hit_rate}")

        # Store batch metrics
        conn.execute(
            text(
                "INSERT INTO batch_metrics "
                "(metric_date, prediction_date, n_predictions, n_outcomes, "
                "positive_rate, roc_auc, pr_auc, threshold_used, "
                "true_positives, false_positives, true_negatives, false_negatives, "
                "precision_score, recall_score, f1_score, "
                "high_tier_count, high_tier_hit_rate, "
                "medium_tier_count, medium_tier_hit_rate, model_version) "
                "VALUES "
                "(:mdate, :pdate, :n_pred, :n_out, "
                ":pos_rate, :roc, :pr, :thresh, "
                ":tp, :fp, :tn, :fn, "
                ":prec, :rec, :f1, "
                ":htc, :hthr, :mtc, :mthr, :mv) "
                "ON DUPLICATE KEY UPDATE "
                "n_predictions = VALUES(n_predictions), "
                "n_outcomes = VALUES(n_outcomes), "
                "positive_rate = VALUES(positive_rate), "
                "roc_auc = VALUES(roc_auc), "
                "pr_auc = VALUES(pr_auc), "
                "true_positives = VALUES(true_positives), "
                "false_positives = VALUES(false_positives), "
                "true_negatives = VALUES(true_negatives), "
                "false_negatives = VALUES(false_negatives), "
                "precision_score = VALUES(precision_score), "
                "recall_score = VALUES(recall_score), "
                "f1_score = VALUES(f1_score), "
                "high_tier_count = VALUES(high_tier_count), "
                "high_tier_hit_rate = VALUES(high_tier_hit_rate), "
                "medium_tier_count = VALUES(medium_tier_count), "
                "medium_tier_hit_rate = VALUES(medium_tier_hit_rate)"
            ),
            {
                "mdate": today,
                "pdate": prediction_date,
                "n_pred": len(predictions),
                "n_out": validated_count,
                "pos_rate": float(positive_rate) if positive_rate is not None else None,
                "roc": float(roc_auc) if roc_auc is not None else None,
                "pr": float(pr_auc) if pr_auc is not None else None,
                "thresh": threshold,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "prec": float(prec),
                "rec": float(rec),
                "f1": float(f1),
                "htc": high_count,
                "hthr": high_hit_rate,
                "mtc": medium_count,
                "mthr": medium_hit_rate,
                "mv": "stacking_v1",
            },
        )

        logger.info(f"Batch metrics saved for prediction_date={prediction_date}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate injury risk predictions")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Prediction date to validate (YYYY-MM-DD). Defaults to lag_days ago.",
    )
    parser.add_argument(
        "--lag", type=int, default=VALIDATION_LAG_DAYS,
        help=f"Days to wait before validating (default: {VALIDATION_LAG_DAYS})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Classification threshold (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()
    run(prediction_date=args.date, lag_days=args.lag, threshold=args.threshold)
