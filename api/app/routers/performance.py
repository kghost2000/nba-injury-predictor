from fastapi import APIRouter, Depends, Query
from sqlalchemy import Connection, text

from app.database import get_connection
from app.schemas import BatchMetric, PerformanceSummary

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/current", response_model=PerformanceSummary)
def get_current_performance(
    window_days: int = Query(30, le=365),
    conn: Connection = Depends(get_connection),
):
    """Aggregate model performance metrics over a recent window."""
    row = conn.execute(
        text(
            "SELECT "
            "  COUNT(*) AS n_batches, "
            "  AVG(roc_auc) AS avg_roc_auc, "
            "  AVG(pr_auc) AS avg_pr_auc, "
            "  AVG(precision_score) AS avg_precision, "
            "  AVG(recall_score) AS avg_recall, "
            "  AVG(f1_score) AS avg_f1, "
            "  AVG(positive_rate) AS avg_positive_rate, "
            "  AVG(high_tier_hit_rate) AS avg_high_tier_hit_rate, "
            "  SUM(n_predictions) AS total_predictions, "
            "  SUM(n_outcomes) AS total_outcomes "
            "FROM batch_metrics "
            "WHERE metric_date >= DATE_SUB(CURDATE(), INTERVAL :days DAY)"
        ),
        {"days": window_days},
    ).fetchone()

    return PerformanceSummary(
        window_days=window_days,
        n_batches=row[0] or 0,
        avg_roc_auc=float(row[1]) if row[1] is not None else None,
        avg_pr_auc=float(row[2]) if row[2] is not None else None,
        avg_precision=float(row[3]) if row[3] is not None else None,
        avg_recall=float(row[4]) if row[4] is not None else None,
        avg_f1=float(row[5]) if row[5] is not None else None,
        avg_positive_rate=float(row[6]) if row[6] is not None else None,
        avg_high_tier_hit_rate=float(row[7]) if row[7] is not None else None,
        total_predictions=int(row[8]) if row[8] is not None else 0,
        total_outcomes=int(row[9]) if row[9] is not None else 0,
    )


@router.get("/history", response_model=list[BatchMetric])
def get_performance_history(
    days: int = Query(30, le=365),
    conn: Connection = Depends(get_connection),
):
    """Daily time series of model performance metrics for charting."""
    rows = conn.execute(
        text(
            "SELECT metric_date, prediction_date, n_predictions, n_outcomes, "
            "positive_rate, roc_auc, pr_auc, threshold_used, "
            "precision_score, recall_score, f1_score, "
            "high_tier_count, high_tier_hit_rate, "
            "medium_tier_count, medium_tier_hit_rate "
            "FROM batch_metrics "
            "WHERE metric_date >= DATE_SUB(CURDATE(), INTERVAL :days DAY) "
            "ORDER BY metric_date ASC"
        ),
        {"days": days},
    ).fetchall()

    return [
        BatchMetric(
            metric_date=r[0],
            prediction_date=r[1],
            n_predictions=r[2],
            n_outcomes=r[3],
            positive_rate=float(r[4]) if r[4] is not None else None,
            roc_auc=float(r[5]) if r[5] is not None else None,
            pr_auc=float(r[6]) if r[6] is not None else None,
            threshold_used=float(r[7]) if r[7] is not None else None,
            precision_score=float(r[8]) if r[8] is not None else None,
            recall_score=float(r[9]) if r[9] is not None else None,
            f1_score=float(r[10]) if r[10] is not None else None,
            high_tier_count=int(r[11]) if r[11] is not None else None,
            high_tier_hit_rate=float(r[12]) if r[12] is not None else None,
            medium_tier_count=int(r[13]) if r[13] is not None else None,
            medium_tier_hit_rate=float(r[14]) if r[14] is not None else None,
        )
        for r in rows
    ]
