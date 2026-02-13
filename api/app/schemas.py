from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class TopFactor(BaseModel):
    feature: str
    z_score: float
    importance_score: float
    direction: str
    value: float


class PredictionResponse(BaseModel):
    player_id: int
    player_name: str
    team_abbreviation: Optional[str] = None
    prediction_date: date
    risk_score: float
    risk_percentile: float
    risk_tier: str
    top_factors: list[TopFactor]
    outcome_verified: bool
    missed_game_actual: Optional[bool] = None


class PlayerPredictionHistory(BaseModel):
    prediction_date: date
    risk_score: float
    risk_percentile: float
    risk_tier: str
    top_factors: list[TopFactor]
    outcome_verified: bool
    missed_game_actual: Optional[bool] = None


class PlayerDetail(BaseModel):
    player_id: int
    player_name: str
    team_abbreviation: Optional[str] = None
    predictions: list[PlayerPredictionHistory]


class BatchMetric(BaseModel):
    metric_date: date
    prediction_date: date
    n_predictions: int
    n_outcomes: int
    positive_rate: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    threshold_used: Optional[float] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    f1_score: Optional[float] = None
    high_tier_count: Optional[int] = None
    high_tier_hit_rate: Optional[float] = None
    medium_tier_count: Optional[int] = None
    medium_tier_hit_rate: Optional[float] = None


class PerformanceSummary(BaseModel):
    window_days: int
    n_batches: int
    avg_roc_auc: Optional[float] = None
    avg_pr_auc: Optional[float] = None
    avg_precision: Optional[float] = None
    avg_recall: Optional[float] = None
    avg_f1: Optional[float] = None
    avg_positive_rate: Optional[float] = None
    avg_high_tier_hit_rate: Optional[float] = None
    total_predictions: int
    total_outcomes: int
