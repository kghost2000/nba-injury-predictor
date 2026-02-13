const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface TopFactor {
  feature: string;
  z_score: number;
  importance_score: number;
  direction: string;
  value: number;
}

export interface Prediction {
  player_id: number;
  player_name: string;
  team_abbreviation: string | null;
  prediction_date: string;
  risk_score: number;
  risk_percentile: number;
  risk_tier: string;
  top_factors: TopFactor[];
  outcome_verified: boolean;
  missed_game_actual: boolean | null;
}

export interface PlayerDetail {
  player_id: number;
  player_name: string;
  team_abbreviation: string | null;
  predictions: {
    prediction_date: string;
    risk_score: number;
    risk_percentile: number;
    risk_tier: string;
    top_factors: TopFactor[];
    outcome_verified: boolean;
    missed_game_actual: boolean | null;
  }[];
}

export interface BatchMetric {
  metric_date: string;
  prediction_date: string;
  n_predictions: number;
  n_outcomes: number;
  positive_rate: number | null;
  roc_auc: number | null;
  pr_auc: number | null;
  threshold_used: number | null;
  precision_score: number | null;
  recall_score: number | null;
  f1_score: number | null;
  high_tier_count: number | null;
  high_tier_hit_rate: number | null;
  medium_tier_count: number | null;
  medium_tier_hit_rate: number | null;
}

export interface PerformanceSummary {
  window_days: number;
  n_batches: number;
  avg_roc_auc: number | null;
  avg_pr_auc: number | null;
  avg_precision: number | null;
  avg_recall: number | null;
  avg_f1: number | null;
  avg_positive_rate: number | null;
  avg_high_tier_hit_rate: number | null;
  total_predictions: number;
  total_outcomes: number;
}

async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { next: { revalidate: 300 } });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getTodayPredictions(tier?: string): Promise<Prediction[]> {
  const params = tier ? `?tier=${tier}` : "";
  return fetchAPI(`/api/predictions/today${params}`);
}

export async function getPlayerDetail(playerId: number): Promise<PlayerDetail> {
  return fetchAPI(`/api/predictions/player/${playerId}`);
}

export async function getPerformanceSummary(
  windowDays = 30
): Promise<PerformanceSummary> {
  return fetchAPI(`/api/performance/current?window_days=${windowDays}`);
}

export async function getPerformanceHistory(
  days = 30
): Promise<BatchMetric[]> {
  return fetchAPI(`/api/performance/history?days=${days}`);
}
