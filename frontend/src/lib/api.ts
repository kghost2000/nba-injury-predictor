const API_URL = "";  // calls same origin — Next.js rewrites proxy to the backend

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
  const res = await fetch(`${API_URL}${path}`);
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

export const featureLabels: Record<string, string> = {
  minutes_over_season_avg: "Minutes Over Avg",
  minutes_over_L10: "Minutes Over L10",
  fga_over_season_avg: "FGA Over Avg",
  fga_over_L10: "FGA Over L10",
  cumulative_excess_minutes_L7: "Excess Minutes (7d)",
  cumulative_excess_fga_L7: "Excess FGA (7d)",
  minutes_spike_7d: "Minutes Spike (7d)",
  games_over_35min_L10: "35+ Min Games (L10)",
  minutes_ewma_deviation: "Minutes EWMA Dev",
  fga_per_100: "FGA per 100",
  fga_per_100_over_avg: "FGA/100 Over Avg",
  usage_rate: "Usage Rate",
  usage_over_avg: "Usage Over Avg",
  minutes_share: "Minutes Share",
  is_back_to_back: "Back-to-Back",
  games_in_last_7d: "Games in 7 Days",
  games_in_last_14d: "Games in 14 Days",
  days_since_last_game: "Days Since Last Game",
  consecutive_heavy_games: "Consec. Heavy Games",
  games_into_season: "Games Into Season",
  age: "Age",
  player_height_inches: "Height",
  age_minutes_interaction: "Age x Minutes",
  season_reb_per_game: "Rebounds/Game",
  season_ast_per_game: "Assists/Game",
  season_ast_pct: "Assist %",
  age_squared: "Age Squared",
  age_over_30: "Age Over 30",
  prior_injury_reports_career: "Career Injuries",
  prior_injury_reports_365d: "Injuries (1yr)",
  days_since_last_injury: "Days Since Injury",
  had_recent_injury_30d: "Recent Injury (30d)",
  back_to_back: "Back-to-Back",
  injury_history_score: "Injury History",
  bmi_load_factor: "BMI Load Factor",
  rest_days_last_14: "Rest Days (14d)",
  age_factor: "Age Factor",
  usage_rate_delta: "Usage Rate Delta",
  games_last_7_days: "Games in 7 Days",
  minutes_over_season_avg_zscore: "Minutes Over Avg (Z)",
};

export function formatFeatureName(name: string): string {
  if (featureLabels[name]) return featureLabels[name];
  return name
    .replace(/_/g, " ")
    .replace(/\bL(\d+)\b/g, "last $1")
    .replace(/\b7d\b/g, "7-day")
    .replace(/\b14d\b/g, "14-day")
    .replace(/\b30d\b/g, "30-day")
    .replace(/\b365d\b/g, "1-year");
}
