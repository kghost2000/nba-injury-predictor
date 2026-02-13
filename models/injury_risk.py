"""Injury risk early warning system: predict missed games from workload anomalies.

Each row = one player-game. Label = did this player miss a game in the next 3 days?
Uses rolling workload deviations, pace-adjusted stats, schedule context, and player
attributes to detect injury risk before it happens.
"""

import logging
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.schema import get_engine

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved" / "injury_risk"


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def load_raw_data(engine):
    """Load player_game_log, game, player_season, and injury_reports tables."""
    logger.info("Loading player_game_log...")
    pgl_df = pd.read_sql(
        "SELECT player_id, game_id, team_id, season_id, min, fga, fta, oreb, tov "
        "FROM player_game_log",
        engine,
    )
    logger.info(f"  {len(pgl_df)} rows")

    logger.info("Loading game schedule...")
    game_df = pd.read_sql(
        "SELECT game_id, team_id_home, team_id_away, season_id, date FROM game ORDER BY date",
        engine,
        parse_dates=["date"],
    )
    logger.info(f"  {len(game_df)} games")

    logger.info("Loading player_season...")
    ps_df = pd.read_sql(
        "SELECT player_id, season_id, age, player_height_inches, reb, ast, ast_pct "
        "FROM player_season",
        engine,
    )
    logger.info(f"  {len(ps_df)} rows")

    logger.info("Loading injury_reports...")
    ir_df = pd.read_sql(
        "SELECT player_name, report_date FROM injury_reports "
        "WHERE injury_status IN ('Out', 'Doubtful', 'Questionable') "
        "AND report_date IS NOT NULL",
        engine,
        parse_dates=["report_date"],
    )
    # Map player_name → player_id
    player_lookup = pd.read_sql("SELECT player_id, player_name FROM player", engine)
    ir_df = ir_df.merge(player_lookup, on="player_name", how="inner")
    logger.info(f"  {len(ir_df)} injury reports (with player_id match)")

    return pgl_df, game_df, ps_df, ir_df


# ---------------------------------------------------------------------------
# 2. Build base dataframe with team aggregates
# ---------------------------------------------------------------------------

def build_base_dataframe(pgl_df, game_df):
    """Join pgl with game dates and compute team-level aggregates.

    Returns df sorted by (player_id, game_date) with team aggregate columns.
    """
    # Map game_id → date
    game_date_map = game_df.set_index("game_id")["date"]
    pgl_df = pgl_df.copy()
    pgl_df["game_date"] = pgl_df["game_id"].map(game_date_map)
    pgl_df = pgl_df.dropna(subset=["game_date"])

    # Map game_id → season_id from game table (more reliable)
    game_season_map = game_df.set_index("game_id")["season_id"]
    pgl_df["season_id"] = pgl_df["game_id"].map(game_season_map).fillna(pgl_df["season_id"]).astype(int)

    # Team aggregates per (game_id, team_id)
    team_agg = (
        pgl_df.groupby(["game_id", "team_id"])
        .agg(
            team_fga=("fga", "sum"),
            team_fta=("fta", "sum"),
            team_oreb=("oreb", "sum"),
            team_tov=("tov", "sum"),
            team_min=("min", "sum"),
        )
        .reset_index()
    )

    # Merge team aggregates back
    df = pgl_df.merge(team_agg, on=["game_id", "team_id"], how="left")

    # Sort chronologically per player
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    logger.info(f"Base dataframe: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, ps_df, ir_df):
    """Add all 32 feature columns."""
    df = df.copy()
    df = _add_workload_features(df)
    df = _add_pace_features(df)
    df = _add_schedule_features(df)
    df = _add_player_attributes(df, ps_df)
    df = _add_injury_history_features(df, ir_df)
    return df


def _add_workload_features(df):
    """8 workload deviation features using rolling windows."""
    grouped = df.groupby("player_id")

    # Expanding means (shifted to exclude current game)
    df["_min_expanding"] = grouped["min"].apply(
        lambda s: s.expanding().mean().shift(1)
    ).values
    df["_fga_expanding"] = grouped["fga"].apply(
        lambda s: s.expanding().mean().shift(1)
    ).values

    # Rolling 10-game means (shifted)
    df["_min_L10"] = grouped["min"].apply(
        lambda s: s.rolling(10, min_periods=3).mean().shift(1)
    ).values
    df["_fga_L10"] = grouped["fga"].apply(
        lambda s: s.rolling(10, min_periods=3).mean().shift(1)
    ).values

    # Deviation features
    df["minutes_over_season_avg"] = df["min"] - df["_min_expanding"]
    df["minutes_over_L10"] = df["min"] - df["_min_L10"]
    df["fga_over_season_avg"] = df["fga"] - df["_fga_expanding"]
    df["fga_over_L10"] = df["fga"] - df["_fga_L10"]

    # Cumulative excess over 7-day window
    # For each row, sum (min - season_avg) for all games in the prior 7 days
    df["_min_excess"] = df["min"] - df["_min_expanding"]
    df["_fga_excess"] = df["fga"] - df["_fga_expanding"]

    def _cumulative_excess_7d(group, col):
        dates = group["game_date"].values
        vals = group[col].values
        result = np.full(len(group), np.nan)
        for i in range(len(group)):
            cutoff = dates[i] - np.timedelta64(7, "D")
            mask = (dates[:i + 1] >= cutoff) & (dates[:i + 1] <= dates[i])
            result[i] = np.nansum(vals[:i + 1][mask[:i + 1]])
        return pd.Series(result, index=group.index)

    df["cumulative_excess_minutes_L7"] = (
        df.groupby("player_id", group_keys=False)
        .apply(lambda g: _cumulative_excess_7d(g, "_min_excess"), include_groups=False)
    )
    df["cumulative_excess_fga_L7"] = (
        df.groupby("player_id", group_keys=False)
        .apply(lambda g: _cumulative_excess_7d(g, "_fga_excess"), include_groups=False)
    )

    # Minutes spike: 7-day rolling avg - 14-day rolling avg
    df["_min_7d_avg"] = grouped["min"].apply(
        lambda s: s.rolling(7, min_periods=3).mean().shift(1)
    ).values
    df["_min_14d_avg"] = grouped["min"].apply(
        lambda s: s.rolling(14, min_periods=5).mean().shift(1)
    ).values
    df["minutes_spike_7d"] = df["_min_7d_avg"] - df["_min_14d_avg"]

    # EWMA workload deviation (span=10, shifted)
    df["_min_ewma"] = grouped["min"].apply(
        lambda s: s.ewm(span=10, min_periods=3).mean().shift(1)
    ).values
    df["minutes_ewma_deviation"] = df["min"] - df["_min_ewma"]

    # Games over 35 minutes in last 10
    df["games_over_35min_L10"] = grouped["min"].apply(
        lambda s: s.rolling(10, min_periods=1).apply(
            lambda x: (x > 35).sum(), raw=True
        ).shift(1)
    ).values

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=temp_cols)

    return df


def _add_pace_features(df):
    """4 pace-adjusted features using team aggregates."""
    # Team possessions estimate
    df["team_poss"] = df["team_fga"] + 0.44 * df["team_fta"] - df["team_oreb"] + df["team_tov"]
    df["team_poss"] = df["team_poss"].replace(0, np.nan)

    # FGA per 100 possessions
    df["fga_per_100"] = (df["fga"] / df["team_poss"]) * 100

    # fga_per_100 deviation from player's expanding mean
    df["_fga100_expanding"] = (
        df.groupby("player_id")["fga_per_100"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .values
    )
    df["fga_per_100_over_avg"] = df["fga_per_100"] - df["_fga100_expanding"]

    # Usage rate: 100 * ((fga + 0.44*fta + tov) * (team_min/5)) / (min * team_poss)
    df["min"] = df["min"].replace(0, np.nan)
    df["usage_rate"] = (
        100
        * ((df["fga"] + 0.44 * df["fta"] + df["tov"]) * (df["team_min"] / 5))
        / (df["min"] * df["team_poss"])
    )

    # Minutes share: player's share of team minutes, normalized for 5-player lineup
    df["minutes_share"] = df["min"] / (df["team_min"] / 5)

    # Usage deviation from player's expanding mean
    df["_usg_expanding"] = (
        df.groupby("player_id")["usage_rate"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .values
    )
    df["usage_over_avg"] = df["usage_rate"] - df["_usg_expanding"]

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=temp_cols)

    return df


def _add_schedule_features(df):
    """4 schedule context features."""
    grouped = df.groupby("player_id")

    # Days since last game
    df["days_since_last_game"] = grouped["game_date"].diff().dt.days
    df["is_back_to_back"] = (df["days_since_last_game"] == 1).astype(int)

    # Games in last 7 / 14 days
    def _games_in_window(group, days):
        dates = group["game_date"].values
        result = np.zeros(len(group), dtype=int)
        for i in range(len(group)):
            cutoff = dates[i] - np.timedelta64(days, "D")
            result[i] = int(np.sum((dates[:i] >= cutoff) & (dates[:i] < dates[i])))
        return pd.Series(result, index=group.index)

    df["games_in_last_7d"] = (
        df.groupby("player_id", group_keys=False)
        .apply(lambda g: _games_in_window(g, 7), include_groups=False)
    )
    df["games_in_last_14d"] = (
        df.groupby("player_id", group_keys=False)
        .apply(lambda g: _games_in_window(g, 14), include_groups=False)
    )

    # Season progression: cumulative game count within each season
    df["games_into_season"] = df.groupby(["player_id", "season_id"]).cumcount() + 1

    # Consecutive heavy games (>32 minutes)
    def _consecutive_heavy(group):
        mins = group["min"].values
        result = np.zeros(len(group), dtype=int)
        for i in range(len(group)):
            count = 0
            for j in range(i - 1, -1, -1):
                if mins[j] > 32:
                    count += 1
                else:
                    break
            result[i] = count
        return pd.Series(result, index=group.index)

    df["consecutive_heavy_games"] = (
        df.groupby("player_id", group_keys=False)
        .apply(_consecutive_heavy, include_groups=False)
    )

    return df


def _add_player_attributes(df, ps_df):
    """8 player attribute features: age, height, age*minutes, position proxies, nonlinear age."""
    ps_key = ps_df.drop_duplicates(subset=["player_id", "season_id"]).set_index(
        ["player_id", "season_id"]
    )[["age", "player_height_inches", "reb", "ast", "ast_pct"]]

    df = df.merge(
        ps_key, left_on=["player_id", "season_id"], right_index=True, how="left"
    )

    # Fill missing with medians
    df["age"] = df["age"].fillna(df["age"].median())
    df["player_height_inches"] = df["player_height_inches"].fillna(
        df["player_height_inches"].median()
    )

    # Interaction
    df["age_minutes_interaction"] = df["age"] * df["minutes_over_season_avg"]

    # Position proxy features (season-level stats)
    df["season_reb_per_game"] = df["reb"].fillna(0)
    df["season_ast_per_game"] = df["ast"].fillna(0)
    df["season_ast_pct"] = df["ast_pct"].fillna(df["ast_pct"].median())

    # Drop raw columns used only for deriving features
    df = df.drop(columns=["reb", "ast", "ast_pct"])

    # Nonlinear age features
    df["age_squared"] = df["age"] ** 2
    df["age_over_30"] = (df["age"] > 30).astype(int)

    return df


def _add_injury_history_features(df, ir_df):
    """4 injury history features using prior injury reports."""
    # Build per-player sorted list of report dates
    ir_by_player = {}
    for _, row in ir_df.iterrows():
        pid = row["player_id"]
        rd = pd.Timestamp(row["report_date"])
        ir_by_player.setdefault(pid, []).append(rd)
    for pid in ir_by_player:
        ir_by_player[pid].sort()

    career_counts = np.zeros(len(df), dtype=int)
    year_counts = np.zeros(len(df), dtype=int)
    days_since = np.full(len(df), 365.0)
    recent_30d = np.zeros(len(df), dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        pid = row["player_id"]
        game_date = pd.Timestamp(row["game_date"])
        dates = ir_by_player.get(pid, [])

        if not dates:
            continue

        # Count reports before this game date
        career_count = 0
        year_count = 0
        latest_days = 365.0
        has_recent = 0
        cutoff_365 = game_date - timedelta(days=365)
        cutoff_30 = game_date - timedelta(days=30)

        for rd in dates:
            if rd >= game_date:
                break
            career_count += 1
            if rd >= cutoff_365:
                year_count += 1
            if rd >= cutoff_30:
                has_recent = 1
            days_diff = (game_date - rd).days
            if days_diff < latest_days:
                latest_days = days_diff

        career_counts[i] = career_count
        year_counts[i] = year_count
        days_since[i] = min(latest_days, 365.0)
        recent_30d[i] = has_recent

    df = df.copy()
    df["prior_injury_reports_career"] = career_counts
    df["prior_injury_reports_365d"] = year_counts
    df["days_since_last_injury"] = days_since
    df["had_recent_injury_30d"] = recent_30d

    return df


# ---------------------------------------------------------------------------
# 4. Build labels
# ---------------------------------------------------------------------------

def build_labels(df, game_df, ir_df):
    """Add missed_game label: did this player miss a team game in the next 3 days?

    Excludes rows where:
    - The team had no game in the window
    - The player missed but had no injury report within 5 days (rest/personal/DNP)
    """
    # Build team schedule lookup: team_id → sorted list of (date, game_id)
    team_games = {}
    for _, g in game_df.iterrows():
        for tid in [g["team_id_home"], g["team_id_away"]]:
            team_games.setdefault(tid, []).append((g["date"], g["game_id"]))
    for tid in team_games:
        team_games[tid].sort()

    # Build set of (player_id, game_id) for presence check
    player_game_set = set(zip(df["player_id"], df["game_id"]))

    # Build injury report lookup: player_id → sorted list of report dates
    ir_by_player = {}
    for _, ir_row in ir_df.iterrows():
        pid = ir_row["player_id"]
        rd = pd.Timestamp(ir_row["report_date"])
        ir_by_player.setdefault(pid, []).append(rd)
    for pid in ir_by_player:
        ir_by_player[pid].sort()

    labels = []
    for _, row in df.iterrows():
        team_id = row["team_id"]
        game_date = row["game_date"]
        player_id = row["player_id"]

        if pd.isna(game_date):
            labels.append(np.nan)
            continue

        # Ensure game_date is a Timestamp for comparison
        if not isinstance(game_date, pd.Timestamp):
            game_date = pd.Timestamp(game_date)

        window_start = game_date + timedelta(days=1)  # exclusive of current day
        window_end = game_date + timedelta(days=3)

        # Find team games in (game_date, game_date + 3d]
        schedule = team_games.get(team_id, [])
        upcoming = [
            (d, gid) for d, gid in schedule
            if pd.Timestamp(d) >= window_start and pd.Timestamp(d) <= window_end
        ]

        if not upcoming:
            labels.append(np.nan)  # No team game in window → exclude
            continue

        # Check if player missed any upcoming team game
        missed = any(
            (player_id, gid) not in player_game_set for _, gid in upcoming
        )

        if missed:
            # Cross-reference with injury reports: require an injury report
            # for this player within 5 days of the missed game to confirm
            # it was injury-related (not rest/personal/load management).
            ir_dates = ir_by_player.get(player_id, [])
            has_injury_report = any(
                abs((rd - game_date).days) <= 5 for rd in ir_dates
            )
            if not has_injury_report:
                labels.append(np.nan)  # Non-injury absence → exclude
                continue

        labels.append(int(missed))

    df = df.copy()
    df["missed_game"] = labels

    before = len(df)
    df = df.dropna(subset=["missed_game"])
    df["missed_game"] = df["missed_game"].astype(int)
    excluded_no_game = before - len(df)
    logger.info(f"Labels: {excluded_no_game} rows excluded (no upcoming game or non-injury absence)")
    logger.info(f"  {len(df)} rows remaining, positive rate: {df['missed_game'].mean():.3f}")

    return df


# ---------------------------------------------------------------------------
# 5. Minimum games filter
# ---------------------------------------------------------------------------

def _apply_min_games_filter(df, min_games=10):
    """Drop rows before each player's Nth game to ensure stable rolling features."""
    df = df.copy()
    df["_game_num"] = df.groupby("player_id").cumcount() + 1
    before = len(df)
    df = df[df["_game_num"] >= min_games].drop(columns=["_game_num"])
    logger.info(f"Min-games filter: dropped {before - len(df)} rows (first {min_games - 1} games per player)")
    return df


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Workload deviation (9)
    "minutes_over_season_avg",
    "minutes_over_L10",
    "fga_over_season_avg",
    "fga_over_L10",
    "cumulative_excess_minutes_L7",
    "cumulative_excess_fga_L7",
    "minutes_spike_7d",
    "games_over_35min_L10",
    "minutes_ewma_deviation",
    # Pace-adjusted (5)
    "fga_per_100",
    "fga_per_100_over_avg",
    "usage_rate",
    "usage_over_avg",
    "minutes_share",
    # Schedule context (6)
    "is_back_to_back",
    "games_in_last_7d",
    "games_in_last_14d",
    "days_since_last_game",
    "consecutive_heavy_games",
    "games_into_season",
    # Player attributes (3)
    "age",
    "player_height_inches",
    "age_minutes_interaction",
    # Position proxy (3)
    "season_reb_per_game",
    "season_ast_per_game",
    "season_ast_pct",
    # Nonlinear age (2)
    "age_squared",
    "age_over_30",
    # Injury history (4)
    "prior_injury_reports_career",
    "prior_injury_reports_365d",
    "days_since_last_injury",
    "had_recent_injury_30d",
]


def build_dataset(engine):
    """Run the full pipeline: load → base → features → labels → filter → cache."""
    pgl_df, game_df, ps_df, ir_df = load_raw_data(engine)
    df = build_base_dataframe(pgl_df, game_df)
    df = engineer_features(df, ps_df, ir_df)
    df = build_labels(df, game_df, ir_df)
    df = _apply_min_games_filter(df)
    return df


# ---------------------------------------------------------------------------
# 7. Hyperparameter tuning helpers
# ---------------------------------------------------------------------------

def _tune_xgboost(X_train, y_train, scale_pos_weight):
    """Tune XGBoost via RandomizedSearchCV with TimeSeriesSplit."""
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from xgboost import XGBClassifier

    param_dist = {
        "n_estimators": [100, 200, 300, 500, 700],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5, 10, 20],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.3, 0.5, 0.7, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5, 1, 3],
        "reg_alpha": [0, 0.01, 0.1, 1, 5],
        "reg_lambda": [0.1, 0.5, 1, 3, 10],
    }

    base_xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=50,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest XGB CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best XGB params: {search.best_params_}")

    # Retrain on full training set with best params
    best_params = search.best_params_
    best_xgb = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    best_xgb.fit(X_train, y_train)

    return best_xgb, best_params


def _tune_logistic_regression(X_train_scaled, y_train):
    """Tune Logistic Regression via RandomizedSearchCV with TimeSeriesSplit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    param_dist = {
        "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }

    base_lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base_lr,
        param_distributions=param_dist,
        n_iter=18,  # 9 C values * 2 penalties = 18 total combos (exhaustive)
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train_scaled, y_train)

    print(f"\nBest LR CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best LR params: {search.best_params_}")

    # Retrain on full training set with best params
    best_params = search.best_params_
    best_lr = LogisticRegression(
        **best_params,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    best_lr.fit(X_train_scaled, y_train)

    return best_lr, best_params


def _tune_lightgbm(X_train, y_train, scale_pos_weight):
    """Tune LightGBM via RandomizedSearchCV with TimeSeriesSplit."""
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from lightgbm import LGBMClassifier

    param_dist = {
        "n_estimators": [100, 200, 300, 500, 700],
        "max_depth": [3, 4, 5, 6, 8, 10, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [5, 10, 20, 50, 100],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.3, 0.5, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1, 5],
        "reg_lambda": [0, 0.1, 1, 5, 10],
    }

    base_lgbm = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )

    cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base_lgbm,
        param_distributions=param_dist,
        n_iter=50,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest LightGBM CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best LightGBM params: {search.best_params_}")

    # Retrain on full training set with best params
    best_params = search.best_params_
    best_lgbm = LGBMClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    best_lgbm.fit(X_train, y_train)

    return best_lgbm, best_params


def _select_features(xgb_model, feature_names, X_train, y_train, X_val, y_val,
                     scaler, lr_model, lgbm_model, scale_pos_weight,
                     xgb_params, lr_params, lgbm_params):
    """Drop bottom 3 features by XGB importance, retrain and compare."""
    from sklearn.metrics import average_precision_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    importances = xgb_model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances, range(len(feature_names))),
                      key=lambda x: x[1])

    # Bottom 3 features by gain
    drop_features = [name for name, imp, idx in feat_imp[:3]]
    keep_mask = [name not in drop_features for name in feature_names]
    keep_features = [name for name in feature_names if name not in drop_features]

    print(f"\nFeature selection: candidates to drop: {drop_features}")

    # Baseline val PR-AUC with current models
    X_val_scaled = scaler.transform(X_val)
    xgb_prob_base = xgb_model.predict_proba(X_val_scaled)[:, 1]
    lr_prob_base = lr_model.predict_proba(X_val_scaled)[:, 1]
    lgbm_prob_base = lgbm_model.predict_proba(X_val_scaled)[:, 1]
    xgb_prauc_base = average_precision_score(y_val, xgb_prob_base)
    lr_prauc_base = average_precision_score(y_val, lr_prob_base)
    lgbm_prauc_base = average_precision_score(y_val, lgbm_prob_base)
    print(f"  Baseline val PR-AUC — LR: {lr_prauc_base:.4f}, XGB: {xgb_prauc_base:.4f}, LightGBM: {lgbm_prauc_base:.4f}")

    # Retrain on reduced features
    X_train_reduced = X_train[:, keep_mask]
    X_val_reduced = X_val[:, keep_mask]

    new_scaler = StandardScaler()
    X_train_reduced_scaled = new_scaler.fit_transform(X_train_reduced)
    X_val_reduced_scaled = new_scaler.transform(X_val_reduced)

    new_lr = LogisticRegression(
        **lr_params,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    new_lr.fit(X_train_reduced_scaled, y_train)

    new_xgb = XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    new_xgb.fit(X_train_reduced_scaled, y_train)

    new_lgbm = LGBMClassifier(
        **lgbm_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    new_lgbm.fit(X_train_reduced_scaled, y_train)

    xgb_prob_new = new_xgb.predict_proba(X_val_reduced_scaled)[:, 1]
    lr_prob_new = new_lr.predict_proba(X_val_reduced_scaled)[:, 1]
    lgbm_prob_new = new_lgbm.predict_proba(X_val_reduced_scaled)[:, 1]
    xgb_prauc_new = average_precision_score(y_val, xgb_prob_new)
    lr_prauc_new = average_precision_score(y_val, lr_prob_new)
    lgbm_prauc_new = average_precision_score(y_val, lgbm_prob_new)
    print(f"  Reduced val PR-AUC  — LR: {lr_prauc_new:.4f}, XGB: {xgb_prauc_new:.4f}, LightGBM: {lgbm_prauc_new:.4f}")

    # Keep reduced set only if XGB PR-AUC improves or is neutral (within 0.002)
    if xgb_prauc_new >= xgb_prauc_base - 0.002:
        print(f"  -> Keeping reduced feature set ({len(keep_features)} features)")
        print(f"     Dropped: {drop_features}")
        return keep_features, new_lr, new_xgb, new_lgbm, new_scaler
    else:
        print(f"  -> Keeping full feature set (reduction hurt XGB PR-AUC by {xgb_prauc_base - xgb_prauc_new:.4f})")
        return list(feature_names), lr_model, xgb_model, lgbm_model, scaler


def _optimize_threshold(y_val, y_prob_val):
    """Sweep thresholds on validation set, return threshold that maximizes F1."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.01, 0.51, 0.01)
    best_f1 = 0
    best_thresh = 0.5

    print("\nThreshold sweep on validation set:")
    print(f"  {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")

    for thresh in thresholds:
        y_pred = (y_prob_val >= thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Print the best
    y_pred_best = (y_prob_val >= best_thresh).astype(int)
    prec_best = precision_score(y_val, y_pred_best, zero_division=0)
    rec_best = recall_score(y_val, y_pred_best, zero_division=0)
    print(f"  {'BEST ->':>10s}  thresh={best_thresh:.2f}  prec={prec_best:.4f}  "
          f"rec={rec_best:.4f}  F1={best_f1:.4f}")

    return best_thresh


def _calibrate_models(models, X_val_scaled, y_val):
    """Calibrate pre-fitted models using isotonic regression on validation set."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    calibrated = {}
    for name, model in models.items():
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        cal.fit(X_val_scaled, y_val)
        calibrated[name] = cal

    return calibrated


def _build_stacking_ensemble(lr, xgb, lgbm, X_train_scaled, y_train, X_val_scaled, y_val):
    """Build stacking ensemble: LR/XGB/LightGBM predictions → logistic meta-learner."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    # Generate out-of-fold predictions on training set (avoids leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    lr_oof = cross_val_predict(lr, X_train_scaled, y_train, cv=cv, method="predict_proba")[:, 1]
    xgb_oof = cross_val_predict(xgb, X_train_scaled, y_train, cv=cv, method="predict_proba")[:, 1]
    lgbm_oof = cross_val_predict(lgbm, X_train_scaled, y_train, cv=cv, method="predict_proba")[:, 1]

    # Stack into meta-features
    meta_train = np.column_stack([lr_oof, xgb_oof, lgbm_oof])

    # Fit meta-learner
    meta_lr = LogisticRegression(random_state=42, max_iter=1000)
    meta_lr.fit(meta_train, y_train)

    # Print meta-learner weights
    print(f"\n  Meta-learner weights: LR={meta_lr.coef_[0][0]:.3f}, "
          f"XGB={meta_lr.coef_[0][1]:.3f}, LightGBM={meta_lr.coef_[0][2]:.3f}")

    return meta_lr


def _predict_stacking(meta_lr, lr, xgb, lgbm, X_scaled):
    """Generate stacking ensemble predictions."""
    lr_prob = lr.predict_proba(X_scaled)[:, 1]
    xgb_prob = xgb.predict_proba(X_scaled)[:, 1]
    lgbm_prob = lgbm.predict_proba(X_scaled)[:, 1]
    meta_features = np.column_stack([lr_prob, xgb_prob, lgbm_prob])
    return meta_lr.predict_proba(meta_features)[:, 1]


def _plot_curves(y_true, model_probs, split_name):
    """Generate ROC and PR curve plots.

    model_probs: dict of {model_name: y_prob}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {split_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVED_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in model_probs.items():
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)
        ax.plot(rec_arr, prec_arr, label=f"{name} (AP={pr_auc_val:.3f})")
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {split_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVED_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved ROC and PR curve plots to {SAVED_DIR}/")


# ---------------------------------------------------------------------------
# 8. Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate():
    """Main entry point: build dataset, tune models, evaluate, save."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    engine = get_engine()

    # --- 1. Load cached dataset ---
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
    logger.info(f"Target distribution:\n{df['missed_game'].value_counts()}")
    logger.info(f"Positive rate: {df['missed_game'].mean():.4f}")
    logger.info(f"Seasons: {sorted(df['season_id'].unique())}")

    # --- 2. Temporal split ---
    train_seasons = {2019, 2020, 2021, 2022}
    val_season = 2023
    test_season = 2024

    train_df = df[df["season_id"].isin(train_seasons)]
    val_df = df[df["season_id"] == val_season]
    test_df = df[df["season_id"] == test_season]

    logger.info(f"\nTrain: {len(train_df)} rows (seasons {sorted(train_seasons)})")
    logger.info(f"Val:   {len(val_df)} rows (season {val_season})")
    logger.info(f"Test:  {len(test_df)} rows (season {test_season})")

    if len(train_df) == 0:
        logger.error("No training data. Aborting.")
        return

    # --- 3. Prepare feature matrices ---
    feature_cols = list(FEATURE_COLS)

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["missed_game"].values.astype(int)
    X_val = val_df[feature_cols].values.astype(float) if len(val_df) > 0 else None
    y_val = val_df["missed_game"].values.astype(int) if len(val_df) > 0 else None
    X_test = test_df[feature_cols].values.astype(float) if len(test_df) > 0 else None
    y_test = test_df["missed_game"].values.astype(int) if len(test_df) > 0 else None

    # Fill NaN with 0
    X_train = np.nan_to_num(X_train, nan=0.0)
    if X_val is not None:
        X_val = np.nan_to_num(X_val, nan=0.0)
    if X_test is not None:
        X_test = np.nan_to_num(X_test, nan=0.0)

    # Class imbalance ratio
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    logger.info(f"Train class balance: neg={neg}, pos={pos}, ratio={scale_pos_weight:.1f}:1")

    # --- 3. Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    # --- 4. Tune Logistic Regression ---
    print("\n" + "=" * 60)
    print("  Tuning Logistic Regression")
    print("=" * 60)
    lr, lr_params = _tune_logistic_regression(X_train_scaled, y_train)

    # --- 5. Tune XGBoost ---
    print("\n" + "=" * 60)
    print("  Tuning XGBoost")
    print("=" * 60)
    xgb, xgb_params = _tune_xgboost(X_train_scaled, y_train, scale_pos_weight)

    # --- 6. Tune LightGBM ---
    print("\n" + "=" * 60)
    print("  Tuning LightGBM")
    print("=" * 60)
    lgbm, lgbm_params = _tune_lightgbm(X_train_scaled, y_train, scale_pos_weight)

    # --- 7. Feature selection ---
    print("\n" + "=" * 60)
    print("  Feature Selection")
    print("=" * 60)
    if X_val is not None:
        feature_cols, lr, xgb, lgbm, scaler = _select_features(
            xgb, feature_cols, X_train, y_train, X_val, y_val,
            scaler, lr, lgbm, scale_pos_weight, xgb_params, lr_params, lgbm_params,
        )
        # Recompute scaled matrices with potentially new scaler/features
        X_train_final = np.nan_to_num(
            train_df[feature_cols].values.astype(float), nan=0.0
        )
        X_val_final = np.nan_to_num(
            val_df[feature_cols].values.astype(float), nan=0.0
        )
        X_test_final = np.nan_to_num(
            test_df[feature_cols].values.astype(float), nan=0.0
        ) if X_test is not None else None

        X_val_scaled = scaler.transform(X_val_final)
        X_test_scaled = scaler.transform(X_test_final) if X_test_final is not None else None
        X_train_scaled = scaler.transform(X_train_final)
    else:
        print("  Skipping (no validation set)")

    # --- 8. Calibrate models ---
    print("\n" + "=" * 60)
    print("  Probability Calibration")
    print("=" * 60)
    calibrated = _calibrate_models(
        {"LR": lr, "XGB": xgb, "LightGBM": lgbm},
        X_val_scaled, y_val,
    )
    cal_lr, cal_xgb, cal_lgbm = calibrated["LR"], calibrated["XGB"], calibrated["LightGBM"]

    # --- 9. Stacking ensemble ---
    print("\n" + "=" * 60)
    print("  Stacking Ensemble")
    print("=" * 60)
    meta_lr = _build_stacking_ensemble(
        cal_lr, cal_xgb, cal_lgbm,
        X_train_scaled, y_train, X_val_scaled, y_val,
    )

    # --- 10. Threshold optimization (on stacking ensemble) ---
    print("\n" + "=" * 60)
    print("  Threshold Optimization")
    print("=" * 60)
    optimal_threshold = 0.5
    if X_val_scaled is not None:
        stacking_val_prob = _predict_stacking(meta_lr, cal_lr, cal_xgb, cal_lgbm, X_val_scaled)
        optimal_threshold = _optimize_threshold(y_val, stacking_val_prob)
        print(f"\nOptimal threshold (Stacking, val F1): {optimal_threshold:.2f}")
    else:
        print("  Skipping (no validation set)")

    # --- 11. Evaluate on val and test ---
    splits = {}
    if X_val_scaled is not None:
        splits["Validation"] = (X_val_scaled, y_val)
    if X_test_scaled is not None:
        splits["Test"] = (X_test_scaled, y_test)

    models_to_eval = [
        ("Logistic Regression", lr),
        ("XGBoost", xgb),
        ("LightGBM", lgbm),
        ("XGB (calibrated)", cal_xgb),
        ("Stacking Ensemble", None),  # special handling via _predict_stacking
    ]

    for split_name, (X_eval_scaled, y_eval) in splits.items():
        print(f"\n{'=' * 60}")
        print(f"  {split_name} Set Results")
        print(f"{'=' * 60}")

        for model_name, model in models_to_eval:
            if model_name == "Stacking Ensemble":
                y_prob = _predict_stacking(meta_lr, cal_lr, cal_xgb, cal_lgbm, X_eval_scaled)
            else:
                y_prob = model.predict_proba(X_eval_scaled)[:, 1]

            # Default threshold (0.5)
            y_pred_default = (y_prob >= 0.5).astype(int)
            # Optimized threshold
            y_pred_opt = (y_prob >= optimal_threshold).astype(int)

            acc = accuracy_score(y_eval, y_pred_default)
            roc = roc_auc_score(y_eval, y_prob)
            pr_auc = average_precision_score(y_eval, y_prob)

            # Precision@100
            top_100_idx = np.argsort(y_prob)[-100:]
            prec_at_100 = y_eval[top_100_idx].mean()

            print(f"\n--- {model_name} ---")
            print(f"ROC-AUC:       {roc:.4f}")
            print(f"PR-AUC:        {pr_auc:.4f}")
            print(f"Precision@100: {prec_at_100:.4f}")

            # Default threshold metrics
            prec_d = precision_score(y_eval, y_pred_default, zero_division=0)
            rec_d = recall_score(y_eval, y_pred_default, zero_division=0)
            f1_d = f1_score(y_eval, y_pred_default, zero_division=0)
            cm_d = confusion_matrix(y_eval, y_pred_default)
            print(f"\n  Threshold=0.50 (default):")
            print(f"    Accuracy:  {acc:.4f}")
            print(f"    Precision: {prec_d:.4f}")
            print(f"    Recall:    {rec_d:.4f}")
            print(f"    F1:        {f1_d:.4f}")
            print(f"    Confusion: TN={cm_d[0][0]:>6d}  FP={cm_d[0][1]:>6d}")
            print(f"               FN={cm_d[1][0]:>6d}  TP={cm_d[1][1]:>6d}")

            # Optimized threshold metrics
            prec_o = precision_score(y_eval, y_pred_opt, zero_division=0)
            rec_o = recall_score(y_eval, y_pred_opt, zero_division=0)
            f1_o = f1_score(y_eval, y_pred_opt, zero_division=0)
            cm_o = confusion_matrix(y_eval, y_pred_opt)
            print(f"\n  Threshold={optimal_threshold:.2f} (optimized):")
            print(f"    Precision: {prec_o:.4f}")
            print(f"    Recall:    {rec_o:.4f}")
            print(f"    F1:        {f1_o:.4f}")
            print(f"    Confusion: TN={cm_o[0][0]:>6d}  FP={cm_o[0][1]:>6d}")
            print(f"               FN={cm_o[1][0]:>6d}  TP={cm_o[1][1]:>6d}")

    # --- 12. Feature importances ---
    print(f"\n{'=' * 60}")
    print("  Feature Importances (Tuned Models)")
    print(f"{'=' * 60}")

    # LR coefficients
    coefs = lr.coef_[0]
    lr_imp = sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
    print("\nLogistic Regression (coefficients):")
    for name, coef in lr_imp:
        print(f"  {name:<35s} {coef:+.4f}")

    # XGBoost gain
    xgb_importances = xgb.feature_importances_
    xgb_imp = sorted(zip(feature_cols, xgb_importances), key=lambda x: x[1], reverse=True)
    print("\nXGBoost (gain):")
    for name, imp in xgb_imp:
        print(f"  {name:<35s} {imp:.4f}")

    # LightGBM gain
    lgbm_importances = lgbm.feature_importances_
    lgbm_imp = sorted(zip(feature_cols, lgbm_importances), key=lambda x: x[1], reverse=True)
    print("\nLightGBM (gain):")
    for name, imp in lgbm_imp:
        print(f"  {name:<35s} {imp:.4f}")

    # --- 13. Generate ROC/PR curve plots ---
    if X_test_scaled is not None:
        lr_test_prob = lr.predict_proba(X_test_scaled)[:, 1]
        xgb_test_prob = xgb.predict_proba(X_test_scaled)[:, 1]
        lgbm_test_prob = lgbm.predict_proba(X_test_scaled)[:, 1]
        cal_xgb_test_prob = cal_xgb.predict_proba(X_test_scaled)[:, 1]
        stacking_test_prob = _predict_stacking(meta_lr, cal_lr, cal_xgb, cal_lgbm, X_test_scaled)
        _plot_curves(
            y_test,
            {
                "Logistic Regression": lr_test_prob,
                "XGBoost": xgb_test_prob,
                "LightGBM": lgbm_test_prob,
                "XGB (calibrated)": cal_xgb_test_prob,
                "Stacking Ensemble": stacking_test_prob,
            },
            "Test",
        )

    # --- 14. Save all models + stacking + calibrated + threshold + scaler ---
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SAVED_DIR / "logistic_regression.pkl", "wb") as f:
        pickle.dump({
            "model": lr,
            "scaler": scaler,
            "features": feature_cols,
            "params": lr_params,
        }, f)

    with open(SAVED_DIR / "xgboost.pkl", "wb") as f:
        pickle.dump({
            "model": xgb,
            "features": feature_cols,
            "params": xgb_params,
            "threshold": optimal_threshold,
        }, f)

    with open(SAVED_DIR / "lightgbm.pkl", "wb") as f:
        pickle.dump({
            "model": lgbm,
            "features": feature_cols,
            "params": lgbm_params,
            "threshold": optimal_threshold,
        }, f)

    with open(SAVED_DIR / "stacking.pkl", "wb") as f:
        pickle.dump({
            "meta_model": meta_lr,
            "calibrated_lr": cal_lr,
            "calibrated_xgb": cal_xgb,
            "calibrated_lgbm": cal_lgbm,
            "features": feature_cols,
            "threshold": optimal_threshold,
        }, f)

    logger.info(f"\nModels saved to {SAVED_DIR}/")
