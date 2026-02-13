-- 001_prediction_tables.sql
-- Creates tables for daily predictions, outcomes, batch metrics, and team lookup.
-- MySQL/MariaDB syntax. Run against the `nba` database.

-- Team lookup: maps numeric team_id to abbreviation for display
CREATE TABLE IF NOT EXISTS team_lookup (
    team_id INT NOT NULL PRIMARY KEY,
    abbreviation VARCHAR(5) NOT NULL,
    full_name VARCHAR(50) NOT NULL
);

-- Seed team_lookup with all 30 NBA teams
INSERT IGNORE INTO team_lookup (team_id, abbreviation, full_name) VALUES
(1610612737, 'ATL', 'Atlanta Hawks'),
(1610612738, 'BOS', 'Boston Celtics'),
(1610612751, 'BKN', 'Brooklyn Nets'),
(1610612766, 'CHA', 'Charlotte Hornets'),
(1610612741, 'CHI', 'Chicago Bulls'),
(1610612739, 'CLE', 'Cleveland Cavaliers'),
(1610612742, 'DAL', 'Dallas Mavericks'),
(1610612743, 'DEN', 'Denver Nuggets'),
(1610612765, 'DET', 'Detroit Pistons'),
(1610612744, 'GSW', 'Golden State Warriors'),
(1610612745, 'HOU', 'Houston Rockets'),
(1610612754, 'IND', 'Indiana Pacers'),
(1610612746, 'LAC', 'Los Angeles Clippers'),
(1610612747, 'LAL', 'Los Angeles Lakers'),
(1610612763, 'MEM', 'Memphis Grizzlies'),
(1610612748, 'MIA', 'Miami Heat'),
(1610612749, 'MIL', 'Milwaukee Bucks'),
(1610612750, 'MIN', 'Minnesota Timberwolves'),
(1610612740, 'NOP', 'New Orleans Pelicans'),
(1610612752, 'NYK', 'New York Knicks'),
(1610612760, 'OKC', 'Oklahoma City Thunder'),
(1610612753, 'ORL', 'Orlando Magic'),
(1610612755, 'PHI', 'Philadelphia 76ers'),
(1610612756, 'PHX', 'Phoenix Suns'),
(1610612757, 'POR', 'Portland Trail Blazers'),
(1610612758, 'SAC', 'Sacramento Kings'),
(1610612759, 'SAS', 'San Antonio Spurs'),
(1610612761, 'TOR', 'Toronto Raptors'),
(1610612762, 'UTA', 'Utah Jazz'),
(1610612764, 'WAS', 'Washington Wizards');

-- Daily predictions: one row per player per prediction date
CREATE TABLE IF NOT EXISTS daily_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    team_id INT,
    prediction_date DATE NOT NULL,
    risk_score FLOAT NOT NULL,
    risk_percentile FLOAT NOT NULL,
    risk_tier VARCHAR(10) NOT NULL,           -- 'high', 'medium', 'low'
    top_factors JSON,                          -- top 5 factors with z-scores
    feature_vector JSON,                       -- full feature vector for debugging
    model_version VARCHAR(50) DEFAULT 'stacking_v1',
    outcome_verified TINYINT(1) DEFAULT 0,
    missed_game_actual TINYINT(1) DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_player_date (player_id, prediction_date),
    INDEX idx_prediction_date (prediction_date),
    INDEX idx_risk_tier (risk_tier),
    INDEX idx_outcome_verified (outcome_verified)
);

-- Prediction outcomes: detailed outcome info linked to daily_predictions
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    player_id INT NOT NULL,
    prediction_date DATE NOT NULL,
    verification_date DATE NOT NULL,
    games_in_window INT DEFAULT 0,            -- team games in 3-day window
    games_missed INT DEFAULT 0,               -- games player missed
    games_played INT DEFAULT 0,               -- games player played
    had_injury_report TINYINT(1) DEFAULT 0,   -- injury report in window
    injury_description VARCHAR(200) DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES daily_predictions(id) ON DELETE CASCADE,
    INDEX idx_prediction_id (prediction_id),
    INDEX idx_verification_date (verification_date)
);

-- Batch metrics: daily aggregate model performance
CREATE TABLE IF NOT EXISTS batch_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_date DATE NOT NULL,                 -- date metrics were computed
    prediction_date DATE NOT NULL,             -- which prediction batch was evaluated
    n_predictions INT NOT NULL,
    n_outcomes INT NOT NULL,
    positive_rate FLOAT,                       -- actual base rate for this batch
    roc_auc FLOAT,
    pr_auc FLOAT,
    threshold_used FLOAT,
    true_positives INT,
    false_positives INT,
    true_negatives INT,
    false_negatives INT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    high_tier_count INT,                       -- how many were flagged 'high'
    high_tier_hit_rate FLOAT,                  -- fraction of 'high' that actually missed
    medium_tier_count INT,
    medium_tier_hit_rate FLOAT,
    model_version VARCHAR(50) DEFAULT 'stacking_v1',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_metric_prediction_date (metric_date, prediction_date),
    INDEX idx_metric_date (metric_date)
);
