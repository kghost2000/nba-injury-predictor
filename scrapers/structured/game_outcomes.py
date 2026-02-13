"""Fetch game outcomes and player box score stats from the NBA Stats API.

Uses the leaguegamelog endpoint which returns all player-game rows for a season
in a single request (~26k rows). Populates both the game_outcomes table (did a
player play?) and the player_game_stats table (full box score).
"""

import logging
from datetime import datetime

import requests

from database.models import GameOutcome, PlayerGameStats

logger = logging.getLogger(__name__)

# NBA regular season approximate date ranges (matches nba_historical.py)
SEASONS = {
    "2021-22": ("2021-10-19", "2022-04-10"),
    "2022-23": ("2022-10-18", "2023-04-09"),
    "2023-24": ("2023-10-24", "2024-04-14"),
    "2024-25": ("2024-10-22", "2025-04-13"),
    "2025-26": ("2025-10-21", "2026-04-12"),
}

API_URL = "https://stats.nba.com/stats/leaguegamelog"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_season(db_session, season):
    """Fetch all player game logs for a season and save to the database.

    Makes a single API call to the leaguegamelog endpoint, then creates
    GameOutcome and PlayerGameStats records for each row.

    Args:
        db_session: SQLAlchemy session.
        season: Season string like "2024-25".

    Returns:
        Number of records saved.
    """
    if season not in SEASONS:
        logger.error(f"Unknown season '{season}'. Available: {list(SEASONS.keys())}")
        return 0

    params = {
        "Counter": 0,
        "Direction": "DESC",
        "LeagueID": "00",
        "PlayerOrTeam": "P",
        "Season": season,
        "SeasonType": "Regular Season",
        "Sorter": "DATE",
    }

    logger.info(f"Fetching game logs for season {season}...")

    try:
        resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        return 0

    result_set = data.get("resultSets", [{}])[0]
    headers = result_set.get("headers", [])
    rows = result_set.get("rowSet", [])

    if not headers or not rows:
        logger.warning("No data returned from API.")
        return 0

    # Build a column-name -> index map for easy access
    col = {name: idx for idx, name in enumerate(headers)}

    logger.info(f"Received {len(rows)} player-game rows. Saving...")

    saved = 0
    for row in rows:
        try:
            game_date = datetime.strptime(row[col["GAME_DATE"]], "%Y-%m-%d")

            # Parse minutes: API returns "MM:SS" string or None
            min_raw = row[col["MIN"]]
            minutes = _parse_minutes(min_raw)

            # GameOutcome record
            outcome = GameOutcome(
                player_name=row[col["PLAYER_NAME"]],
                game_date=game_date,
                did_play=True,
                minutes_played=minutes,
                game_id=row[col["GAME_ID"]],
            )
            db_session.add(outcome)

            # PlayerGameStats record
            stats = PlayerGameStats(
                player_name=row[col["PLAYER_NAME"]],
                player_id=row[col["PLAYER_ID"]],
                team=row[col["TEAM_ABBREVIATION"]],
                game_id=row[col["GAME_ID"]],
                game_date=game_date,
                matchup=row[col["MATCHUP"]],
                wl=row[col["WL"]],
                min=minutes,
                pts=row[col["PTS"]],
                reb=row[col["REB"]],
                ast=row[col["AST"]],
                stl=row[col["STL"]],
                blk=row[col["BLK"]],
                tov=row[col["TOV"]],
                pf=row[col["PF"]],
                fgm=row[col["FGM"]],
                fga=row[col["FGA"]],
                fg_pct=row[col["FG_PCT"]],
                fg3m=row[col["FG3M"]],
                fg3a=row[col["FG3A"]],
                fg3_pct=row[col["FG3_PCT"]],
                ftm=row[col["FTM"]],
                fta=row[col["FTA"]],
                ft_pct=row[col["FT_PCT"]],
                oreb=row[col["OREB"]],
                dreb=row[col["DREB"]],
                plus_minus=row[col["PLUS_MINUS"]],
                season=season,
            )
            db_session.add(stats)
            saved += 1

        except Exception as e:
            logger.debug(f"Failed to save row: {e}")

    logger.info(f"Saved {saved} game outcome + stats records for season {season}.")
    return saved


def _parse_minutes(min_raw):
    """Convert minutes value from API to a float.

    The API may return minutes as "MM:SS" string, an integer, a float, or None.
    """
    if min_raw is None:
        return 0.0
    if isinstance(min_raw, (int, float)):
        return float(min_raw)
    if isinstance(min_raw, str) and ":" in min_raw:
        parts = min_raw.split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    try:
        return float(min_raw)
    except (ValueError, TypeError):
        return 0.0
