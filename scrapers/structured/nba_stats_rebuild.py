"""Wipe and rebuild core NBA stats tables from the NBA Stats API.

Covers seasons 2017-18 through 2025-26. Populates:
  - season
  - player
  - player_season       (from leaguedashplayerbiostats)
  - game                (from leaguegamelog PlayerOrTeam=T)
  - player_game_log     (from leaguegamelog PlayerOrTeam=P)
  - player_general_traditional_total (from leaguedashplayerstats)

Also drops the three empty tables: play_by_play, play_by_playv3, shot_chart_detail.
"""

import logging
import time
from datetime import datetime

import requests
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Map season start year → NBA season string
SEASONS_MAP = {
    2017: "2017-18",
    2018: "2018-19",
    2019: "2019-20",
    2020: "2020-21",
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}

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

BASE_URL = "https://stats.nba.com/stats"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch_endpoint(endpoint, params, result_set_index=0):
    """Fetch a single result set from an NBA Stats API endpoint.

    Returns (col_map, rows) where col_map is {HEADER_NAME: index}.
    """
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    rs = data["resultSets"][result_set_index]
    headers = rs["headers"]
    rows = rs["rowSet"]
    col = {name: idx for idx, name in enumerate(headers)}
    return col, rows


def _parse_game_id(raw):
    """Convert API game_id string (e.g. '0021700001') to int (21700001)."""
    return int(raw)


# ---------------------------------------------------------------------------
# Rebuild functions
# ---------------------------------------------------------------------------

def rebuild_seasons(db):
    """Populate the season table with season_id 2017–2025."""
    logger.info("Rebuilding season table...")
    db.execute(text("DELETE FROM season"))
    for season_id in SEASONS_MAP:
        db.execute(text("INSERT INTO season (season_id) VALUES (:sid)"),
                   {"sid": season_id})
    logger.info(f"  Inserted {len(SEASONS_MAP)} seasons.")


def rebuild_players_and_player_seasons(db):
    """Fetch leaguedashplayerbiostats for each season.

    Populates both `player` (deduplicated, latest bio) and `player_season`.
    """
    logger.info("Rebuilding player + player_season tables...")
    db.execute(text("DELETE FROM player_season"))
    db.execute(text("DELETE FROM player"))

    # Collect all player bio data keyed by player_id; keep latest season's data
    players = {}  # player_id → {player_name, college, country, draft_year, draft_round, draft_number}
    player_season_rows = []

    for season_id, season_str in SEASONS_MAP.items():
        logger.info(f"  Fetching leaguedashplayerbiostats for {season_str}...")
        params = {
            "LeagueID": "00",
            "PerMode": "Totals",
            "Season": season_str,
            "SeasonType": "Regular Season",
        }
        try:
            col, rows = _fetch_endpoint("leaguedashplayerbiostats", params)
        except Exception as e:
            logger.error(f"  Failed to fetch {season_str}: {e}")
            time.sleep(1)
            continue

        for row in rows:
            pid = row[col["PLAYER_ID"]]
            pname = row[col["PLAYER_NAME"]]
            team_id = row[col["TEAM_ID"]]

            # Player bio — always overwrite so latest season wins
            players[pid] = {
                "player_id": pid,
                "player_name": pname,
                "college": row[col.get("COLLEGE", -1)] if "COLLEGE" in col else None,
                "country": row[col.get("COUNTRY", -1)] if "COUNTRY" in col else None,
                "draft_year": str(row[col["DRAFT_YEAR"]]) if "DRAFT_YEAR" in col and row[col["DRAFT_YEAR"]] is not None else None,
                "draft_round": str(row[col["DRAFT_ROUND"]]) if "DRAFT_ROUND" in col and row[col["DRAFT_ROUND"]] is not None else None,
                "draft_number": str(row[col["DRAFT_NUMBER"]]) if "DRAFT_NUMBER" in col and row[col["DRAFT_NUMBER"]] is not None else None,
            }

            # Player season row
            age = row[col["AGE"]] if "AGE" in col else None
            player_height = row[col["PLAYER_HEIGHT"]] if "PLAYER_HEIGHT" in col else None
            player_height_inches = row[col["PLAYER_HEIGHT_INCHES"]] if "PLAYER_HEIGHT_INCHES" in col else None
            player_weight = row[col["PLAYER_WEIGHT"]] if "PLAYER_WEIGHT" in col else None
            gp = row[col["GP"]] if "GP" in col else None
            pts = row[col["PTS"]] if "PTS" in col else None
            reb = row[col["REB"]] if "REB" in col else None
            ast = row[col["AST"]] if "AST" in col else None
            net_rating = row[col["NET_RATING"]] if "NET_RATING" in col else None
            oreb_pct = row[col["OREB_PCT"]] if "OREB_PCT" in col else None
            dreb_pct = row[col["DREB_PCT"]] if "DREB_PCT" in col else None
            usg_pct = row[col["USG_PCT"]] if "USG_PCT" in col else None
            ts_pct = row[col["TS_PCT"]] if "TS_PCT" in col else None
            ast_pct = row[col["AST_PCT"]] if "AST_PCT" in col else None

            # Convert player_height_inches to int if possible
            if player_height_inches is not None:
                try:
                    player_height_inches = int(float(player_height_inches))
                except (ValueError, TypeError):
                    player_height_inches = None

            player_season_rows.append({
                "player_id": pid,
                "season_id": season_id,
                "team_id": team_id,
                "age": int(age) if age is not None else None,
                "player_height": str(player_height) if player_height else None,
                "player_height_inches": player_height_inches,
                "player_weight": str(player_weight) if player_weight else None,
                "gp": int(gp) if gp is not None else None,
                "pts": float(pts) if pts is not None else None,
                "reb": float(reb) if reb is not None else None,
                "ast": float(ast) if ast is not None else None,
                "net_rating": float(net_rating) if net_rating is not None else None,
                "oreb_pct": float(oreb_pct) if oreb_pct is not None else None,
                "dreb_pct": float(dreb_pct) if dreb_pct is not None else None,
                "usg_pct": float(usg_pct) if usg_pct is not None else None,
                "ts_pct": float(ts_pct) if ts_pct is not None else None,
                "ast_pct": float(ast_pct) if ast_pct is not None else None,
            })

        logger.info(f"    {len(rows)} player-season rows from {season_str}")
        time.sleep(1)

    # Insert players
    for p in players.values():
        db.execute(text(
            "INSERT INTO player (player_id, player_name, college, country, "
            "draft_year, draft_round, draft_number) "
            "VALUES (:player_id, :player_name, :college, :country, "
            ":draft_year, :draft_round, :draft_number)"
        ), p)
    logger.info(f"  Inserted {len(players)} unique players.")

    # Insert player_season
    for ps in player_season_rows:
        db.execute(text(
            "INSERT INTO player_season (player_id, season_id, team_id, age, "
            "player_height, player_height_inches, player_weight, gp, pts, reb, ast, "
            "net_rating, oreb_pct, dreb_pct, usg_pct, ts_pct, ast_pct) "
            "VALUES (:player_id, :season_id, :team_id, :age, :player_height, "
            ":player_height_inches, :player_weight, :gp, :pts, :reb, :ast, "
            ":net_rating, :oreb_pct, :dreb_pct, :usg_pct, :ts_pct, :ast_pct)"
        ), ps)
    logger.info(f"  Inserted {len(player_season_rows)} player_season rows.")


def rebuild_games(db):
    """Fetch leaguegamelog with PlayerOrTeam=T to populate the game table.

    Two rows per game (one per team). Group by game_id, parse MATCHUP to
    determine home/away, use WL to determine winner/loser.
    """
    logger.info("Rebuilding game table...")
    db.execute(text("DELETE FROM game"))

    all_games = {}  # game_id → {home_team_id, away_team_id, winner_team_id, loser_team_id, season_id, date}

    for season_id, season_str in SEASONS_MAP.items():
        logger.info(f"  Fetching leaguegamelog (Team) for {season_str}...")
        params = {
            "Counter": 0,
            "Direction": "DESC",
            "LeagueID": "00",
            "PlayerOrTeam": "T",
            "Season": season_str,
            "SeasonType": "Regular Season",
            "Sorter": "DATE",
        }
        try:
            col, rows = _fetch_endpoint("leaguegamelog", params)
        except Exception as e:
            logger.error(f"  Failed to fetch {season_str}: {e}")
            time.sleep(1)
            continue

        for row in rows:
            gid = _parse_game_id(row[col["GAME_ID"]])
            team_id = row[col["TEAM_ID"]]
            matchup = row[col["MATCHUP"]]  # e.g. "BOS vs. CHA" or "CHA @ BOS"
            wl = row[col["WL"]]
            game_date = datetime.strptime(row[col["GAME_DATE"]], "%Y-%m-%d").date()

            if gid not in all_games:
                all_games[gid] = {
                    "game_id": gid,
                    "season_id": season_id,
                    "date": game_date,
                    "team_id_home": None,
                    "team_id_away": None,
                    "team_id_winner": None,
                    "team_id_loser": None,
                }

            game = all_games[gid]

            # "BOS vs. CHA" means BOS is home; "CHA @ BOS" means CHA is away
            if " vs. " in matchup:
                game["team_id_home"] = team_id
            elif " @ " in matchup:
                game["team_id_away"] = team_id

            if wl == "W":
                game["team_id_winner"] = team_id
            elif wl == "L":
                game["team_id_loser"] = team_id

        logger.info(f"    {len(rows)} team-game rows from {season_str}")
        time.sleep(1)

    # Insert games (skip any with incomplete data)
    inserted = 0
    skipped = 0
    for g in all_games.values():
        if all(g[k] is not None for k in ("team_id_home", "team_id_away", "team_id_winner", "team_id_loser")):
            db.execute(text(
                "INSERT INTO game (game_id, team_id_home, team_id_away, "
                "team_id_winner, team_id_loser, season_id, date) "
                "VALUES (:game_id, :team_id_home, :team_id_away, "
                ":team_id_winner, :team_id_loser, :season_id, :date)"
            ), g)
            inserted += 1
        else:
            skipped += 1

    logger.info(f"  Inserted {inserted} games ({skipped} skipped incomplete).")


def rebuild_player_game_logs(db):
    """Fetch leaguegamelog with PlayerOrTeam=P to populate player_game_log."""
    logger.info("Rebuilding player_game_log table...")
    db.execute(text("DELETE FROM player_game_log"))

    inserted = 0

    for season_id, season_str in SEASONS_MAP.items():
        logger.info(f"  Fetching leaguegamelog (Player) for {season_str}...")
        params = {
            "Counter": 0,
            "Direction": "DESC",
            "LeagueID": "00",
            "PlayerOrTeam": "P",
            "Season": season_str,
            "SeasonType": "Regular Season",
            "Sorter": "DATE",
        }
        try:
            col, rows = _fetch_endpoint("leaguegamelog", params)
        except Exception as e:
            logger.error(f"  Failed to fetch {season_str}: {e}")
            time.sleep(1)
            continue

        for row in rows:
            pid = row[col["PLAYER_ID"]]
            gid = _parse_game_id(row[col["GAME_ID"]])
            team_id = row[col["TEAM_ID"]]

            # Parse minutes
            min_raw = row[col["MIN"]]
            minutes = _parse_minutes(min_raw)

            db.execute(text(
                "INSERT INTO player_game_log "
                "(player_id, game_id, team_id, season_id, wl, min, "
                "fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, "
                "oreb, dreb, reb, ast, tov, stl, blk, blka, pf, pfd, "
                "pts, plus_minus, nba_fantasy_pts, dd2, td3) "
                "VALUES "
                "(:player_id, :game_id, :team_id, :season_id, :wl, :min, "
                ":fgm, :fga, :fg_pct, :fg3m, :fg3a, :fg3_pct, :ftm, :fta, :ft_pct, "
                ":oreb, :dreb, :reb, :ast, :tov, :stl, :blk, :blka, :pf, :pfd, "
                ":pts, :plus_minus, :nba_fantasy_pts, :dd2, :td3)"
            ), {
                "player_id": pid,
                "game_id": gid,
                "team_id": team_id,
                "season_id": season_id,
                "wl": row[col["WL"]],
                "min": minutes,
                "fgm": _float(row, col, "FGM"),
                "fga": _float(row, col, "FGA"),
                "fg_pct": _float(row, col, "FG_PCT"),
                "fg3m": _float(row, col, "FG3M"),
                "fg3a": _float(row, col, "FG3A"),
                "fg3_pct": _float(row, col, "FG3_PCT"),
                "ftm": _float(row, col, "FTM"),
                "fta": _float(row, col, "FTA"),
                "ft_pct": _float(row, col, "FT_PCT"),
                "oreb": _float(row, col, "OREB"),
                "dreb": _float(row, col, "DREB"),
                "reb": _float(row, col, "REB"),
                "ast": _float(row, col, "AST"),
                "tov": _float(row, col, "TOV"),
                "stl": _float(row, col, "STL"),
                "blk": _float(row, col, "BLK"),
                "blka": _float(row, col, "BLKA"),
                "pf": _float(row, col, "PF"),
                "pfd": _float(row, col, "PFD"),
                "pts": _float(row, col, "PTS"),
                "plus_minus": _float(row, col, "PLUS_MINUS"),
                "nba_fantasy_pts": _float(row, col, "FANTASY_PTS") or _float(row, col, "NBA_FANTASY_PTS"),
                "dd2": _float(row, col, "DD2"),
                "td3": _float(row, col, "TD3"),
            })
            inserted += 1

        logger.info(f"    {len(rows)} player-game rows from {season_str}")
        time.sleep(1)

    logger.info(f"  Inserted {inserted} player_game_log rows.")


def rebuild_player_traditional(db):
    """Fetch leaguedashplayerstats for each season → player_general_traditional_total."""
    logger.info("Rebuilding player_general_traditional_total table...")
    db.execute(text("DELETE FROM player_general_traditional_total"))

    inserted = 0

    for season_id, season_str in SEASONS_MAP.items():
        logger.info(f"  Fetching leaguedashplayerstats for {season_str}...")
        params = {
            "LastNGames": 0,
            "LeagueID": "00",
            "MeasureType": "Base",
            "Month": 0,
            "OpponentTeamID": 0,
            "PORound": 0,
            "PaceAdjust": "N",
            "PerMode": "Totals",
            "Period": 0,
            "PlusMinus": "N",
            "Rank": "Y",
            "Season": season_str,
            "SeasonType": "Regular Season",
            "TeamID": 0,
        }
        try:
            col, rows = _fetch_endpoint("leaguedashplayerstats", params)
        except Exception as e:
            logger.error(f"  Failed to fetch {season_str}: {e}")
            time.sleep(1)
            continue

        for row in rows:
            pid = row[col["PLAYER_ID"]]
            team_id = row[col["TEAM_ID"]]

            db.execute(text(
                "INSERT INTO player_general_traditional_total "
                "(player_id, season_id, team_id, age, gp, w, l, w_pct, min, "
                "fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, "
                "oreb, dreb, reb, ast, tov, stl, blk, blka, pf, pfd, pts, "
                "plus_minus, nba_fantasy_pts, dd2, td3, "
                "gp_rank, w_rank, l_rank, w_pct_rank, min_rank, "
                "fgm_rank, fga_rank, fg_pct_rank, fg3m_rank, fg3a_rank, fg3_pct_rank, "
                "ftm_rank, fta_rank, ft_pct_rank, oreb_rank, dreb_rank, reb_rank, "
                "ast_rank, tov_rank, stl_rank, blk_rank, blka_rank, pf_rank, pfd_rank, "
                "pts_rank, plus_minus_rank, nba_fantasy_pts_rank, dd2_rank, td3_rank, "
                "cfid, cfparams) "
                "VALUES "
                "(:player_id, :season_id, :team_id, :age, :gp, :w, :l, :w_pct, :min, "
                ":fgm, :fga, :fg_pct, :fg3m, :fg3a, :fg3_pct, :ftm, :fta, :ft_pct, "
                ":oreb, :dreb, :reb, :ast, :tov, :stl, :blk, :blka, :pf, :pfd, :pts, "
                ":plus_minus, :nba_fantasy_pts, :dd2, :td3, "
                ":gp_rank, :w_rank, :l_rank, :w_pct_rank, :min_rank, "
                ":fgm_rank, :fga_rank, :fg_pct_rank, :fg3m_rank, :fg3a_rank, :fg3_pct_rank, "
                ":ftm_rank, :fta_rank, :ft_pct_rank, :oreb_rank, :dreb_rank, :reb_rank, "
                ":ast_rank, :tov_rank, :stl_rank, :blk_rank, :blka_rank, :pf_rank, :pfd_rank, "
                ":pts_rank, :plus_minus_rank, :nba_fantasy_pts_rank, :dd2_rank, :td3_rank, "
                ":cfid, :cfparams)"
            ), {
                "player_id": pid,
                "season_id": season_id,
                "team_id": team_id,
                "age": _int(row, col, "AGE"),
                "gp": _int(row, col, "GP"),
                "w": _int(row, col, "W"),
                "l": _int(row, col, "L"),
                "w_pct": _float(row, col, "W_PCT"),
                "min": _float(row, col, "MIN"),
                "fgm": _float(row, col, "FGM"),
                "fga": _float(row, col, "FGA"),
                "fg_pct": _float(row, col, "FG_PCT"),
                "fg3m": _float(row, col, "FG3M"),
                "fg3a": _float(row, col, "FG3A"),
                "fg3_pct": _float(row, col, "FG3_PCT"),
                "ftm": _float(row, col, "FTM"),
                "fta": _float(row, col, "FTA"),
                "ft_pct": _float(row, col, "FT_PCT"),
                "oreb": _float(row, col, "OREB"),
                "dreb": _float(row, col, "DREB"),
                "reb": _float(row, col, "REB"),
                "ast": _float(row, col, "AST"),
                "tov": _float(row, col, "TOV"),
                "stl": _float(row, col, "STL"),
                "blk": _float(row, col, "BLK"),
                "blka": _float(row, col, "BLKA"),
                "pf": _float(row, col, "PF"),
                "pfd": _float(row, col, "PFD"),
                "pts": _float(row, col, "PTS"),
                "plus_minus": _float(row, col, "PLUS_MINUS"),
                "nba_fantasy_pts": _float(row, col, "NBA_FANTASY_PTS"),
                "dd2": _float(row, col, "DD2"),
                "td3": _float(row, col, "TD3"),
                "gp_rank": _int(row, col, "GP_RANK"),
                "w_rank": _int(row, col, "W_RANK"),
                "l_rank": _int(row, col, "L_RANK"),
                "w_pct_rank": _int(row, col, "W_PCT_RANK"),
                "min_rank": _int(row, col, "MIN_RANK"),
                "fgm_rank": _int(row, col, "FGM_RANK"),
                "fga_rank": _int(row, col, "FGA_RANK"),
                "fg_pct_rank": _int(row, col, "FG_PCT_RANK"),
                "fg3m_rank": _int(row, col, "FG3M_RANK"),
                "fg3a_rank": _int(row, col, "FG3A_RANK"),
                "fg3_pct_rank": _int(row, col, "FG3_PCT_RANK"),
                "ftm_rank": _int(row, col, "FTM_RANK"),
                "fta_rank": _int(row, col, "FTA_RANK"),
                "ft_pct_rank": _int(row, col, "FT_PCT_RANK"),
                "oreb_rank": _int(row, col, "OREB_RANK"),
                "dreb_rank": _int(row, col, "DREB_RANK"),
                "reb_rank": _int(row, col, "REB_RANK"),
                "ast_rank": _int(row, col, "AST_RANK"),
                "tov_rank": _int(row, col, "TOV_RANK"),
                "stl_rank": _int(row, col, "STL_RANK"),
                "blk_rank": _int(row, col, "BLK_RANK"),
                "blka_rank": _int(row, col, "BLKA_RANK"),
                "pf_rank": _int(row, col, "PF_RANK"),
                "pfd_rank": _int(row, col, "PFD_RANK"),
                "pts_rank": _int(row, col, "PTS_RANK"),
                "plus_minus_rank": _int(row, col, "PLUS_MINUS_RANK"),
                "nba_fantasy_pts_rank": _int(row, col, "NBA_FANTASY_PTS_RANK"),
                "dd2_rank": _int(row, col, "DD2_RANK"),
                "td3_rank": _int(row, col, "TD3_RANK"),
                "cfid": _int(row, col, "CFID"),
                "cfparams": str(row[col["CFPARAMS"]]) if "CFPARAMS" in col and row[col["CFPARAMS"]] is not None else None,
            })
            inserted += 1

        logger.info(f"    {len(rows)} rows from {season_str}")
        time.sleep(1)

    logger.info(f"  Inserted {inserted} player_general_traditional_total rows.")


def drop_empty_tables(db):
    """Drop the three empty tables: play_by_play, play_by_playv3, shot_chart_detail."""
    logger.info("Dropping empty tables...")
    for table in ("play_by_play", "play_by_playv3", "shot_chart_detail"):
        db.execute(text(f"DROP TABLE IF EXISTS {table}"))
        logger.info(f"  Dropped {table}")


def rebuild_all(db):
    """Orchestrate full rebuild: truncate → repopulate → drop empties.

    Order matters due to FK constraints:
      1. Delete from child tables first (player_game_log, player_general_traditional_total, player_season)
      2. Delete from game, player
      3. Re-insert in parent→child order
    """
    logger.info("=" * 60)
    logger.info("Starting full NBA stats rebuild")
    logger.info("=" * 60)

    # Disable FK checks for clean truncation
    db.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

    # Drop empty tables first (they have FKs to game/player/team)
    drop_empty_tables(db)

    # Rebuild in order
    rebuild_seasons(db)
    rebuild_players_and_player_seasons(db)
    rebuild_games(db)
    rebuild_player_game_logs(db)
    rebuild_player_traditional(db)

    # Re-enable FK checks
    db.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    logger.info("=" * 60)
    logger.info("Rebuild complete!")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _float(row, col, key):
    """Safely extract a float from a row by column name."""
    if key not in col:
        return None
    val = row[col[key]]
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _int(row, col, key):
    """Safely extract an int from a row by column name."""
    if key not in col:
        return None
    val = row[col[key]]
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


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
