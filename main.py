"""NBA Injury Prediction - Main Entry Point.

Orchestrates structured and unstructured scrapers to collect injury data
from NBA.com, RotoWire, Twitter (Nitter), and Reddit, storing results in SQLite.
"""

import argparse
import logging
import sys

from database.schema import get_session, init_db
from scrapers.structured.nba_official import NBAOfficialScraper
from scrapers.structured.rotowire import RotoWireScraper
from utils.config import ANTHROPIC_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_structured_scrapers():
    """Run all structured scrapers and save results to the database."""
    logger.info("Initializing database...")
    init_db()

    scrapers = [
        ("NBA.com", NBAOfficialScraper()),
        ("RotoWire", RotoWireScraper()),
    ]

    total_saved = 0

    for name, scraper in scrapers:
        logger.info(f"--- Running {name} scraper ---")
        try:
            reports = scraper.scrape()
            logger.info(f"{name}: scraped {len(reports)} reports")

            if reports:
                with get_session() as session:
                    saved = scraper.save_reports(session, reports)
                    total_saved += saved
        except Exception as e:
            logger.error(f"{name} scraper failed: {e}")

    logger.info(f"Finished structured scrapers. Total reports saved: {total_saved}")
    return total_saved


def run_unstructured_scrapers():
    """Run unstructured scrapers (Twitter/Reddit) with Claude extraction."""
    if not ANTHROPIC_API_KEY:
        logger.warning(
            "ANTHROPIC_API_KEY not set. Skipping unstructured scrapers. "
            "Set it in your .env file to enable Claude-powered extraction."
        )
        return 0

    from agents.claude_client import ClaudeClient
    from agents.injury_parser import InjuryParser
    from scrapers.unstructured.twitter import NitterScraper

    logger.info("Initializing database...")
    init_db()

    claude_client = ClaudeClient()
    parser = InjuryParser(claude_client)

    total_saved = 0

    # Twitter via Nitter
    logger.info("--- Running Nitter (Twitter) scraper ---")
    try:
        nitter = NitterScraper(injury_parser=parser)
        reports = nitter.scrape()
        logger.info(f"Nitter: scraped {len(reports)} injury tweets")
        if reports:
            with get_session() as session:
                saved = nitter.save_reports(session, reports)
                total_saved += saved
    except Exception as e:
        logger.error(f"Nitter scraper failed: {e}")

    # Reddit
    logger.info("--- Running Reddit scraper ---")
    try:
        from scrapers.unstructured.reddit import RedditScraper

        reddit = RedditScraper(injury_parser=parser)
        reports = reddit.scrape()
        logger.info(f"Reddit: scraped {len(reports)} injury posts")
        if reports:
            with get_session() as session:
                saved = reddit.save_reports(session, reports)
                total_saved += saved
    except ImportError:
        logger.warning("praw not installed. Skipping Reddit scraper. Run: pip install praw")
    except Exception as e:
        logger.error(f"Reddit scraper failed: {e}")

    logger.info(f"Finished unstructured scrapers. Total signals saved: {total_saved}")
    return total_saved


def run_backfill(season, interval_days=1):
    """Backfill historical injury data for a given season using nbainjuries."""
    from scrapers.structured.nba_historical import backfill_season

    logger.info("Initializing database...")
    init_db()

    with get_session() as session:
        saved = backfill_season(session, season, interval_days=interval_days)

    logger.info(f"Backfill finished. {saved} records saved.")
    return saved


def run_outcomes(season):
    """Fetch game outcomes and player box score stats for a season."""
    from scrapers.structured.game_outcomes import fetch_season

    logger.info("Initializing database...")
    init_db()

    with get_session() as session:
        saved = fetch_season(session, season)

    logger.info(f"Game outcomes finished. {saved} records saved.")
    return saved


def run_seed():
    """Seed the reporters table with known NBA insiders."""
    from database.seed import seed_reporters

    logger.info("Initializing database...")
    init_db()

    with get_session() as session:
        seed_reporters(session)


def run_rebuild():
    """Wipe and rebuild core NBA stats tables from the NBA Stats API."""
    from scrapers.structured.nba_stats_rebuild import rebuild_all

    logger.info("Initializing database...")
    init_db()

    with get_session() as session:
        rebuild_all(session)


def run_train():
    """Train game availability prediction models."""
    from models.game_availability import train_and_evaluate

    logger.info("Initializing database...")
    init_db()
    train_and_evaluate()


def run_injury_risk():
    """Train injury risk early warning models."""
    from models.injury_risk import train_and_evaluate

    logger.info("Initializing database...")
    init_db()
    train_and_evaluate()


def run_predict(date=None):
    """Run daily batch predictions."""
    from pipeline.daily_predictions import run as run_predictions

    logger.info("Initializing database...")
    init_db()
    run_predictions(prediction_date=date)


def run_validate(date=None, lag=3, threshold=0.07):
    """Validate predictions against actual outcomes."""
    from pipeline.validate_outcomes import run as run_validation

    logger.info("Initializing database...")
    init_db()
    run_validation(prediction_date=date, lag_days=lag, threshold=threshold)


def run_migrate():
    """Run database migrations (create prediction tracking tables)."""
    from pathlib import Path

    from sqlalchemy import text

    from database.schema import get_engine

    engine = get_engine()
    migrations_dir = Path(__file__).parent / "database" / "migrations"

    with engine.begin() as conn:
        for sql_file in sorted(migrations_dir.glob("*.sql")):
            logger.info(f"Running migration: {sql_file.name}")
            sql = sql_file.read_text()
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            logger.info(f"  Completed: {sql_file.name}")

    logger.info("All migrations complete.")


def run_all():
    """Run all scrapers (structured + unstructured)."""
    structured = run_structured_scrapers()
    unstructured = run_unstructured_scrapers()
    logger.info(
        f"All done. Structured: {structured} reports, "
        f"Unstructured: {unstructured} signals."
    )
    return structured + unstructured


def main():
    """CLI entry point with argument parsing."""
    arg_parser = argparse.ArgumentParser(
        description="NBA Injury Prediction - Data Collection Pipeline"
    )
    arg_parser.add_argument(
        "--structured", action="store_true",
        help="Run structured scrapers only (NBA.com, RotoWire)",
    )
    arg_parser.add_argument(
        "--unstructured", action="store_true",
        help="Run unstructured scrapers only (Twitter, Reddit)",
    )
    arg_parser.add_argument(
        "--seed", action="store_true",
        help="Seed the reporters table with known NBA insiders",
    )
    arg_parser.add_argument(
        "--backfill", type=str, metavar="SEASON",
        help="Backfill historical injury data for a season (e.g. 2024-25)",
    )
    arg_parser.add_argument(
        "--interval", type=int, default=1,
        help="Days between samples when backfilling (default: 1 = daily)",
    )
    arg_parser.add_argument(
        "--outcomes", type=str, metavar="SEASON",
        help="Fetch game outcomes and player stats for a season (e.g. 2024-25)",
    )
    arg_parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe and rebuild core NBA stats tables (season, player, game, etc.) from the NBA Stats API",
    )
    arg_parser.add_argument(
        "--train", action="store_true",
        help="Train game availability prediction models",
    )
    arg_parser.add_argument(
        "--injury-risk", action="store_true",
        help="Train injury risk early warning models",
    )
    arg_parser.add_argument(
        "--predict", action="store_true",
        help="Run daily batch predictions (writes to daily_predictions table)",
    )
    arg_parser.add_argument(
        "--predict-date", type=str, metavar="DATE",
        help="Prediction date (YYYY-MM-DD) for --predict. Defaults to today.",
    )
    arg_parser.add_argument(
        "--validate", action="store_true",
        help="Validate predictions against actual outcomes",
    )
    arg_parser.add_argument(
        "--validate-date", type=str, metavar="DATE",
        help="Prediction date to validate (YYYY-MM-DD). Defaults to 3 days ago.",
    )
    arg_parser.add_argument(
        "--migrate", action="store_true",
        help="Run database migrations (create prediction tracking tables)",
    )

    args = arg_parser.parse_args()

    if args.migrate:
        run_migrate()
    elif args.predict:
        run_predict(date=args.predict_date)
    elif args.validate:
        run_validate(date=args.validate_date)
    elif args.injury_risk:
        run_injury_risk()
    elif args.train:
        run_train()
    elif args.rebuild:
        run_rebuild()
    elif args.outcomes:
        run_outcomes(args.outcomes)
    elif args.backfill:
        run_backfill(args.backfill, interval_days=args.interval)
    elif args.seed:
        run_seed()
    elif args.structured:
        run_structured_scrapers()
    elif args.unstructured:
        run_unstructured_scrapers()
    else:
        run_all()


if __name__ == "__main__":
    main()
