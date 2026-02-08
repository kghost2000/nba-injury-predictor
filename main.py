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


def run_seed():
    """Seed the reporters table with known NBA insiders."""
    from database.seed import seed_reporters

    logger.info("Initializing database...")
    init_db()

    with get_session() as session:
        seed_reporters(session)


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

    args = arg_parser.parse_args()

    if args.backfill:
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
