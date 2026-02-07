"""NBA Injury Prediction - Main Entry Point.

Orchestrates the structured scrapers to collect injury report data
from NBA.com and RotoWire, storing results in SQLite.
"""

import logging
import sys

from database.schema import get_session, init_db
from scrapers.structured.nba_official import NBAOfficialScraper
from scrapers.structured.rotowire import RotoWireScraper

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

    logger.info(f"Finished. Total reports saved: {total_saved}")
    return total_saved


if __name__ == "__main__":
    run_structured_scrapers()
