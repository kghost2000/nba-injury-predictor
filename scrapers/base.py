import logging
import random
import time
from abc import ABC, abstractmethod

import requests

from utils.config import (
    MAX_RETRIES,
    REQUEST_DELAY_MAX,
    REQUEST_DELAY_MIN,
    REQUEST_TIMEOUT,
    USER_AGENTS,
)

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all scrapers with shared utilities."""

    def __init__(self):
        self.session = requests.Session()
        self._update_headers()

    def _update_headers(self):
        """Set a random user agent on the requests session."""
        self.session.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

    def _rate_limit(self):
        """Sleep for a random interval between requests."""
        delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
        logger.debug(f"Rate limiting: sleeping {delay:.2f}s")
        time.sleep(delay)

    def fetch(self, url, **kwargs):
        """Fetch a URL with retry logic and exponential backoff.

        Returns the Response object on success, or None after all retries fail.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._update_headers()
                response = self.session.get(
                    url, timeout=REQUEST_TIMEOUT, **kwargs
                )
                response.raise_for_status()
                logger.info(f"Successfully fetched {url} (attempt {attempt})")
                return response
            except requests.RequestException as e:
                wait = 2 ** attempt + random.uniform(0, 1)
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        logger.error(f"All {MAX_RETRIES} attempts failed for {url}")
        return None

    @abstractmethod
    def scrape(self):
        """Scrape data and return a list of dicts."""

    @abstractmethod
    def save_reports(self, db_session, reports):
        """Persist scraped reports to the database."""

    def _validate_report(self, report, required_fields):
        """Check that a report dict has all required fields with non-empty values."""
        for field in required_fields:
            if not report.get(field):
                logger.warning(f"Report missing required field '{field}': {report}")
                return False
        return True
