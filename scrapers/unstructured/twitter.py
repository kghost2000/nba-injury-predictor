"""Twitter/X scraper for NBA injury signals via Nitter mirrors."""

import logging
from datetime import datetime

from bs4 import BeautifulSoup

from agents.injury_parser import InjuryParser
from scrapers.base import BaseScraper
from utils.config import NITTER_INSTANCES

logger = logging.getLogger(__name__)

INJURY_KEYWORDS = [
    "injury", "out", "questionable", "doubtful", "probable", "ruled out",
    "will not play", "day-to-day", "mri", "sprain", "strain", "surgery",
    "concussion", "sidelined", "miss", "torn", "fracture",
]

TRACKED_ACCOUNTS = [
    "wojespn",
    "ShamsCharania",
    "ChrisBHaynes",
    "TheSteinLine",
    "KeithSmithNBA",
    "MarcJSpears",
    "JakeLFischer",
    "WindhorstESPN",
]


class NitterScraper(BaseScraper):
    """Scrapes NBA injury tweets from Nitter (Twitter mirror)."""

    SOURCE = "tweet"

    def __init__(self, injury_parser=None):
        super().__init__()
        self.injury_parser = injury_parser
        self._working_instance = None

    def scrape(self):
        """Scrape tracked accounts for injury-related tweets.

        Returns:
            list of dicts with raw_text, reporter_name, timestamp, source_type.
        """
        reports = []
        instance = self._find_working_instance()
        if not instance:
            logger.warning("No working Nitter instance found. Skipping Twitter scraping.")
            return reports

        for handle in TRACKED_ACCOUNTS:
            url = f"{instance}/{handle}"
            logger.info(f"Fetching tweets from {url}")
            self._rate_limit()
            response = self.fetch(url)

            if response is None:
                logger.warning(f"Failed to fetch tweets for @{handle}")
                continue

            try:
                tweets = self._parse_tweets(response.text, handle)
                filtered = self._filter_injury_tweets(tweets)
                reports.extend(filtered)
                logger.info(
                    f"@{handle}: {len(tweets)} tweets found, "
                    f"{len(filtered)} injury-related"
                )
            except Exception as e:
                logger.error(f"Error parsing tweets for @{handle}: {e}")

        logger.info(f"Total injury tweets collected: {len(reports)}")
        return reports

    def save_reports(self, db_session, reports):
        """Save reports to DB using InjuryParser for Claude extraction.

        Returns:
            Number of signals saved.
        """
        if not self.injury_parser:
            logger.warning("No InjuryParser configured, cannot save reports.")
            return 0

        signals = self.injury_parser.parse_batch(reports, db_session=db_session)
        for signal in signals:
            db_session.add(signal)

        logger.info(f"Saved {len(signals)}/{len(reports)} tweet signals to database.")
        return len(signals)

    def _find_working_instance(self):
        """Try each Nitter instance and return the first one that responds."""
        if self._working_instance:
            return self._working_instance

        for instance in NITTER_INSTANCES:
            try:
                response = self.session.get(instance, timeout=10)
                if response.status_code < 500:
                    logger.info(f"Found working Nitter instance: {instance}")
                    self._working_instance = instance
                    return instance
            except Exception:
                logger.debug(f"Nitter instance {instance} is not responding.")
                continue

        return None

    def _parse_tweets(self, html, handle):
        """Extract tweet text and timestamps from Nitter HTML.

        Returns:
            list of dicts with raw_text, reporter_name, timestamp, source_type.
        """
        soup = BeautifulSoup(html, "lxml")
        tweets = []

        # Nitter renders tweets in .timeline-item containers
        tweet_containers = soup.find_all("div", class_="timeline-item")

        for container in tweet_containers:
            content_el = container.find("div", class_="tweet-content")
            if not content_el:
                continue

            raw_text = content_el.get_text(strip=True)
            if not raw_text:
                continue

            # Extract timestamp
            timestamp = datetime.utcnow()
            time_el = container.find("span", class_="tweet-date")
            if time_el:
                link = time_el.find("a")
                if link and link.get("title"):
                    try:
                        timestamp = datetime.strptime(
                            link["title"], "%b %d, %Y · %I:%M %p %Z"
                        )
                    except ValueError:
                        pass

            tweets.append({
                "raw_text": raw_text,
                "reporter_name": f"@{handle}",
                "timestamp": timestamp,
                "source_type": self.SOURCE,
            })

        return tweets

    def _filter_injury_tweets(self, tweets):
        """Filter tweets to only those containing injury-related keywords."""
        filtered = []
        for tweet in tweets:
            text_lower = tweet["raw_text"].lower()
            if any(kw in text_lower for kw in INJURY_KEYWORDS):
                filtered.append(tweet)
        return filtered
