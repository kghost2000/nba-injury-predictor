"""Reddit scraper for NBA injury signals using PRAW."""

import logging
from datetime import datetime

import praw

from agents.injury_parser import InjuryParser
from scrapers.base import BaseScraper
from utils.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

logger = logging.getLogger(__name__)

INJURY_KEYWORDS = [
    "injury", "out", "questionable", "doubtful", "probable", "ruled out",
    "will not play", "day-to-day", "mri", "sprain", "strain", "surgery",
    "concussion", "sidelined", "miss", "torn", "fracture",
]

SUBREDDITS = ["nba"]

# Number of top comments to include per post
TOP_COMMENTS_LIMIT = 5


class RedditScraper(BaseScraper):
    """Scrapes NBA injury discussion from Reddit using PRAW."""

    SOURCE = "reddit"

    def __init__(self, injury_parser=None):
        super().__init__()
        self.injury_parser = injury_parser
        self.reddit = None
        self._init_reddit()

    def _init_reddit(self):
        """Initialize the PRAW Reddit client."""
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            logger.warning(
                "Reddit API credentials not configured. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env"
            )
            return

        try:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            logger.info("Reddit client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")

    def scrape(self):
        """Scrape NBA subreddits for injury-related posts.

        Returns:
            list of dicts with raw_text, reporter_name, timestamp, source_type.
        """
        reports = []
        if not self.reddit:
            logger.warning("Reddit client not initialized. Skipping Reddit scraping.")
            return reports

        for subreddit_name in SUBREDDITS:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                # Search for injury-related posts
                search_reports = self._search_injury_posts(subreddit)
                reports.extend(search_reports)

                # Also check hot posts for injury keywords
                hot_reports = self._check_hot_posts(subreddit)
                reports.extend(hot_reports)

                logger.info(
                    f"r/{subreddit_name}: {len(search_reports)} from search, "
                    f"{len(hot_reports)} from hot"
                )
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")

        # Deduplicate by raw_text
        seen = set()
        unique = []
        for r in reports:
            key = r["raw_text"][:200]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        logger.info(f"Total unique Reddit reports collected: {len(unique)}")
        return unique

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

        logger.info(f"Saved {len(signals)}/{len(reports)} Reddit signals to database.")
        return len(signals)

    def _search_injury_posts(self, subreddit):
        """Search subreddit for recent injury-related posts."""
        reports = []
        try:
            for submission in subreddit.search(
                "injury OR questionable OR ruled out OR day-to-day",
                sort="new",
                time_filter="day",
                limit=25,
            ):
                report = self._submission_to_report(submission)
                if report:
                    reports.append(report)
        except Exception as e:
            logger.error(f"Error searching subreddit: {e}")
        return reports

    def _check_hot_posts(self, subreddit):
        """Check hot posts for injury-related content."""
        reports = []
        try:
            for submission in subreddit.hot(limit=50):
                title_lower = submission.title.lower()
                if any(kw in title_lower for kw in INJURY_KEYWORDS):
                    report = self._submission_to_report(submission)
                    if report:
                        reports.append(report)
        except Exception as e:
            logger.error(f"Error fetching hot posts: {e}")
        return reports

    def _submission_to_report(self, submission):
        """Convert a Reddit submission to a report dict."""
        # Build text from title + body + top comments
        parts = [submission.title]
        if submission.selftext:
            parts.append(submission.selftext[:1000])

        try:
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:TOP_COMMENTS_LIMIT]:
                parts.append(comment.body[:500])
        except Exception:
            pass

        raw_text = "\n\n".join(parts)

        # Check that the combined text actually contains injury keywords
        text_lower = raw_text.lower()
        if not any(kw in text_lower for kw in INJURY_KEYWORDS):
            return None

        timestamp = datetime.utcfromtimestamp(submission.created_utc)

        return {
            "raw_text": raw_text,
            "reporter_name": f"u/{submission.author.name}" if submission.author else "reddit_user",
            "timestamp": timestamp,
            "source_type": self.SOURCE,
        }
