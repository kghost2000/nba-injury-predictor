"""Tests for unstructured scrapers (Phase 5)."""

from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, InjurySignal
from scrapers.unstructured.twitter import NitterScraper, INJURY_KEYWORDS


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# NitterScraper tests
# ---------------------------------------------------------------------------

SAMPLE_NITTER_HTML = """
<html><body>
<div class="timeline-item">
    <div class="tweet-content">LeBron James ruled out tonight with ankle soreness. Lakers list him as OUT on the injury report.</div>
    <span class="tweet-date"><a title="Jan 15, 2024 · 03:30 PM UTC">Jan 15</a></span>
</div>
<div class="timeline-item">
    <div class="tweet-content">Great trade deadline coverage coming up on ESPN at 7pm!</div>
    <span class="tweet-date"><a title="Jan 15, 2024 · 02:00 PM UTC">Jan 15</a></span>
</div>
<div class="timeline-item">
    <div class="tweet-content">Steph Curry is questionable for tomorrow's game. MRI on his knee came back clean.</div>
    <span class="tweet-date"><a title="Jan 15, 2024 · 01:00 PM UTC">Jan 15</a></span>
</div>
</body></html>
"""


class TestNitterScraper:
    def test_parse_tweets(self):
        """Test that tweets are extracted from Nitter HTML."""
        scraper = NitterScraper()
        tweets = scraper._parse_tweets(SAMPLE_NITTER_HTML, "wojespn")

        assert len(tweets) == 3
        assert tweets[0]["reporter_name"] == "@wojespn"
        assert tweets[0]["source_type"] == "tweet"
        assert "LeBron James" in tweets[0]["raw_text"]
        assert "ruled out" in tweets[0]["raw_text"]

    def test_filter_injury_tweets(self):
        """Test that only injury-related tweets pass the keyword filter."""
        scraper = NitterScraper()
        tweets = scraper._parse_tweets(SAMPLE_NITTER_HTML, "wojespn")
        filtered = scraper._filter_injury_tweets(tweets)

        # Should keep tweet 1 (ruled out, ankle, OUT, injury) and tweet 3 (questionable, MRI)
        # Should skip tweet 2 (trade deadline coverage)
        assert len(filtered) == 2
        assert "LeBron" in filtered[0]["raw_text"]
        assert "Steph Curry" in filtered[1]["raw_text"]

    @patch.object(NitterScraper, "fetch")
    @patch.object(NitterScraper, "_find_working_instance")
    @patch.object(NitterScraper, "_rate_limit")
    def test_scrape_fetches_all_accounts(self, mock_rate, mock_instance, mock_fetch):
        """Test that scrape attempts to fetch all tracked accounts."""
        mock_instance.return_value = "https://nitter.test"
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NITTER_HTML
        mock_fetch.return_value = mock_response

        scraper = NitterScraper()
        reports = scraper.scrape()

        # Each tracked account should produce filtered tweets
        assert len(reports) > 0
        assert mock_fetch.call_count > 0

    @patch.object(NitterScraper, "_find_working_instance")
    def test_scrape_returns_empty_when_no_instance(self, mock_instance):
        """Test graceful handling when no Nitter instance is available."""
        mock_instance.return_value = None
        scraper = NitterScraper()
        reports = scraper.scrape()
        assert reports == []

    def test_instance_failover(self):
        """Test that _find_working_instance tries multiple instances."""
        scraper = NitterScraper()

        responses = [
            Exception("Connection refused"),
            Exception("Timeout"),
            MagicMock(status_code=200),
        ]
        with patch.object(scraper.session, "get", side_effect=responses):
            instance = scraper._find_working_instance()

        # Should return the third instance (first two fail)
        assert instance is not None

    def test_instance_failover_all_fail(self):
        """Test that None is returned when all instances fail."""
        scraper = NitterScraper()

        with patch.object(scraper.session, "get", side_effect=Exception("fail")):
            instance = scraper._find_working_instance()

        assert instance is None

    def test_save_reports_with_parser(self, db_session):
        """Test save_reports delegates to InjuryParser and saves results."""
        mock_parser = MagicMock()
        mock_signal = InjurySignal(
            player_name="LeBron James",
            source_type="tweet",
            raw_text="LeBron out tonight",
            confidence_score=0.9,
            timestamp=datetime(2024, 1, 15),
        )
        mock_parser.parse_batch.return_value = [mock_signal]

        scraper = NitterScraper(injury_parser=mock_parser)
        reports = [{"raw_text": "LeBron out", "source_type": "tweet"}]
        saved = scraper.save_reports(db_session, reports)
        db_session.commit()

        assert saved == 1
        assert db_session.query(InjurySignal).count() == 1

    def test_save_reports_without_parser(self, db_session):
        """Test save_reports returns 0 when no parser is configured."""
        scraper = NitterScraper()
        saved = scraper.save_reports(db_session, [{"raw_text": "test"}])
        assert saved == 0


# ---------------------------------------------------------------------------
# RedditScraper tests
# ---------------------------------------------------------------------------

class TestRedditScraper:
    @patch("scrapers.unstructured.reddit.praw.Reddit")
    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_ID", "test_id")
    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_SECRET", "test_secret")
    def test_scrape_returns_empty_when_no_results(self, mock_reddit_cls):
        """Test that scrape returns empty list when no posts found."""
        from scrapers.unstructured.reddit import RedditScraper

        mock_reddit = MagicMock()
        mock_reddit_cls.return_value = mock_reddit
        mock_subreddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_subreddit.search.return_value = []
        mock_subreddit.hot.return_value = []

        scraper = RedditScraper()
        reports = scraper.scrape()
        assert reports == []

    @patch("scrapers.unstructured.reddit.praw.Reddit")
    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_ID", "test_id")
    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_SECRET", "test_secret")
    def test_scrape_hot_posts_keyword_filter(self, mock_reddit_cls):
        """Test that hot posts are filtered by injury keywords."""
        from scrapers.unstructured.reddit import RedditScraper

        mock_reddit = MagicMock()
        mock_reddit_cls.return_value = mock_reddit
        mock_subreddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_subreddit.search.return_value = []

        # Create mock submissions
        injury_post = MagicMock()
        injury_post.title = "LeBron James injury update: ruled out for tonight"
        injury_post.selftext = "Source: Woj reports LeBron has ankle soreness"
        injury_post.created_utc = 1705334400.0  # Jan 15 2024
        injury_post.author.name = "nba_fan"
        injury_post.comments.replace_more.return_value = None
        injury_post.comments.__iter__ = MagicMock(return_value=iter([]))
        injury_post.comments.__getitem__ = MagicMock(return_value=[])

        non_injury_post = MagicMock()
        non_injury_post.title = "Best plays of the week compilation"
        non_injury_post.selftext = ""
        non_injury_post.created_utc = 1705334400.0

        mock_subreddit.hot.return_value = [injury_post, non_injury_post]

        scraper = RedditScraper()
        reports = scraper.scrape()

        # Only the injury post should pass
        assert len(reports) == 1
        assert "LeBron James" in reports[0]["raw_text"]

    @patch("scrapers.unstructured.reddit.praw.Reddit")
    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_ID", "test_id")
    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_SECRET", "test_secret")
    def test_submission_includes_comments(self, mock_reddit_cls):
        """Test that top comments are included in raw_text."""
        from scrapers.unstructured.reddit import RedditScraper

        mock_reddit = MagicMock()
        mock_reddit_cls.return_value = mock_reddit

        post = MagicMock()
        post.title = "Giannis injury: sprain confirmed"
        post.selftext = "Just announced by the Bucks."
        post.created_utc = 1705334400.0
        post.author.name = "bucks_fan"

        comment1 = MagicMock()
        comment1.body = "Hope he recovers quickly"
        comment2 = MagicMock()
        comment2.body = "Out at least 2 weeks"
        post.comments.replace_more.return_value = None
        post.comments.__getitem__ = lambda self, s: [comment1, comment2][s] if isinstance(s, int) else [comment1, comment2][:s.stop]

        scraper = RedditScraper()
        report = scraper._submission_to_report(post)

        assert report is not None
        assert "Giannis" in report["raw_text"]
        assert "sprain" in report["raw_text"]

    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_ID", "")
    @patch("scrapers.unstructured.reddit.REDDIT_CLIENT_SECRET", "")
    def test_scrape_without_credentials(self):
        """Test graceful handling when Reddit credentials are missing."""
        from scrapers.unstructured.reddit import RedditScraper

        scraper = RedditScraper()
        assert scraper.reddit is None
        reports = scraper.scrape()
        assert reports == []

    def test_save_reports_with_parser(self, db_session):
        """Test save_reports delegates to InjuryParser."""
        from scrapers.unstructured.reddit import RedditScraper

        mock_parser = MagicMock()
        mock_signal = InjurySignal(
            player_name="Giannis",
            source_type="reddit",
            raw_text="Giannis out",
            confidence_score=0.7,
            timestamp=datetime(2024, 1, 15),
        )
        mock_parser.parse_batch.return_value = [mock_signal]

        scraper = RedditScraper.__new__(RedditScraper)
        scraper.injury_parser = mock_parser

        reports = [{"raw_text": "Giannis out", "source_type": "reddit"}]
        saved = scraper.save_reports(db_session, reports)
        db_session.commit()

        assert saved == 1
        assert db_session.query(InjurySignal).count() == 1

    def test_save_reports_without_parser(self, db_session):
        """Test save_reports returns 0 when no parser is configured."""
        from scrapers.unstructured.reddit import RedditScraper

        scraper = RedditScraper.__new__(RedditScraper)
        scraper.injury_parser = None

        saved = scraper.save_reports(db_session, [{"raw_text": "test"}])
        assert saved == 0
