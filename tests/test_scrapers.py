from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, InjuryReport
from scrapers.base import BaseScraper
from scrapers.structured.nba_official import NBAOfficialScraper
from scrapers.structured.rotowire import RotoWireScraper


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestBaseScraper:
    def test_validate_report_valid(self):
        """Concrete subclass for testing the base class."""
        class DummyScraper(BaseScraper):
            def scrape(self):
                return []
            def save_reports(self, db_session, reports):
                pass

        scraper = DummyScraper()
        report = {"player_name": "Test", "team": "Team", "injury_status": "Out"}
        assert scraper._validate_report(report, ["player_name", "team", "injury_status"])

    def test_validate_report_missing_field(self):
        class DummyScraper(BaseScraper):
            def scrape(self):
                return []
            def save_reports(self, db_session, reports):
                pass

        scraper = DummyScraper()
        report = {"player_name": "Test", "team": ""}
        assert not scraper._validate_report(report, ["player_name", "team", "injury_status"])

    def test_validate_report_empty_value(self):
        class DummyScraper(BaseScraper):
            def scrape(self):
                return []
            def save_reports(self, db_session, reports):
                pass

        scraper = DummyScraper()
        report = {"player_name": "", "team": "Team", "injury_status": "Out"}
        assert not scraper._validate_report(report, ["player_name", "team"])


class TestNBAOfficialScraper:
    def test_extract_status(self):
        scraper = NBAOfficialScraper()
        assert scraper._extract_status("Player is listed as Out") == "Out"
        assert scraper._extract_status("Status: Questionable") == "Questionable"
        assert scraper._extract_status("Probable for tonight") == "Probable"
        assert scraper._extract_status("No status info") == "Unknown"

    def test_save_reports(self, db_session):
        scraper = NBAOfficialScraper()
        reports = [
            {
                "player_name": "LeBron James",
                "team": "Lakers",
                "injury_status": "Questionable",
                "injury_description": "Ankle",
                "report_date": datetime(2024, 1, 15),
                "source": "nba.com",
            },
        ]
        saved = scraper.save_reports(db_session, reports)
        db_session.commit()

        assert saved == 1
        assert db_session.query(InjuryReport).count() == 1
        result = db_session.query(InjuryReport).first()
        assert result.player_name == "LeBron James"

    @patch.object(NBAOfficialScraper, "fetch")
    def test_scrape_handles_failed_fetch(self, mock_fetch):
        mock_fetch.return_value = None
        scraper = NBAOfficialScraper()
        reports = scraper.scrape()
        assert reports == []

    @patch.object(NBAOfficialScraper, "fetch")
    def test_scrape_parses_table_html(self, mock_fetch):
        html = """
        <html><body>
        <table>
            <tr>
                <td>Stephen Curry</td>
                <td>Golden State Warriors</td>
                <td>Out</td>
                <td>Right knee soreness</td>
            </tr>
        </table>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_fetch.return_value = mock_response

        scraper = NBAOfficialScraper()
        reports = scraper.scrape()

        assert len(reports) == 1
        assert reports[0]["player_name"] == "Stephen Curry"
        assert reports[0]["injury_status"] == "Out"


class TestRotoWireScraper:
    def test_normalize_status(self):
        scraper = RotoWireScraper()
        assert scraper._normalize_status("Out") == "Out"
        assert scraper._normalize_status("GTD") == "Questionable"
        assert scraper._normalize_status("day-to-day") == "Questionable"
        assert scraper._normalize_status("DOUBTFUL") == "Doubtful"
        assert scraper._normalize_status("something else") == "something else"

    def test_extract_timeline(self):
        scraper = RotoWireScraper()
        assert scraper._extract_timeline("Expected to miss 2-3 weeks") == "2-3 weeks"
        assert scraper._extract_timeline("day-to-day with ankle") == "day-to-day"
        assert scraper._extract_timeline("season-ending injury") == "season-ending"
        assert scraper._extract_timeline("No timeline info") is None

    def test_save_reports(self, db_session):
        scraper = RotoWireScraper()
        reports = [
            {
                "player_name": "Ja Morant",
                "team": "Grizzlies",
                "injury_status": "Out",
                "injury_description": "Shoulder",
                "report_date": datetime(2024, 1, 15),
                "source": "rotowire",
            },
            {
                "player_name": "Zion Williamson",
                "team": "Pelicans",
                "injury_status": "Doubtful",
                "injury_description": "Hamstring",
                "report_date": datetime(2024, 1, 15),
                "source": "rotowire",
            },
        ]
        saved = scraper.save_reports(db_session, reports)
        db_session.commit()

        assert saved == 2
        assert db_session.query(InjuryReport).count() == 2

    @patch.object(RotoWireScraper, "fetch")
    def test_scrape_handles_failed_fetch(self, mock_fetch):
        mock_fetch.return_value = None
        scraper = RotoWireScraper()
        reports = scraper.scrape()
        assert reports == []
