import logging
from datetime import datetime

from bs4 import BeautifulSoup

from database.models import InjuryReport
from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# NBA.com injury report page
NBA_INJURY_URL = "https://www.nba.com/players/injuries"

# NBA stats API endpoint (JSON, avoids need for Selenium)
NBA_API_URL = "https://stats.nba.com/stats/playerindex"


class NBAOfficialScraper(BaseScraper):
    """Scrapes injury reports from NBA.com."""

    SOURCE = "nba.com"
    REQUIRED_FIELDS = ["player_name", "team", "injury_status"]

    def __init__(self):
        super().__init__()
        # NBA.com requires specific headers for API access
        self.session.headers.update({
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
        })

    def scrape(self):
        """Scrape NBA.com injury report page.

        Tries the HTML page first. NBA.com heavily uses JavaScript rendering,
        so this may return incomplete data without Selenium.
        """
        reports = []

        logger.info("Fetching NBA.com injury report page...")
        self._rate_limit()
        response = self.fetch(NBA_INJURY_URL)

        if response is None:
            logger.error("Failed to fetch NBA.com injury page.")
            return reports

        try:
            reports = self._parse_injury_page(response.text)
            logger.info(f"Parsed {len(reports)} injury reports from NBA.com")
        except Exception as e:
            logger.error(f"Error parsing NBA.com injury page: {e}")

        return reports

    def _parse_injury_page(self, html):
        """Parse the NBA.com injury report HTML page."""
        soup = BeautifulSoup(html, "lxml")
        reports = []

        # NBA.com renders injury data in tables/divs with player cards
        # The exact selectors depend on the current page structure
        # Try multiple selector strategies

        # Strategy 1: Look for injury table rows
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    report = self._parse_table_row(cells)
                    if report and self._validate_report(report, self.REQUIRED_FIELDS):
                        reports.append(report)

        # Strategy 2: Look for player injury cards/divs
        if not reports:
            injury_sections = soup.find_all(
                "div", class_=lambda c: c and "injury" in c.lower()
            ) or soup.find_all(
                "section", class_=lambda c: c and "injury" in c.lower()
            )
            for section in injury_sections:
                report = self._parse_injury_section(section)
                if report and self._validate_report(report, self.REQUIRED_FIELDS):
                    reports.append(report)

        if not reports:
            logger.warning(
                "No injury data found on NBA.com. The page likely requires "
                "JavaScript rendering (Selenium) or the page structure has changed."
            )

        return reports

    def _parse_table_row(self, cells):
        """Parse a table row into a report dict."""
        try:
            return {
                "player_name": cells[0].get_text(strip=True),
                "team": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                "injury_status": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                "injury_description": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                "report_date": datetime.utcnow(),
                "source": self.SOURCE,
            }
        except (IndexError, AttributeError) as e:
            logger.debug(f"Failed to parse table row: {e}")
            return None

    def _parse_injury_section(self, section):
        """Parse an injury card/section div into a report dict."""
        try:
            text_content = section.get_text(" ", strip=True)
            # Attempt to extract structured data from text
            # This is a best-effort parse; exact selectors depend on page structure
            name_elem = section.find(["h3", "h4", "a", "span"])
            player_name = name_elem.get_text(strip=True) if name_elem else ""

            return {
                "player_name": player_name,
                "team": "",
                "injury_status": self._extract_status(text_content),
                "injury_description": text_content[:200],
                "report_date": datetime.utcnow(),
                "source": self.SOURCE,
            }
        except Exception as e:
            logger.debug(f"Failed to parse injury section: {e}")
            return None

    def _extract_status(self, text):
        """Extract injury status from text content."""
        text_lower = text.lower()
        for status in ["out", "doubtful", "questionable", "probable", "available"]:
            if status in text_lower:
                return status.capitalize()
        return "Unknown"

    def save_reports(self, db_session, reports):
        """Save scraped reports to the injury_reports table."""
        saved = 0
        for report in reports:
            try:
                record = InjuryReport(
                    player_name=report["player_name"],
                    team=report["team"],
                    injury_status=report["injury_status"],
                    injury_description=report.get("injury_description", ""),
                    report_date=report.get("report_date"),
                    source=report["source"],
                )
                db_session.add(record)
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save report for {report.get('player_name')}: {e}")

        logger.info(f"Saved {saved}/{len(reports)} NBA.com injury reports to database.")
        return saved
