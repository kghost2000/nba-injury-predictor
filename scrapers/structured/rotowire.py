import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup

from database.models import InjuryReport
from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

ROTOWIRE_INJURY_URL = "https://www.rotowire.com/basketball/injury-report.php"


class RotoWireScraper(BaseScraper):
    """Scrapes injury reports from RotoWire."""

    SOURCE = "rotowire"
    REQUIRED_FIELDS = ["player_name", "team", "injury_status"]

    def scrape(self):
        """Scrape RotoWire NBA injury report page."""
        reports = []

        logger.info("Fetching RotoWire injury report page...")
        self._rate_limit()
        response = self.fetch(ROTOWIRE_INJURY_URL)

        if response is None:
            logger.error("Failed to fetch RotoWire injury page.")
            return reports

        try:
            reports = self._parse_injury_page(response.text)
            logger.info(f"Parsed {len(reports)} injury reports from RotoWire")
        except Exception as e:
            logger.error(f"Error parsing RotoWire injury page: {e}")

        return reports

    def _parse_injury_page(self, html):
        """Parse the RotoWire injury report HTML page."""
        soup = BeautifulSoup(html, "lxml")
        reports = []

        # RotoWire uses a table-based layout for injury reports
        # Look for the main injury table
        table = soup.find("table", class_=lambda c: c and "injury" in c.lower()) or \
                soup.find("div", class_=lambda c: c and "injury" in c.lower())

        if table:
            rows = table.find_all("tr")
            current_team = ""
            for row in rows:
                # Team header rows
                team_header = row.find("th") or row.find(
                    class_=lambda c: c and "team" in str(c).lower()
                )
                if team_header and not row.find("td"):
                    current_team = team_header.get_text(strip=True)
                    continue

                cells = row.find_all("td")
                if len(cells) >= 2:
                    report = self._parse_row(cells, current_team)
                    if report and self._validate_report(report, self.REQUIRED_FIELDS):
                        reports.append(report)

        # Fallback: look for individual player injury entries
        if not reports:
            entries = soup.find_all(
                "div", class_=lambda c: c and any(
                    k in str(c).lower() for k in ["player", "injury", "report"]
                )
            )
            for entry in entries:
                report = self._parse_entry(entry)
                if report and self._validate_report(report, self.REQUIRED_FIELDS):
                    reports.append(report)

        if not reports:
            logger.warning(
                "No injury data found on RotoWire. The page structure may "
                "have changed or access may be restricted."
            )

        return reports

    def _parse_row(self, cells, current_team):
        """Parse a table row into a report dict."""
        try:
            player_name = cells[0].get_text(strip=True)
            # Some layouts have team in the row rather than as a header
            team = current_team
            if not team and len(cells) > 3:
                team = cells[1].get_text(strip=True)

            status_idx = 1 if team == current_team else 2
            status = cells[status_idx].get_text(strip=True) if len(cells) > status_idx else ""

            desc_idx = status_idx + 1
            description = cells[desc_idx].get_text(strip=True) if len(cells) > desc_idx else ""

            # Try to extract timeline from description
            timeline = self._extract_timeline(description)
            if timeline:
                description = f"{description} ({timeline})"

            return {
                "player_name": player_name,
                "team": team,
                "injury_status": self._normalize_status(status),
                "injury_description": description[:200],
                "report_date": datetime.utcnow(),
                "source": self.SOURCE,
            }
        except (IndexError, AttributeError) as e:
            logger.debug(f"Failed to parse RotoWire row: {e}")
            return None

    def _parse_entry(self, entry):
        """Parse an individual injury entry div."""
        try:
            text = entry.get_text(" ", strip=True)
            name_elem = entry.find(["a", "span", "strong"])
            player_name = name_elem.get_text(strip=True) if name_elem else ""

            return {
                "player_name": player_name,
                "team": "",
                "injury_status": self._extract_status(text),
                "injury_description": text[:200],
                "report_date": datetime.utcnow(),
                "source": self.SOURCE,
            }
        except Exception as e:
            logger.debug(f"Failed to parse RotoWire entry: {e}")
            return None

    def _normalize_status(self, status):
        """Normalize injury status to standard values."""
        status_lower = status.lower().strip()
        mapping = {
            "out": "Out",
            "doubtful": "Doubtful",
            "questionable": "Questionable",
            "probable": "Probable",
            "available": "Available",
            "gtd": "Questionable",  # Game-time decision
            "day-to-day": "Questionable",
        }
        for key, value in mapping.items():
            if key in status_lower:
                return value
        return status or "Unknown"

    def _extract_status(self, text):
        """Extract injury status from free text."""
        text_lower = text.lower()
        for status in ["out", "doubtful", "questionable", "probable", "available"]:
            if status in text_lower:
                return status.capitalize()
        return "Unknown"

    def _extract_timeline(self, description):
        """Extract injury timeline from description text if available."""
        patterns = [
            r"expected to miss (\d+[-–]\d+ (?:weeks?|games?|days?))",
            r"(out \d+[-–]\d+ (?:weeks?|games?|days?))",
            r"(week-to-week|day-to-day|indefinitely|season-ending)",
        ]
        desc_lower = description.lower()
        for pattern in patterns:
            match = re.search(pattern, desc_lower)
            if match:
                return match.group(1)
        return None

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

        logger.info(f"Saved {saved}/{len(reports)} RotoWire injury reports to database.")
        return saved
