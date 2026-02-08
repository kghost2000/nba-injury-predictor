import logging
import re
from datetime import datetime

from database.models import InjuryReport
from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

ROTOWIRE_INJURY_API = (
    "https://www.rotowire.com/basketball/tables/injury-report.php"
    "?team=ALL&pos=ALL"
)


class RotoWireScraper(BaseScraper):
    """Scrapes injury reports from RotoWire's JSON API."""

    SOURCE = "rotowire"
    REQUIRED_FIELDS = ["player_name", "team", "injury_status"]

    def scrape(self):
        """Fetch injury data from the RotoWire JSON endpoint."""
        reports = []

        logger.info("Fetching RotoWire injury data...")
        self._rate_limit()

        # The API requires a Referer header to respond
        response = self.fetch(
            ROTOWIRE_INJURY_API,
            headers={
                "Referer": "https://www.rotowire.com/basketball/injury-report.php",
                "X-Requested-With": "XMLHttpRequest",
            },
        )

        if response is None:
            logger.error("Failed to fetch RotoWire injury data.")
            return reports

        try:
            data = response.json()
            reports = self._parse_json(data)
            logger.info(f"Parsed {len(reports)} injury reports from RotoWire")
        except Exception as e:
            logger.error(f"Error parsing RotoWire injury data: {e}")

        return reports

    def _parse_json(self, data):
        """Parse the JSON array of injury entries."""
        reports = []
        for entry in data:
            report = self._parse_entry(entry)
            if report and self._validate_report(report, self.REQUIRED_FIELDS):
                reports.append(report)
        return reports

    def _parse_entry(self, entry):
        """Convert a single JSON injury entry to a report dict."""
        try:
            player_name = entry.get("player", "").strip()
            team = entry.get("team", "").strip()
            status = entry.get("status", "").strip()
            injury = entry.get("injury", "").strip()

            description = injury
            timeline = self._extract_timeline(status)
            if timeline:
                description = f"{injury} ({timeline})"

            return {
                "player_name": player_name,
                "team": team,
                "injury_status": self._normalize_status(status),
                "injury_description": description[:200],
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
            "out for season": "Out",
            "out": "Out",
            "doubtful": "Doubtful",
            "questionable": "Questionable",
            "probable": "Probable",
            "available": "Available",
            "gtd": "Questionable",
            "day-to-day": "Questionable",
        }
        for key, value in mapping.items():
            if key in status_lower:
                return value
        return status or "Unknown"

    def _extract_timeline(self, status):
        """Extract injury timeline from the status field."""
        patterns = [
            r"expected to miss (\d+[-–]\d+ (?:weeks?|games?|days?))",
            r"(out \d+[-–]\d+ (?:weeks?|games?|days?))",
            r"(week-to-week|day-to-day|indefinitely|out for season)",
        ]
        status_lower = status.lower()
        for pattern in patterns:
            match = re.search(pattern, status_lower)
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
