import io
import logging
import re
from datetime import datetime

import pdfplumber

from database.models import InjuryReport
from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# NBA publishes official injury reports as PDFs on this CDN path.
# The filename includes the date and time of the report.
NBA_INJURY_PDF_BASE = "https://ak-static.cms.nba.com/referee/injury"

# Common report times (ET) published each day
REPORT_TIMES = [
    "05_00PM", "04_00PM", "03_00PM", "02_00PM",
    "01_30PM", "01_00PM", "12_00PM",
]

VALID_STATUSES = {"Out", "Questionable", "Probable", "Doubtful", "Available"}


class NBAOfficialScraper(BaseScraper):
    """Scrapes injury reports from the official NBA injury report PDFs."""

    SOURCE = "nba.com"
    REQUIRED_FIELDS = ["player_name", "team", "injury_status"]

    def scrape(self):
        """Download and parse the latest NBA injury report PDF."""
        reports = []

        logger.info("Fetching NBA official injury report PDF...")
        self._rate_limit()

        pdf_bytes = self._fetch_latest_pdf()
        if pdf_bytes is None:
            logger.error("Failed to fetch any NBA injury report PDF.")
            return reports

        try:
            reports = self._parse_pdf(pdf_bytes)
            logger.info(f"Parsed {len(reports)} injury reports from NBA.com")
        except Exception as e:
            logger.error(f"Error parsing NBA injury report PDF: {e}")

        return reports

    def _fetch_latest_pdf(self):
        """Try fetching the most recent injury report PDF for today.

        Uses HEAD requests to quickly probe URLs, then downloads the first
        match. Tries today and the previous two days with common report times.
        Returns the PDF content as bytes, or None if all attempts fail.
        """
        from datetime import timedelta

        today = datetime.utcnow()
        dates_to_try = [
            (today - timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range(3)
        ]

        for date_str in dates_to_try:
            for time_str in REPORT_TIMES:
                url = (
                    f"{NBA_INJURY_PDF_BASE}/"
                    f"Injury-Report_{date_str}_{time_str}.pdf"
                )
                # Quick HEAD check to avoid slow retries on missing PDFs
                try:
                    head = self.session.head(url, timeout=5)
                    if head.status_code != 200:
                        continue
                except Exception:
                    continue

                response = self.fetch(url)
                if response is not None:
                    content_type = response.headers.get("content-type", "")
                    if "pdf" in content_type or len(response.content) > 1000:
                        logger.info(f"Found injury report: {url}")
                        return response.content
            logger.debug(f"No report found for {date_str}, trying earlier date...")

        return None

    def _parse_pdf(self, pdf_bytes):
        """Extract injury data from the PDF bytes."""
        reports = []
        current_team = ""

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.split("\n"):
                    line = line.strip()
                    result = self._parse_line(line, current_team)
                    if result is None:
                        continue
                    report, team = result
                    if team:
                        current_team = team
                    if report and self._validate_report(report, self.REQUIRED_FIELDS):
                        reports.append(report)

        if not reports:
            logger.warning("No injury data found in the NBA PDF.")

        return reports

    def _parse_line(self, line, current_team):
        """Parse a single line from the extracted PDF text.

        Returns (report_dict, team) or None if the line isn't a player entry.
        """
        if not line:
            return None
        # Skip headers and page footers
        if line.startswith("Injury Report:") or line.startswith("GameDate"):
            return None
        if re.match(r"^Page\s*\d+\s*of\s*\d+$", line):
            return None

        for status in VALID_STATUSES:
            if f" {status} " in line or line.endswith(f" {status}"):
                idx = line.rfind(f" {status}")
                before = line[:idx].strip()
                reason = line[idx + len(status) + 1:].strip()

                # Player names appear as "Last,First"
                name_match = re.search(
                    r"([A-Za-z\'\-\.]+(?:III|II|IV|Jr|Sr)?),\s*([A-Za-z\'\-\.]+)",
                    before,
                )
                if not name_match:
                    return None

                # Check for team name (CamelCase) before the player name
                name_start = before.find(name_match.group(0))
                prefix = before[:name_start].strip()

                team = current_team
                team_match = re.search(r"([A-Z][a-z]+(?:[A-Z][a-z]+)+)", prefix)
                if team_match:
                    team = team_match.group(1)

                player_name = f"{name_match.group(2)} {name_match.group(1)}"
                description = self._clean_reason(reason)

                report = {
                    "player_name": player_name,
                    "team": self._format_team(team),
                    "injury_status": status,
                    "injury_description": description[:200],
                    "report_date": datetime.utcnow(),
                    "source": self.SOURCE,
                }
                return report, team_match.group(1) if team_match else None

        return None

    def _clean_reason(self, reason):
        """Clean up the reason string from the PDF."""
        if not reason:
            return ""
        # Remove "Injury/Illness-" prefix for cleaner descriptions
        reason = re.sub(r"^Injury/Illness-", "", reason)
        # Replace semicolons with more readable separators
        reason = reason.replace(";", " - ")
        return reason.strip()

    def _format_team(self, team):
        """Convert CamelCase team name to readable format."""
        if not team:
            return ""
        # Insert spaces before capital letters: "WashingtonWizards" -> "Washington Wizards"
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", team)

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
