"""Historical NBA injury data using the nbainjuries package.

Pulls official injury reports from the NBA CDN going back to the 2021-22 season.
Each report is a snapshot of injuries for a given date/time.
"""

import logging
import re
from datetime import datetime, timedelta

from nbainjuries import injury

from database.models import InjuryReport

logger = logging.getLogger(__name__)

# NBA regular season approximate date ranges
SEASONS = {
    "2021-22": ("2021-10-19", "2022-04-10"),
    "2022-23": ("2022-10-18", "2023-04-09"),
    "2023-24": ("2023-10-24", "2024-04-14"),
    "2024-25": ("2024-10-22", "2025-04-13"),
    "2025-26": ("2025-10-21", "2026-04-12"),
}


def backfill_season(db_session, season, sample_hour=17, sample_minute=0,
                    interval_days=1):
    """Pull injury reports for every Nth day of a season and save to the DB.

    Args:
        db_session: SQLAlchemy session.
        season: Season string like "2024-25".
        sample_hour: Hour (ET) to pull reports for (default 17 = 5 PM).
        sample_minute: Minute to pull reports for.
        interval_days: Pull a report every N days (1 = daily, 7 = weekly).

    Returns:
        Total number of records saved.
    """
    if season not in SEASONS:
        logger.error(f"Unknown season '{season}'. Available: {list(SEASONS.keys())}")
        return 0

    start_str, end_str = SEASONS[season]
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    # Don't try to fetch future dates
    now = datetime.utcnow()
    if end > now:
        end = now

    total_saved = 0
    total_records = 0
    current = start
    days_total = (end - start).days
    day_num = 0

    logger.info(
        f"Backfilling season {season}: {start_str} to {end.strftime('%Y-%m-%d')} "
        f"(every {interval_days} day(s), sample time {sample_hour}:{sample_minute:02d} ET)"
    )

    while current <= end:
        day_num += 1
        dt = current.replace(hour=sample_hour, minute=sample_minute)
        date_str = current.strftime("%Y-%m-%d")

        try:
            df = injury.get_reportdata(dt, return_df=True)

            if df is not None and not df.empty:
                saved = _save_dataframe(db_session, df, date_str)
                total_saved += saved
                total_records += len(df)
                logger.info(
                    f"  [{day_num}/{days_total}] {date_str}: "
                    f"{len(df)} records fetched, {saved} saved"
                )
            else:
                logger.debug(f"  [{day_num}/{days_total}] {date_str}: no data")

        except Exception as e:
            # Some dates won't have reports (All-Star break, off days, etc.)
            logger.debug(f"  [{day_num}/{days_total}] {date_str}: skipped ({e})")

        current += timedelta(days=interval_days)

    logger.info(
        f"Backfill complete for {season}: "
        f"{total_records} total records, {total_saved} saved to DB"
    )
    return total_saved


def _save_dataframe(db_session, df, date_str):
    """Convert a DataFrame from nbainjuries to InjuryReport records."""
    saved = 0
    report_date = datetime.strptime(date_str, "%Y-%m-%d")

    for _, row in df.iterrows():
        try:
            raw_name = row.get("Player Name", "")
            player_name = _format_name(raw_name)
            team = row.get("Team", "")
            status = row.get("Current Status", "")
            reason = row.get("Reason", "")
            description = _clean_reason(reason)

            if not player_name or not team:
                continue

            record = InjuryReport(
                player_name=player_name,
                team=team,
                injury_status=status,
                injury_description=description[:200],
                report_date=report_date,
                source="nba.com-historical",
            )
            db_session.add(record)
            saved += 1
        except Exception as e:
            logger.debug(f"Failed to save record: {e}")

    return saved


def _format_name(name):
    """Convert 'Last, First' to 'First Last'."""
    if not name:
        return ""
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name.strip()


def _clean_reason(reason):
    """Clean the reason string for storage."""
    if not reason:
        return ""
    reason = re.sub(r"^Injury/Illness\s*-\s*", "", reason)
    return reason.strip()
