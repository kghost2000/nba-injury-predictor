"""Seed the reporters table with known NBA insiders."""

import logging

from database.models import Reporter

logger = logging.getLogger(__name__)

KNOWN_REPORTERS = [
    {
        "name": "Adrian Wojnarowski",
        "twitter_handle": "@wojespn",
        "team_coverage": "League-wide",
        "reliability_score": 0.98,
    },
    {
        "name": "Shams Charania",
        "twitter_handle": "@ShamsCharania",
        "team_coverage": "League-wide",
        "reliability_score": 0.97,
    },
    {
        "name": "Chris Haynes",
        "twitter_handle": "@ChrisBHaynes",
        "team_coverage": "League-wide",
        "reliability_score": 0.90,
    },
    {
        "name": "Marc Stein",
        "twitter_handle": "@TheSteinLine",
        "team_coverage": "League-wide",
        "reliability_score": 0.92,
    },
    {
        "name": "Keith Smith",
        "twitter_handle": "@KeithSmithNBA",
        "team_coverage": "League-wide",
        "reliability_score": 0.85,
    },
    {
        "name": "Marc J. Spears",
        "twitter_handle": "@MarcJSpears",
        "team_coverage": "League-wide",
        "reliability_score": 0.88,
    },
    {
        "name": "Jake Fischer",
        "twitter_handle": "@JakeLFischer",
        "team_coverage": "League-wide",
        "reliability_score": 0.86,
    },
    {
        "name": "Chris Mannix",
        "twitter_handle": "@SIChrisMannix",
        "team_coverage": "League-wide",
        "reliability_score": 0.84,
    },
    {
        "name": "Ramona Shelburne",
        "twitter_handle": "@raaborern",
        "team_coverage": "League-wide",
        "reliability_score": 0.89,
    },
    {
        "name": "Brian Windhorst",
        "twitter_handle": "@WindhorstESPN",
        "team_coverage": "League-wide",
        "reliability_score": 0.87,
    },
]


def seed_reporters(session):
    """Seed the reporters table with known NBA insiders. Idempotent."""
    added = 0
    for data in KNOWN_REPORTERS:
        existing = (
            session.query(Reporter)
            .filter(Reporter.twitter_handle == data["twitter_handle"])
            .first()
        )
        if existing:
            logger.debug(f"Reporter {data['name']} already exists, skipping.")
            continue

        reporter = Reporter(**data)
        session.add(reporter)
        added += 1

    if added:
        logger.info(f"Seeded {added} reporters into the database.")
    else:
        logger.info("All reporters already seeded, nothing to add.")
    return added
