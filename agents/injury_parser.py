"""Injury signal parser powered by Claude."""

import json
import logging
from datetime import datetime, timedelta

from database.models import InjurySignal, Reporter

logger = logging.getLogger(__name__)


class InjuryParser:
    """Parses raw text into structured InjurySignal objects using Claude."""

    def __init__(self, claude_client):
        self.claude_client = claude_client

    def parse_signal(self, raw_text, source_type, reporter_name=None,
                     timestamp=None, db_session=None):
        """Parse raw text into an InjurySignal, or None if not injury-related.

        Args:
            raw_text: The raw text to parse.
            source_type: e.g. "tweet", "reddit", "article".
            reporter_name: Name or handle of the reporter.
            timestamp: When the text was posted.
            db_session: Optional DB session for dedup and reporter lookup.

        Returns:
            An InjurySignal ORM object (unsaved), or None.
        """
        if not raw_text or not raw_text.strip():
            return None

        context = f"Source: {source_type}"
        if reporter_name:
            context += f", Reporter: {reporter_name}"

        extracted = self.claude_client.extract_injury_info(raw_text, context)

        if not extracted.get("is_injury_related"):
            logger.debug(f"Text not injury-related, skipping: {raw_text[:80]}...")
            return None

        if not extracted.get("player_name"):
            logger.debug(f"No player name extracted, skipping: {raw_text[:80]}...")
            return None

        # Compute confidence: combine Claude's confidence with reporter reliability
        confidence = extracted.get("confidence", 0.5)
        if db_session and reporter_name:
            confidence = self._adjust_confidence(confidence, reporter_name, db_session)

        # Check for duplicates
        ts = timestamp or datetime.utcnow()
        if db_session and self._is_duplicate(
            extracted["player_name"], source_type, ts, db_session
        ):
            logger.debug(
                f"Duplicate signal for {extracted['player_name']} "
                f"from {source_type}, skipping."
            )
            return None

        signal = InjurySignal(
            player_name=extracted["player_name"],
            source_type=source_type,
            raw_text=raw_text,
            extracted_info=json.dumps(extracted),
            confidence_score=round(confidence, 3),
            timestamp=ts,
            reporter_name=reporter_name,
        )
        return signal

    def parse_batch(self, items, db_session=None):
        """Parse multiple items into InjurySignal objects.

        Args:
            items: list of dicts with keys: raw_text, source_type,
                   reporter_name (optional), timestamp (optional).
            db_session: Optional DB session for dedup and reporter lookup.

        Returns:
            list of InjurySignal objects (unsaved).
        """
        signals = []
        for item in items:
            signal = self.parse_signal(
                raw_text=item.get("raw_text", ""),
                source_type=item.get("source_type", "unknown"),
                reporter_name=item.get("reporter_name"),
                timestamp=item.get("timestamp"),
                db_session=db_session,
            )
            if signal is not None:
                signals.append(signal)
        return signals

    def _adjust_confidence(self, claude_confidence, reporter_name, db_session):
        """Adjust confidence based on reporter reliability score."""
        reporter = (
            db_session.query(Reporter)
            .filter(
                (Reporter.name == reporter_name)
                | (Reporter.twitter_handle == reporter_name)
            )
            .first()
        )
        if reporter and reporter.reliability_score is not None:
            # Weighted average: 60% Claude confidence, 40% reporter reliability
            return (0.6 * claude_confidence) + (0.4 * reporter.reliability_score)
        return claude_confidence

    def _is_duplicate(self, player_name, source_type, timestamp, db_session):
        """Check if a similar signal exists within a 1-hour window."""
        window_start = timestamp - timedelta(hours=1)
        window_end = timestamp + timedelta(hours=1)
        existing = (
            db_session.query(InjurySignal)
            .filter(
                InjurySignal.player_name == player_name,
                InjurySignal.source_type == source_type,
                InjurySignal.timestamp >= window_start,
                InjurySignal.timestamp <= window_end,
            )
            .first()
        )
        return existing is not None
