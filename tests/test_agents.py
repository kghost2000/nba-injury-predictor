"""Tests for Claude client and injury parser (Phase 4)."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agents.claude_client import ClaudeClient, EXTRACTION_PROMPT
from agents.injury_parser import InjuryParser
from database.models import Base, InjurySignal, Reporter


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
# ClaudeClient tests
# ---------------------------------------------------------------------------

class TestClaudeClient:
    @patch("agents.claude_client.anthropic.Anthropic")
    def test_extract_injury_info_valid_response(self, mock_anthropic_cls):
        """Test that a valid JSON response is parsed correctly."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        response_json = json.dumps({
            "player_name": "LeBron James",
            "team": "Los Angeles Lakers",
            "injury_type": "ankle sprain",
            "severity": "moderate",
            "estimated_timeline": "day-to-day",
            "is_injury_related": True,
            "confidence": 0.92,
        })
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_json)]
        mock_client.messages.create.return_value = mock_message

        client = ClaudeClient(api_key="test-key")
        result = client.extract_injury_info("LeBron out with ankle sprain")

        assert result["player_name"] == "LeBron James"
        assert result["team"] == "Los Angeles Lakers"
        assert result["injury_type"] == "ankle sprain"
        assert result["severity"] == "moderate"
        assert result["is_injury_related"] is True
        assert result["confidence"] == 0.92

    @patch("agents.claude_client.anthropic.Anthropic")
    def test_extract_injury_info_malformed_json(self, mock_anthropic_cls):
        """Test graceful handling of malformed JSON responses."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="This is not valid JSON")]
        mock_client.messages.create.return_value = mock_message

        client = ClaudeClient(api_key="test-key")
        result = client.extract_injury_info("some text")

        assert result["is_injury_related"] is False
        assert result["confidence"] == 0.0
        assert result["player_name"] is None

    @patch("agents.claude_client.anthropic.Anthropic")
    def test_extract_injury_info_non_injury_text(self, mock_anthropic_cls):
        """Test that non-injury text is correctly identified."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        response_json = json.dumps({
            "player_name": None,
            "team": None,
            "injury_type": None,
            "severity": "unknown",
            "estimated_timeline": None,
            "is_injury_related": False,
            "confidence": 0.95,
        })
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_json)]
        mock_client.messages.create.return_value = mock_message

        client = ClaudeClient(api_key="test-key")
        result = client.extract_injury_info("Lakers win 110-105 in overtime")

        assert result["is_injury_related"] is False
        assert result["player_name"] is None

    @patch("agents.claude_client.anthropic.Anthropic")
    def test_extract_injury_info_code_fenced_json(self, mock_anthropic_cls):
        """Test parsing JSON wrapped in markdown code fences."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        fenced = '```json\n{"player_name": "Steph Curry", "team": "GSW", "injury_type": "knee", "severity": "minor", "estimated_timeline": "day-to-day", "is_injury_related": true, "confidence": 0.88}\n```'
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=fenced)]
        mock_client.messages.create.return_value = mock_message

        client = ClaudeClient(api_key="test-key")
        result = client.extract_injury_info("Steph questionable with knee soreness")

        assert result["player_name"] == "Steph Curry"
        assert result["is_injury_related"] is True

    @patch("agents.claude_client.anthropic.Anthropic")
    def test_prompt_contains_expected_fields(self, mock_anthropic_cls):
        """Test that the prompt sent to Claude includes all extraction fields."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text='{"is_injury_related": false, "confidence": 0.0}')]
        mock_client.messages.create.return_value = mock_message

        client = ClaudeClient(api_key="test-key")
        client.extract_injury_info("test text", source_context="tweet from @wojespn")

        call_args = mock_client.messages.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]

        assert "player_name" in prompt
        assert "injury_type" in prompt
        assert "severity" in prompt
        assert "estimated_timeline" in prompt
        assert "is_injury_related" in prompt
        assert "confidence" in prompt
        assert "test text" in prompt
        assert "@wojespn" in prompt

    @patch("agents.claude_client.anthropic.Anthropic")
    def test_extract_injury_info_api_error(self, mock_anthropic_cls):
        """Test graceful handling of API errors."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        import anthropic
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="rate limited",
            request=MagicMock(),
            body=None,
        )

        client = ClaudeClient(api_key="test-key")
        result = client.extract_injury_info("some text")

        assert result["is_injury_related"] is False
        assert result["confidence"] == 0.0

    @patch("agents.claude_client.anthropic.Anthropic")
    def test_extract_batch(self, mock_anthropic_cls):
        """Test batch extraction processes all items."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        response_json = json.dumps({
            "player_name": "Test Player",
            "team": "Test Team",
            "injury_type": "ankle",
            "severity": "minor",
            "estimated_timeline": None,
            "is_injury_related": True,
            "confidence": 0.8,
        })
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_json)]
        mock_client.messages.create.return_value = mock_message

        client = ClaudeClient(api_key="test-key")
        texts = [("text1", "ctx1"), ("text2", "ctx2")]

        with patch("agents.claude_client.time.sleep"):
            results = client.extract_batch(texts)

        assert len(results) == 2
        assert all(r["player_name"] == "Test Player" for r in results)


# ---------------------------------------------------------------------------
# InjuryParser tests
# ---------------------------------------------------------------------------

class TestInjuryParser:
    def _make_claude_client(self, response):
        """Create a mock ClaudeClient that returns the given response."""
        mock = MagicMock()
        mock.extract_injury_info.return_value = response
        return mock

    def test_parse_signal_returns_injury_signal(self):
        """Test that injury-related text produces an InjurySignal."""
        client = self._make_claude_client({
            "player_name": "Kevin Durant",
            "team": "Phoenix Suns",
            "injury_type": "calf strain",
            "severity": "moderate",
            "estimated_timeline": "2-3 weeks",
            "is_injury_related": True,
            "confidence": 0.9,
        })
        parser = InjuryParser(client)
        signal = parser.parse_signal(
            raw_text="KD expected to miss 2-3 weeks with calf strain",
            source_type="tweet",
            reporter_name="@wojespn",
            timestamp=datetime(2024, 1, 15, 12, 0),
        )

        assert signal is not None
        assert isinstance(signal, InjurySignal)
        assert signal.player_name == "Kevin Durant"
        assert signal.source_type == "tweet"
        assert signal.confidence_score == 0.9
        assert signal.reporter_name == "@wojespn"

    def test_parse_signal_returns_none_for_non_injury(self):
        """Test that non-injury text returns None."""
        client = self._make_claude_client({
            "player_name": None,
            "team": None,
            "injury_type": None,
            "severity": "unknown",
            "estimated_timeline": None,
            "is_injury_related": False,
            "confidence": 0.95,
        })
        parser = InjuryParser(client)
        signal = parser.parse_signal(
            raw_text="Lakers win 110-105",
            source_type="tweet",
        )

        assert signal is None

    def test_parse_signal_returns_none_for_empty_text(self):
        """Test that empty/blank text returns None."""
        client = self._make_claude_client({})
        parser = InjuryParser(client)

        assert parser.parse_signal("", "tweet") is None
        assert parser.parse_signal("   ", "tweet") is None
        assert parser.parse_signal(None, "tweet") is None

    def test_deduplication(self, db_session):
        """Test that duplicate signals within 1hr are skipped."""
        # Insert an existing signal
        existing = InjurySignal(
            player_name="LeBron James",
            source_type="tweet",
            raw_text="LeBron out tonight",
            confidence_score=0.9,
            timestamp=datetime(2024, 1, 15, 12, 0),
        )
        db_session.add(existing)
        db_session.commit()

        client = self._make_claude_client({
            "player_name": "LeBron James",
            "team": "Lakers",
            "injury_type": "ankle",
            "severity": "minor",
            "estimated_timeline": None,
            "is_injury_related": True,
            "confidence": 0.85,
        })
        parser = InjuryParser(client)

        # Signal within 1hr window should be skipped
        signal = parser.parse_signal(
            raw_text="LeBron will not play tonight",
            source_type="tweet",
            timestamp=datetime(2024, 1, 15, 12, 30),
            db_session=db_session,
        )
        assert signal is None

        # Signal outside 1hr window should go through
        signal = parser.parse_signal(
            raw_text="LeBron still out for tomorrow",
            source_type="tweet",
            timestamp=datetime(2024, 1, 15, 14, 0),
            db_session=db_session,
        )
        assert signal is not None

    def test_confidence_combines_reporter_reliability(self, db_session):
        """Test that confidence is adjusted with reporter reliability."""
        # Seed a reporter
        reporter = Reporter(
            name="Adrian Wojnarowski",
            twitter_handle="@wojespn",
            team_coverage="League-wide",
            reliability_score=0.98,
        )
        db_session.add(reporter)
        db_session.commit()

        client = self._make_claude_client({
            "player_name": "Giannis",
            "team": "Bucks",
            "injury_type": "knee",
            "severity": "moderate",
            "estimated_timeline": None,
            "is_injury_related": True,
            "confidence": 0.80,
        })
        parser = InjuryParser(client)
        signal = parser.parse_signal(
            raw_text="Giannis out with knee soreness",
            source_type="tweet",
            reporter_name="@wojespn",
            timestamp=datetime(2024, 1, 15, 12, 0),
            db_session=db_session,
        )

        assert signal is not None
        # Expected: 0.6 * 0.80 + 0.4 * 0.98 = 0.48 + 0.392 = 0.872
        assert abs(signal.confidence_score - 0.872) < 0.01

    def test_confidence_without_reporter(self):
        """Test that confidence is just Claude's score when no reporter found."""
        client = self._make_claude_client({
            "player_name": "Jokic",
            "team": "Nuggets",
            "injury_type": "rest",
            "severity": "minor",
            "estimated_timeline": None,
            "is_injury_related": True,
            "confidence": 0.75,
        })
        parser = InjuryParser(client)
        signal = parser.parse_signal(
            raw_text="Jokic resting tonight",
            source_type="tweet",
            reporter_name="@random_user",
            timestamp=datetime(2024, 1, 15),
        )

        assert signal is not None
        assert signal.confidence_score == 0.75

    def test_parse_batch(self):
        """Test batch parsing returns only injury-related signals."""
        call_count = 0

        def mock_extract(text, context=""):
            nonlocal call_count
            call_count += 1
            if "injury" in text.lower():
                return {
                    "player_name": f"Player {call_count}",
                    "team": "Team",
                    "injury_type": "ankle",
                    "severity": "minor",
                    "estimated_timeline": None,
                    "is_injury_related": True,
                    "confidence": 0.8,
                }
            return {
                "player_name": None,
                "is_injury_related": False,
                "confidence": 0.9,
                "severity": "unknown",
                "estimated_timeline": None,
                "team": None,
                "injury_type": None,
            }

        client = MagicMock()
        client.extract_injury_info.side_effect = mock_extract

        parser = InjuryParser(client)
        items = [
            {"raw_text": "Player A has an ankle injury", "source_type": "tweet",
             "timestamp": datetime(2024, 1, 15, 10, 0)},
            {"raw_text": "Great game last night!", "source_type": "tweet",
             "timestamp": datetime(2024, 1, 15, 11, 0)},
            {"raw_text": "Player B injury update: out 2 weeks", "source_type": "reddit",
             "timestamp": datetime(2024, 1, 15, 12, 0)},
        ]
        signals = parser.parse_batch(items)

        assert len(signals) == 2
        assert all(isinstance(s, InjurySignal) for s in signals)
