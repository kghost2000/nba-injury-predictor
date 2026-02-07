import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, GameOutcome, InjuryReport, InjurySignal, Reporter


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestInjuryReport:
    def test_create_injury_report(self, db_session):
        report = InjuryReport(
            player_name="LeBron James",
            team="Los Angeles Lakers",
            injury_status="Questionable",
            injury_description="Left ankle soreness",
            report_date=datetime(2024, 1, 15),
            source="nba.com",
        )
        db_session.add(report)
        db_session.commit()

        result = db_session.query(InjuryReport).first()
        assert result.player_name == "LeBron James"
        assert result.team == "Los Angeles Lakers"
        assert result.injury_status == "Questionable"
        assert result.source == "nba.com"
        assert result.created_at is not None

    def test_multiple_reports(self, db_session):
        reports = [
            InjuryReport(player_name="Player A", team="Team A",
                         injury_status="Out", source="nba.com"),
            InjuryReport(player_name="Player B", team="Team B",
                         injury_status="Probable", source="rotowire"),
        ]
        db_session.add_all(reports)
        db_session.commit()

        assert db_session.query(InjuryReport).count() == 2

    def test_repr(self, db_session):
        report = InjuryReport(
            player_name="Steph Curry", team="GSW",
            injury_status="Out", source="nba.com",
        )
        assert "Steph Curry" in repr(report)
        assert "Out" in repr(report)


class TestInjurySignal:
    def test_create_signal(self, db_session):
        signal = InjurySignal(
            player_name="Kevin Durant",
            source_type="tweet",
            raw_text="KD is expected to miss tonight's game",
            extracted_info='{"status": "out"}',
            confidence_score=0.85,
            timestamp=datetime(2024, 1, 15),
            reporter_name="Woj",
        )
        db_session.add(signal)
        db_session.commit()

        result = db_session.query(InjurySignal).first()
        assert result.player_name == "Kevin Durant"
        assert result.confidence_score == 0.85


class TestGameOutcome:
    def test_create_outcome(self, db_session):
        outcome = GameOutcome(
            player_name="Giannis Antetokounmpo",
            game_date=datetime(2024, 1, 15),
            did_play=True,
            minutes_played=34.5,
            game_id="0022300123",
        )
        db_session.add(outcome)
        db_session.commit()

        result = db_session.query(GameOutcome).first()
        assert result.did_play is True
        assert result.minutes_played == 34.5


class TestReporter:
    def test_create_reporter(self, db_session):
        reporter = Reporter(
            name="Adrian Wojnarowski",
            twitter_handle="@wojespn",
            team_coverage="League-wide",
            reliability_score=0.98,
        )
        db_session.add(reporter)
        db_session.commit()

        result = db_session.query(Reporter).first()
        assert result.name == "Adrian Wojnarowski"
        assert result.reliability_score == 0.98


class TestSchema:
    def test_init_db_creates_tables(self):
        """Test that init_db creates all expected tables."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        table_names = Base.metadata.tables.keys()
        assert "injury_reports" in table_names
        assert "injury_signals" in table_names
        assert "game_outcomes" in table_names
        assert "reporters" in table_names
