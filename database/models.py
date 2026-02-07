from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class InjuryReport(Base):
    """Structured data from official sources (NBA.com, RotoWire)."""

    __tablename__ = "injury_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String(100), nullable=False)
    team = Column(String(50), nullable=False)
    injury_status = Column(String(20), nullable=False)  # Out/Doubtful/Questionable/Probable/Available
    injury_description = Column(String(200))
    report_date = Column(DateTime)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<InjuryReport(player={self.player_name}, status={self.injury_status}, source={self.source})>"


class InjurySignal(Base):
    """Unstructured extractions from tweets, articles, Reddit (Phase 4-5)."""

    __tablename__ = "injury_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String(100))
    source_type = Column(String(20), nullable=False)  # tweet/article/reddit
    raw_text = Column(Text)
    extracted_info = Column(Text)  # JSON string
    confidence_score = Column(Float)
    timestamp = Column(DateTime)
    reporter_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<InjurySignal(player={self.player_name}, source={self.source_type})>"


class GameOutcome(Base):
    """Validation labels for checking prediction accuracy."""

    __tablename__ = "game_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String(100), nullable=False)
    game_date = Column(DateTime, nullable=False)
    did_play = Column(Boolean)
    minutes_played = Column(Float)
    game_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<GameOutcome(player={self.player_name}, date={self.game_date}, played={self.did_play})>"


class Reporter(Base):
    """Metadata about injury reporters and their reliability."""

    __tablename__ = "reporters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    twitter_handle = Column(String(50))
    team_coverage = Column(String(100))
    reliability_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Reporter(name={self.name}, reliability={self.reliability_score})>"
