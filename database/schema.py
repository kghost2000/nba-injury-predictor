import logging
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base
from utils.config import DATABASE_URL

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        # Ensure the data directory exists
        db_path = DATABASE_URL.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(DATABASE_URL, echo=False)
        logger.info(f"Database engine created: {DATABASE_URL}")
    return _engine


def init_db():
    """Create all tables in the database."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully.")
    return engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_session():
    """Context manager that provides a transactional DB session."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
