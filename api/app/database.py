from sqlalchemy import create_engine

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)


def get_connection():
    """Yield a database connection for use in endpoints."""
    with engine.connect() as conn:
        yield conn
