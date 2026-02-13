import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://nba_sql:nba_sql@localhost:3306/nba",
)
