import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Connection, text

from app.database import get_connection
from app.schemas import PlayerDetail, PlayerPredictionHistory, PredictionResponse, TopFactor

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


def _parse_top_factors(raw) -> list[TopFactor]:
    """Parse top_factors JSON column into list of TopFactor."""
    if raw is None:
        return []
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw
    return [TopFactor(**f) for f in data]


@router.get("/today", response_model=list[PredictionResponse])
def get_today_predictions(
    tier: Optional[str] = Query(None, description="Filter by risk tier: high, medium, low"),
    min_percentile: float = Query(0, description="Minimum risk percentile"),
    limit: int = Query(500, le=500),
    conn: Connection = Depends(get_connection),
):
    """Get today's predictions, ordered by risk score descending."""
    query = """
        SELECT dp.player_id, dp.player_name, tl.abbreviation,
               dp.prediction_date, dp.risk_score, dp.risk_percentile, dp.risk_tier,
               dp.top_factors, dp.outcome_verified, dp.missed_game_actual
        FROM daily_predictions dp
        LEFT JOIN team_lookup tl ON dp.team_id = tl.team_id
        WHERE dp.prediction_date = (
            SELECT MAX(prediction_date) FROM daily_predictions
        )
        AND dp.risk_percentile >= :min_pct
    """
    params = {"min_pct": min_percentile}

    if tier:
        query += " AND dp.risk_tier = :tier"
        params["tier"] = tier

    query += " ORDER BY dp.risk_score DESC LIMIT :lim"
    params["lim"] = limit

    rows = conn.execute(text(query), params).fetchall()

    return [
        PredictionResponse(
            player_id=r[0],
            player_name=r[1],
            team_abbreviation=r[2],
            prediction_date=r[3],
            risk_score=r[4],
            risk_percentile=r[5],
            risk_tier=r[6],
            top_factors=_parse_top_factors(r[7]),
            outcome_verified=bool(r[8]),
            missed_game_actual=bool(r[9]) if r[9] is not None else None,
        )
        for r in rows
    ]


@router.get("/player/{player_id}", response_model=PlayerDetail)
def get_player_predictions(
    player_id: int,
    limit: int = Query(30, le=90),
    conn: Connection = Depends(get_connection),
):
    """Get prediction history for a specific player."""
    # Get player info
    player_info = conn.execute(
        text(
            "SELECT p.player_name, tl.abbreviation "
            "FROM player p "
            "LEFT JOIN player_season ps ON p.player_id = ps.player_id "
            "LEFT JOIN team_lookup tl ON ps.team_id = tl.team_id "
            "WHERE p.player_id = :pid "
            "ORDER BY ps.season_id DESC LIMIT 1"
        ),
        {"pid": player_id},
    ).fetchone()

    player_name = player_info[0] if player_info else f"Player {player_id}"
    team_abbr = player_info[1] if player_info else None

    # Get prediction history
    rows = conn.execute(
        text(
            "SELECT prediction_date, risk_score, risk_percentile, risk_tier, "
            "top_factors, outcome_verified, missed_game_actual "
            "FROM daily_predictions "
            "WHERE player_id = :pid "
            "ORDER BY prediction_date DESC "
            "LIMIT :lim"
        ),
        {"pid": player_id, "lim": limit},
    ).fetchall()

    predictions = [
        PlayerPredictionHistory(
            prediction_date=r[0],
            risk_score=r[1],
            risk_percentile=r[2],
            risk_tier=r[3],
            top_factors=_parse_top_factors(r[4]),
            outcome_verified=bool(r[5]),
            missed_game_actual=bool(r[6]) if r[6] is not None else None,
        )
        for r in rows
    ]

    return PlayerDetail(
        player_id=player_id,
        player_name=player_name,
        team_abbreviation=team_abbr,
        predictions=predictions,
    )
