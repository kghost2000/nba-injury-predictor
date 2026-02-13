"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getPlayerDetail, type PlayerDetail } from "@/lib/api";

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\bL(\d+)\b/g, "last $1")
    .replace(/\b7d\b/g, "7-day")
    .replace(/\b14d\b/g, "14-day")
    .replace(/\b30d\b/g, "30-day")
    .replace(/\b365d\b/g, "1-year");
}

const tierColors: Record<string, string> = {
  high: "text-red-400",
  medium: "text-yellow-400",
  low: "text-green-400",
};

export default function PlayerPage() {
  const params = useParams();
  const playerId = parseInt(params.id as string, 10);
  const [player, setPlayer] = useState<PlayerDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getPlayerDetail(playerId)
      .then(setPlayer)
      .catch(() => setPlayer(null))
      .finally(() => setLoading(false));
  }, [playerId]);

  if (loading) {
    return <div className="text-center py-12 text-gray-500">Loading player data...</div>;
  }

  if (!player) {
    return <div className="text-center py-12 text-gray-500">Player not found.</div>;
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">
          {player.player_name}
        </h1>
        {player.team_abbreviation && (
          <p className="text-sm text-gray-400">{player.team_abbreviation}</p>
        )}
      </div>

      {player.predictions.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          No prediction history available for this player.
        </div>
      ) : (
        <div className="space-y-4">
          {player.predictions.map((pred) => {
            const percentileDisplay = Math.round(100 - pred.risk_percentile);
            return (
              <div
                key={pred.prediction_date}
                className="bg-gray-900 rounded-lg border border-gray-800 p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm text-gray-400">
                    {pred.prediction_date}
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className={`text-sm font-semibold ${tierColors[pred.risk_tier] || tierColors.low}`}
                    >
                      {pred.risk_tier.toUpperCase()} - Top {percentileDisplay}%
                    </span>
                    <span className="text-sm text-gray-300">
                      {(pred.risk_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {pred.top_factors.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-3">
                    {pred.top_factors.map((f) => (
                      <div
                        key={f.feature}
                        className="text-xs text-gray-400 flex justify-between"
                      >
                        <span>{formatFeatureName(f.feature)}</span>
                        <span>
                          {Math.abs(f.z_score).toFixed(1)} SD {f.direction} avg
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {pred.outcome_verified && (
                  <div className="text-xs pt-2 border-t border-gray-800">
                    {pred.missed_game_actual ? (
                      <span className="text-red-400">
                        Outcome: Missed game
                      </span>
                    ) : (
                      <span className="text-green-400">
                        Outcome: Played
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
