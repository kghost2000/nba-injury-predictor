"use client";

import Link from "next/link";
import type { Prediction } from "@/lib/api";

const tierColors: Record<string, string> = {
  high: "bg-red-900/50 border-red-700 text-red-200",
  medium: "bg-yellow-900/50 border-yellow-700 text-yellow-200",
  low: "bg-green-900/50 border-green-700 text-green-200",
};

const tierBadge: Record<string, string> = {
  high: "bg-red-600 text-white",
  medium: "bg-yellow-600 text-white",
  low: "bg-green-700 text-white",
};

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\bL(\d+)\b/g, "last $1")
    .replace(/\b7d\b/g, "7-day")
    .replace(/\b14d\b/g, "14-day")
    .replace(/\b30d\b/g, "30-day")
    .replace(/\b365d\b/g, "1-year");
}

export default function PredictionCard({ pred }: { pred: Prediction }) {
  const percentileDisplay = Math.round(100 - pred.risk_percentile);

  return (
    <div
      className={`rounded-lg border p-4 ${tierColors[pred.risk_tier] || tierColors.low}`}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <Link
            href={`/player/${pred.player_id}`}
            className="font-semibold text-white hover:underline"
          >
            {pred.player_name}
          </Link>
          {pred.team_abbreviation && (
            <span className="ml-2 text-sm text-gray-400">
              {pred.team_abbreviation}
            </span>
          )}
        </div>
        <span
          className={`text-xs font-bold px-2 py-1 rounded ${tierBadge[pred.risk_tier] || tierBadge.low}`}
        >
          Top {percentileDisplay}% risk
        </span>
      </div>

      <div className="text-sm text-gray-300 mb-3">
        Risk score: {(pred.risk_score * 100).toFixed(1)}%
      </div>

      {pred.top_factors.length > 0 && (
        <div className="space-y-1">
          <div className="text-xs text-gray-500 uppercase tracking-wide">
            Key factors
          </div>
          {pred.top_factors.slice(0, 3).map((f) => (
            <div key={f.feature} className="text-xs text-gray-300 flex justify-between">
              <span>{formatFeatureName(f.feature)}</span>
              <span className="text-gray-400">
                {Math.abs(f.z_score).toFixed(1)} SD {f.direction} avg
              </span>
            </div>
          ))}
        </div>
      )}

      {pred.outcome_verified && (
        <div className="mt-3 pt-2 border-t border-gray-700 text-xs">
          {pred.missed_game_actual ? (
            <span className="text-red-400">Missed game (confirmed)</span>
          ) : (
            <span className="text-green-400">Played (no missed game)</span>
          )}
        </div>
      )}
    </div>
  );
}
