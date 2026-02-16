"use client";

import { useEffect, useState } from "react";
import { getTodayPredictions, type Prediction } from "@/lib/api";
import PlayerCard from "./PlayerCard";
import StatCard from "./StatCard";
import { AlertTriangle, Users, Activity, TrendingUp } from "lucide-react";

const TodaysPredictions = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTodayPredictions()
      .then(setPredictions)
      .catch(() => setPredictions([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        Loading predictions...
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No predictions available. Run the daily prediction pipeline first.
      </div>
    );
  }

  const highRisk = predictions.filter((p) => p.risk_tier === "high");
  const moderateRisk = predictions.filter((p) => p.risk_tier === "medium");
  const avgRisk = predictions.reduce((s, p) => s + p.risk_score, 0) / predictions.length;

  return (
    <div className="space-y-8">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Scanned" value={predictions.length} icon={Users} subtitle="active players" />
        <StatCard label="High Risk" value={highRisk.length} icon={AlertTriangle} subtitle="players flagged" accent />
        <StatCard label="Moderate Risk" value={moderateRisk.length} icon={Activity} subtitle="monitoring" />
        <StatCard label="Avg Risk Score" value={`${(avgRisk * 100).toFixed(1)}%`} icon={TrendingUp} />
      </div>

      {/* High Risk */}
      {highRisk.length > 0 && (
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-2 h-2 rounded-full bg-risk-high animate-pulse-glow" />
            <h2 className="font-display font-bold text-xl text-foreground">High Risk Players</h2>
            <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-mono bg-risk-high/10 text-risk-high">
              {highRisk.length}
            </span>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {highRisk.map((p) => (
              <PlayerCard key={p.player_id} player={p} />
            ))}
          </div>
        </section>
      )}

      {/* Moderate Risk */}
      {moderateRisk.length > 0 && (
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-2 h-2 rounded-full bg-risk-moderate" />
            <h2 className="font-display font-bold text-xl text-foreground">Moderate Risk Players</h2>
            <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-mono bg-risk-moderate/10 text-risk-moderate">
              {moderateRisk.length}
            </span>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {moderateRisk.map((p) => (
              <PlayerCard key={p.player_id} player={p} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
};

export default TodaysPredictions;
