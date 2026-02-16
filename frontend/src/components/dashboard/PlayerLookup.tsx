"use client";

import { useState, useMemo, useEffect } from "react";
import { getTodayPredictions, getPlayerDetail, type Prediction, type PlayerDetail } from "@/lib/api";
import { Search, User, Calendar, TrendingUp } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

const PlayerLookup = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<PlayerDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    getTodayPredictions()
      .then(setPredictions)
      .catch(() => setPredictions([]))
      .finally(() => setLoadingList(false));
  }, []);

  useEffect(() => {
    if (selectedId === null) {
      setDetail(null);
      return;
    }
    setLoadingDetail(true);
    getPlayerDetail(selectedId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoadingDetail(false));
  }, [selectedId]);

  const filtered = useMemo(() => {
    if (!query) return predictions;
    return predictions.filter((p) =>
      p.player_name.toLowerCase().includes(query.toLowerCase())
    );
  }, [query, predictions]);

  const chartData = useMemo(() => {
    if (!detail) return [];
    return detail.predictions
      .slice()
      .reverse()
      .map((p) => ({
        date: p.prediction_date,
        risk: p.risk_score,
      }));
  }, [detail]);

  const selectedPlayer = predictions.find((p) => p.player_id === selectedId);

  if (loadingList) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        Loading players...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <input
          type="text"
          placeholder="Search player name..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-3 rounded-lg bg-secondary border border-border text-foreground placeholder:text-muted-foreground/50 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 transition"
        />
      </div>

      {selectedId === null && (
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3">
          {filtered.map((p) => (
            <button
              key={p.player_id}
              onClick={() => setSelectedId(p.player_id)}
              className="glass-card-hover p-4 text-left"
            >
              <div className="font-display font-bold text-foreground">{p.player_name}</div>
              <div className="text-xs text-muted-foreground mt-1">
                {p.team_abbreviation && (
                  <span className="font-mono">{p.team_abbreviation}</span>
                )}
              </div>
            </button>
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full text-center py-8 text-muted-foreground">
              No players found matching &ldquo;{query}&rdquo;
            </div>
          )}
        </div>
      )}

      {selectedId !== null && (
        <div className="space-y-6">
          <button
            onClick={() => setSelectedId(null)}
            className="text-xs text-primary hover:underline"
          >
            &larr; Back to all players
          </button>

          {loadingDetail ? (
            <div className="text-center py-12 text-muted-foreground">
              Loading player details...
            </div>
          ) : detail ? (
            <>
              {/* Profile Card */}
              <div className="glass-card p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-full bg-secondary flex items-center justify-center">
                      <User className="w-6 h-6 text-muted-foreground" />
                    </div>
                    <div>
                      <h2 className="font-display font-bold text-2xl">{detail.player_name}</h2>
                      <div className="flex gap-3 text-sm text-muted-foreground mt-1">
                        {detail.team_abbreviation && (
                          <span className="font-mono">{detail.team_abbreviation}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  {selectedPlayer && (
                    <div className="text-right">
                      <div
                        className={`font-display font-black text-4xl ${
                          selectedPlayer.risk_tier === "high"
                            ? "text-risk-high"
                            : selectedPlayer.risk_tier === "medium"
                            ? "text-risk-moderate"
                            : "text-risk-low"
                        }`}
                      >
                        {Math.round(selectedPlayer.risk_score * 100)}%
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">Current Risk</div>
                    </div>
                  )}
                </div>
              </div>

              {/* Risk Over Time */}
              {chartData.length > 1 && (
                <div className="glass-card p-6">
                  <h3 className="font-display font-bold text-lg mb-4 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-primary" />
                    Risk Score Trend
                  </h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: "hsl(215,16%,47%)" }}
                        tickFormatter={(v) => v.slice(5)}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tick={{ fontSize: 10, fill: "hsl(215,16%,47%)" }}
                        tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#fff",
                          border: "1px solid hsl(220,13%,87%)",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                        formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                      />
                      <Line type="monotone" dataKey="risk" stroke="#06b6d4" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Prediction History */}
              {detail.predictions.length > 0 && (
                <div className="glass-card p-6">
                  <h3 className="font-display font-bold text-lg mb-4 flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-primary" />
                    Prediction History
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border/50 text-left text-xs text-muted-foreground uppercase tracking-wider">
                          <th className="pb-3 pr-4">Date</th>
                          <th className="pb-3 pr-4">Risk Score</th>
                          <th className="pb-3 pr-4">Tier</th>
                          <th className="pb-3">Outcome</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/30">
                        {detail.predictions.map((p, i) => (
                          <tr key={i} className="hover:bg-secondary/30 transition-colors">
                            <td className="py-2.5 pr-4 text-muted-foreground">{p.prediction_date}</td>
                            <td className="py-2.5 pr-4">
                              <span
                                className={`font-mono font-semibold ${
                                  p.risk_tier === "high"
                                    ? "text-risk-high"
                                    : p.risk_tier === "medium"
                                    ? "text-risk-moderate"
                                    : "text-risk-low"
                                }`}
                              >
                                {(p.risk_score * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="py-2.5 pr-4 text-muted-foreground capitalize">{p.risk_tier}</td>
                            <td className="py-2.5 text-muted-foreground">
                              {p.outcome_verified
                                ? p.missed_game_actual
                                  ? <span className="text-risk-high">Missed game</span>
                                  : <span className="text-risk-low">Played</span>
                                : <span className="text-muted-foreground/50">Pending</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              Could not load player details.
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PlayerLookup;
