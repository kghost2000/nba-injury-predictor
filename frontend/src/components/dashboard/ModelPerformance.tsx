"use client";

import { useEffect, useState } from "react";
import { getPerformanceSummary, getPerformanceHistory, type PerformanceSummary, type BatchMetric } from "@/lib/api";
import StatCard from "./StatCard";
import { Target, Crosshair, Gauge, BarChart3, Database, TrendingUp } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

const ModelPerformance = () => {
  const [summary, setSummary] = useState<PerformanceSummary | null>(null);
  const [history, setHistory] = useState<BatchMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getPerformanceSummary(), getPerformanceHistory()])
      .then(([s, h]) => {
        setSummary(s);
        setHistory(h);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        Loading performance data...
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No performance data available yet. Predictions need outcome verification first.
      </div>
    );
  }

  const chartData = history.map((h) => ({
    date: h.prediction_date,
    precision: h.precision_score,
    recall: h.recall_score,
    f1: h.f1_score,
  }));

  return (
    <div className="space-y-8">
      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard
          label="Precision"
          value={summary.avg_precision?.toFixed(2) ?? "N/A"}
          icon={Target}
          accent
        />
        <StatCard
          label="Recall"
          value={summary.avg_recall?.toFixed(2) ?? "N/A"}
          icon={Crosshair}
        />
        <StatCard
          label="F1 Score"
          value={summary.avg_f1?.toFixed(2) ?? "N/A"}
          icon={Gauge}
        />
        <StatCard
          label="Base Rate"
          value={summary.avg_positive_rate?.toFixed(3) ?? "N/A"}
          icon={BarChart3}
        />
        <StatCard
          label="AUC"
          value={summary.avg_roc_auc?.toFixed(3) ?? "N/A"}
          icon={TrendingUp}
          subtitle="ROC curve"
          accent
        />
        <StatCard
          label="Total Preds"
          value={summary.total_predictions.toLocaleString()}
          icon={Database}
          subtitle={`${summary.n_batches} batches`}
        />
      </div>

      {/* Performance Over Time */}
      {chartData.length > 0 && (
        <div className="glass-card p-6">
          <h3 className="font-display font-bold text-lg mb-4">Performance Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220,13%,87%)" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "hsl(215,16%,47%)" }}
                tickFormatter={(v) => v.slice(5)}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 11, fill: "hsl(215,16%,47%)" }}
              />
              <Tooltip
                contentStyle={{
                  background: "#fff",
                  border: "1px solid hsl(220,13%,87%)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Line type="monotone" dataKey="precision" stroke="#06b6d4" strokeWidth={2} dot={false} name="Precision" />
              <Line type="monotone" dataKey="recall" stroke="#f59e0b" strokeWidth={2} dot={false} name="Recall" />
              <Line type="monotone" dataKey="f1" stroke="#22c55e" strokeWidth={2} dot={false} name="F1" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-6 mt-2 justify-center text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-[#06b6d4] inline-block rounded" /> Precision
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-[#f59e0b] inline-block rounded" /> Recall
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-[#22c55e] inline-block rounded" /> F1
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelPerformance;
