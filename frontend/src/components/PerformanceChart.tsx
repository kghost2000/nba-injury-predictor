"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { BatchMetric } from "@/lib/api";

export default function PerformanceChart({
  data,
}: {
  data: BatchMetric[];
}) {
  const chartData = data.map((d) => ({
    date: d.metric_date,
    "ROC-AUC": d.roc_auc,
    "PR-AUC": d.pr_auc,
    "High Tier Hit Rate": d.high_tier_hit_rate,
  }));

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-4">
        Model Performance Over Time
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#9CA3AF", fontSize: 12 }}
            tickFormatter={(v) => v.slice(5)}
          />
          <YAxis
            tick={{ fill: "#9CA3AF", fontSize: 12 }}
            domain={[0, 1]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1F2937",
              border: "1px solid #374151",
              borderRadius: "0.5rem",
              color: "#F3F4F6",
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="ROC-AUC"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="PR-AUC"
            stroke="#10B981"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="High Tier Hit Rate"
            stroke="#EF4444"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
