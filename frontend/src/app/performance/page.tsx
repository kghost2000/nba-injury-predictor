"use client";

import { useEffect, useState } from "react";
import {
  getPerformanceSummary,
  getPerformanceHistory,
  type PerformanceSummary,
  type BatchMetric,
} from "@/lib/api";
import MetricsDisplay from "@/components/MetricsDisplay";
import PerformanceChart from "@/components/PerformanceChart";

export default function PerformancePage() {
  const [summary, setSummary] = useState<PerformanceSummary | null>(null);
  const [history, setHistory] = useState<BatchMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getPerformanceSummary(30), getPerformanceHistory(30)])
      .then(([s, h]) => {
        setSummary(s);
        setHistory(h);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-gray-500">Loading performance data...</div>;
  }

  if (!summary) {
    return <div className="text-center py-12 text-gray-500">Could not load performance data.</div>;
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">
          Model Performance
        </h1>
        <p className="text-sm text-gray-400">
          Last {summary.window_days} days | {summary.n_batches} evaluation
          batches
        </p>
      </div>

      <div className="space-y-6">
        <MetricsDisplay summary={summary} />

        {history.length > 0 ? (
          <PerformanceChart data={history} />
        ) : (
          <div className="text-center py-12 text-gray-500 bg-gray-900 rounded-lg border border-gray-800">
            No performance data yet. Run the validation pipeline for 3+ days
            to see metrics.
          </div>
        )}

        {history.length > 0 && (
          <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-800">
                <tr>
                  <th className="px-4 py-2 text-left text-gray-400">Date</th>
                  <th className="px-4 py-2 text-right text-gray-400">
                    Predictions
                  </th>
                  <th className="px-4 py-2 text-right text-gray-400">
                    ROC-AUC
                  </th>
                  <th className="px-4 py-2 text-right text-gray-400">
                    PR-AUC
                  </th>
                  <th className="px-4 py-2 text-right text-gray-400">
                    High Hit Rate
                  </th>
                  <th className="px-4 py-2 text-right text-gray-400">
                    Base Rate
                  </th>
                </tr>
              </thead>
              <tbody>
                {history
                  .slice()
                  .reverse()
                  .map((m) => (
                    <tr
                      key={m.metric_date}
                      className="border-t border-gray-800"
                    >
                      <td className="px-4 py-2 text-gray-300">
                        {m.metric_date}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">
                        {m.n_predictions}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">
                        {m.roc_auc?.toFixed(3) ?? "N/A"}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">
                        {m.pr_auc?.toFixed(3) ?? "N/A"}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">
                        {m.high_tier_hit_rate != null
                          ? `${(m.high_tier_hit_rate * 100).toFixed(1)}%`
                          : "N/A"}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">
                        {m.positive_rate != null
                          ? `${(m.positive_rate * 100).toFixed(1)}%`
                          : "N/A"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
