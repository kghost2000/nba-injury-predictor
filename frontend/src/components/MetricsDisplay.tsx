import type { PerformanceSummary } from "@/lib/api";

function MetricCard({
  label,
  value,
  format = "percent",
}: {
  label: string;
  value: number | null;
  format?: "percent" | "number";
}) {
  const display =
    value === null
      ? "N/A"
      : format === "percent"
        ? `${(value * 100).toFixed(1)}%`
        : value.toLocaleString();

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
        {label}
      </div>
      <div className="text-2xl font-bold text-white">{display}</div>
    </div>
  );
}

export default function MetricsDisplay({
  summary,
}: {
  summary: PerformanceSummary;
}) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard label="Avg ROC-AUC" value={summary.avg_roc_auc} />
      <MetricCard label="Avg PR-AUC" value={summary.avg_pr_auc} />
      <MetricCard label="Avg Precision" value={summary.avg_precision} />
      <MetricCard label="Avg Recall" value={summary.avg_recall} />
      <MetricCard label="Avg F1" value={summary.avg_f1} />
      <MetricCard label="Avg Base Rate" value={summary.avg_positive_rate} />
      <MetricCard
        label="High Tier Hit Rate"
        value={summary.avg_high_tier_hit_rate}
      />
      <MetricCard
        label="Total Predictions"
        value={summary.total_predictions}
        format="number"
      />
    </div>
  );
}
