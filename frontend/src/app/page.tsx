"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getTodayPredictions, type Prediction } from "@/lib/api";
import PredictionCard from "@/components/PredictionCard";

export default function Home() {
  const searchParams = useSearchParams();
  const tier = searchParams.get("tier") || undefined;
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTodayPredictions(tier)
      .then(setPredictions)
      .catch(() => setPredictions([]))
      .finally(() => setLoading(false));
  }, [tier]);

  const highRisk = predictions.filter((p) => p.risk_tier === "high");
  const mediumRisk = predictions.filter((p) => p.risk_tier === "medium");
  const lowRisk = predictions.filter((p) => p.risk_tier === "low");

  if (loading) {
    return <div className="text-center py-12 text-gray-500">Loading predictions...</div>;
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">
          Today&apos;s Injury Risk Predictions
        </h1>
        <p className="text-sm text-gray-400">
          {predictions.length} players evaluated
          {predictions.length > 0 &&
            ` | ${predictions[0].prediction_date}`}
        </p>
      </div>

      {/* Tier filter tabs */}
      <div className="flex gap-2 mb-6">
        {[
          { label: "All", value: "" },
          { label: `High (${highRisk.length})`, value: "high" },
          { label: `Medium (${mediumRisk.length})`, value: "medium" },
          { label: `Low (${lowRisk.length})`, value: "low" },
        ].map((tab) => (
          <a
            key={tab.value}
            href={tab.value ? `/?tier=${tab.value}` : "/"}
            className={`px-3 py-1.5 rounded text-sm transition ${
              (tier || "") === tab.value
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            {tab.label}
          </a>
        ))}
      </div>

      {predictions.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          No predictions available. Run the daily prediction pipeline first.
        </div>
      ) : (
        <>
          {(!tier || tier === "high") && highRisk.length > 0 && (
            <section className="mb-8">
              <h2 className="text-lg font-semibold text-red-400 mb-3">
                High Risk ({highRisk.length})
              </h2>
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {highRisk.map((p) => (
                  <PredictionCard key={p.player_id} pred={p} />
                ))}
              </div>
            </section>
          )}

          {(!tier || tier === "medium") && mediumRisk.length > 0 && (
            <section className="mb-8">
              <h2 className="text-lg font-semibold text-yellow-400 mb-3">
                Medium Risk ({mediumRisk.length})
              </h2>
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {mediumRisk.map((p) => (
                  <PredictionCard key={p.player_id} pred={p} />
                ))}
              </div>
            </section>
          )}

          {tier === "low" && lowRisk.length > 0 && (
            <section className="mb-8">
              <h2 className="text-lg font-semibold text-green-400 mb-3">
                Low Risk ({lowRisk.length})
              </h2>
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {lowRisk.map((p) => (
                  <PredictionCard key={p.player_id} pred={p} />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
