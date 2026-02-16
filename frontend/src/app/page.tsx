"use client";

import { useState } from "react";
import { Activity, BarChart3, Search, Clock, Cpu } from "lucide-react";
import TodaysPredictions from "@/components/dashboard/TodaysPredictions";
import ModelPerformance from "@/components/dashboard/ModelPerformance";
import PlayerLookup from "@/components/dashboard/PlayerLookup";

const tabs = [
  { id: "today", label: "Today's Predictions", icon: Activity },
  { id: "performance", label: "Model Performance", icon: BarChart3 },
  { id: "lookup", label: "Player Lookup", icon: Search },
] as const;

type TabId = (typeof tabs)[number]["id"];

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("today");

  return (
    <div className="min-h-screen bg-background court-pattern">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/40 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h1 className="font-display font-bold text-xl text-foreground tracking-tight">
                  NBA Injury Predictor
                </h1>
                <p className="text-xs text-muted-foreground">
                  Machine learning-powered injury risk assessment
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <Clock className="w-3 h-3" /> Daily predictions
              </span>
              <span className="px-2 py-1 rounded-md bg-secondary font-mono text-[11px] flex items-center gap-1.5">
                <Cpu className="w-3 h-3" /> LightGBM v1.0
              </span>
              <span className="px-2 py-1 rounded-md bg-primary/10 text-primary font-mono text-[11px]">
                AUC 0.738
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b border-border/50 bg-card/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <nav className="flex gap-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? "border-primary text-primary"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {activeTab === "today" && (
          <div className="mb-6">
            <h2 className="font-display font-bold text-2xl text-foreground">
              {new Date().toLocaleDateString("en-US", {
                weekday: "long",
                month: "long",
                day: "numeric",
                year: "numeric",
              })}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Showing injury risk predictions for today&apos;s and tomorrow&apos;s games
            </p>
          </div>
        )}
        {activeTab === "performance" && (
          <div className="mb-6">
            <h2 className="font-display font-bold text-2xl text-foreground">
              Model Performance
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Last 30 days &middot; LightGBM v1.0
            </p>
          </div>
        )}
        {activeTab === "lookup" && (
          <div className="mb-6">
            <h2 className="font-display font-bold text-2xl text-foreground">
              Player Lookup
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Search for any player to view their risk history
            </p>
          </div>
        )}

        {activeTab === "today" && <TodaysPredictions />}
        {activeTab === "performance" && <ModelPerformance />}
        {activeTab === "lookup" && <PlayerLookup />}
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted-foreground">
          <span>Powered by LightGBM &middot; Built by Andy</span>
          <span className="font-mono">Portfolio Project</span>
        </div>
      </footer>
    </div>
  );
}
