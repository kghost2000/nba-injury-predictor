import { type Prediction, formatFeatureName } from "@/lib/api";
import { CalendarDays } from "lucide-react";

interface PlayerCardProps {
  player: Prediction;
}

const riskDisplay: Record<string, string> = {
  high: "High",
  medium: "Moderate",
  low: "Low",
};

const PlayerCard = ({ player }: PlayerCardProps) => {
  const tier = player.risk_tier;
  const borderClass = tier === "high" ? "risk-border-high" : tier === "medium" ? "risk-border-moderate" : "risk-border-low";
  const riskColorClass = tier === "high" ? "text-risk-high" : tier === "medium" ? "text-risk-moderate" : "text-risk-low";
  const riskBgClass = tier === "high" ? "bg-risk-high" : tier === "medium" ? "bg-risk-moderate" : "bg-risk-low";

  return (
    <div className={`glass-card-hover ${borderClass} p-5`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-display font-bold text-lg text-foreground">{player.player_name}</h3>
          <div className="flex items-center gap-2 mt-0.5">
            {player.team_abbreviation && (
              <span className="font-mono text-xs text-muted-foreground">{player.team_abbreviation}</span>
            )}
          </div>
        </div>
        <div className="text-right">
          <div className={`font-display font-black text-3xl ${riskColorClass}`}>
            {Math.round(player.risk_score * 100)}%
          </div>
          <span className={`inline-block mt-1 px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider ${riskBgClass}/15 ${riskColorClass}`}>
            {riskDisplay[tier] || tier} risk
          </span>
        </div>
      </div>

      <div className="space-y-2 mb-3">
        {player.top_factors.slice(0, 3).map((factor, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="flex-1">
              <div className="flex justify-between text-xs mb-0.5">
                <span className="text-muted-foreground">{formatFeatureName(factor.feature)}</span>
                <span className="font-mono text-foreground/80">
                  {factor.feature === "is_back_to_back" || factor.feature === "back_to_back"
                    ? "Yes"
                    : `${factor.z_score > 0 ? "+" : ""}${factor.z_score} SD`}
                </span>
              </div>
              <div className="h-1 bg-secondary rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${riskBgClass}/60`}
                  style={{ width: `${Math.min(factor.importance_score * 100 * 3, 100)}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-1.5 text-xs text-muted-foreground pt-2 border-t border-border/50">
        <CalendarDays className="w-3 h-3" />
        <span>{player.prediction_date}</span>
        {player.outcome_verified && (
          <span className="ml-auto">
            {player.missed_game_actual ? (
              <span className="text-risk-high">Missed game</span>
            ) : (
              <span className="text-risk-low">Played</span>
            )}
          </span>
        )}
      </div>
    </div>
  );
};

export default PlayerCard;
