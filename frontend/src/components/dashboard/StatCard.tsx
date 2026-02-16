import { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  trend?: number;
  subtitle?: string;
  accent?: boolean;
}

const StatCard = ({ label, value, icon: Icon, trend, subtitle, accent }: StatCardProps) => {
  return (
    <div className={`glass-card p-4 ${accent ? "stat-glow border-primary/20" : ""}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-muted-foreground/60" />}
      </div>
      <div className="font-display font-black text-2xl text-foreground">{value}</div>
      <div className="flex items-center gap-2 mt-1">
        {trend !== undefined && (
          <span className={`text-xs font-mono font-semibold ${trend >= 0 ? "text-risk-low" : "text-risk-high"}`}>
            {trend >= 0 ? "\u2191" : "\u2193"} {Math.abs(trend).toFixed(1)}%
          </span>
        )}
        {subtitle && <span className="text-[11px] text-muted-foreground">{subtitle}</span>}
      </div>
    </div>
  );
};

export default StatCard;
