import { LucideIcon } from "lucide-react";
import { SectionSpeaker } from "@/components/ui/section-speaker";

interface StatsCardProps {
  icon: LucideIcon;
  value: string;
  label: string;
  trend?: "up" | "down" | "neutral";
}

export const StatsCard = ({ 
  icon: Icon, 
  value, 
  label, 
  trend = "neutral" 
}: StatsCardProps) => {
  const getText = () => `${value} ${label}`;

  return (
    <div className="text-center group">
      <div className="inline-flex items-center justify-center w-12 h-12 bg-primary/10 rounded-full mb-4 transition-smooth group-hover:bg-primary/20 group-hover:scale-110">
        <Icon className="h-6 w-6 text-primary" />
      </div>
      <div className="relative">
        <div className="absolute right-0 top-0">
          <SectionSpeaker 
            getText={getText}
            sectionId={`stat-${label.toLowerCase().replace(/\s+/g, '-')}`}
            ariaLabel={`Read ${label} statistic`}
          />
        </div>
        <div 
          className="text-3xl font-bold text-primary transition-smooth group-hover:scale-105 mb-2"
          data-tts="title"
        >
          {value}
        </div>
        <div 
          className="text-sm text-muted-foreground leading-relaxed"
          data-tts="desc"
        >
          {label}
        </div>
      </div>
      {trend !== "neutral" && (
        <div className={`text-xs mt-1 ${
          trend === "up" ? "text-success" : "text-warning"
        }`}>
          {trend === "up" ? "↗ Improving" : "↘ Needs attention"}
        </div>
      )}
    </div>
  );
};