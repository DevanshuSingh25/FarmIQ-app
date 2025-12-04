import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  color: "success" | "accent" | "warning" | "primary";
  isActive?: boolean;
  onClick?: () => void;
}

export const FeatureCard = ({ 
  icon: Icon, 
  title, 
  description, 
  color, 
  isActive = false, 
  onClick 
}: FeatureCardProps) => {
  const getText = () => `${title}. ${description}`;

  return (
    <Card 
      className={`cursor-pointer transition-smooth hover:shadow-medium group ${
        isActive ? 'ring-2 ring-primary shadow-glow scale-105' : 'hover:scale-102'
      }`}
      onClick={onClick}
    >
      <CardHeader className="pb-4">
        <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 transition-smooth group-hover:scale-110 ${
          color === 'success' ? 'bg-success/10 group-hover:bg-success/20' :
          color === 'accent' ? 'bg-accent/10 group-hover:bg-accent/20' :
          color === 'warning' ? 'bg-warning/10 group-hover:bg-warning/20' :
          'bg-primary/10 group-hover:bg-primary/20'
        }`}>
          <Icon className={`h-6 w-6 transition-smooth ${
            color === 'success' ? 'text-success' :
            color === 'accent' ? 'text-accent' :
            color === 'warning' ? 'text-warning' :
            'text-primary'
          }`} />
        </div>
        <div className="flex items-center justify-between">
          <CardTitle 
            className="text-lg group-hover:text-primary transition-smooth"
            data-tts="title"
          >
            {title}
          </CardTitle>
          <SectionSpeaker 
            getText={getText}
            sectionId={`feature-${title.toLowerCase().replace(/\s+/g, '-')}`}
            ariaLabel={`Read ${title} section`}
          />
        </div>
      </CardHeader>
      <CardContent>
        <CardDescription 
          className="text-sm leading-relaxed"
          data-tts="desc"
        >
          {description}
        </CardDescription>
      </CardContent>
    </Card>
  );
};