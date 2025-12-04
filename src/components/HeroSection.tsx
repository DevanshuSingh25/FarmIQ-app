import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { PlayCircle, Phone } from "lucide-react";

interface HeroSectionProps {
  backgroundImage: string;
  badge?: string;
  title: string;
  description: string;
  primaryAction: {
    text: string;
    icon?: React.ComponentType<any>;
  };
  secondaryAction: {
    text: string;
    icon?: React.ComponentType<any>;
  };
}

export const HeroSection = ({
  backgroundImage,
  badge,
  title,
  description,
  primaryAction,
  secondaryAction
}: HeroSectionProps) => {
  const getText = () => `${badge ? badge + '. ' : ''}${title}. ${description}`;

  return (
    <section className="relative overflow-hidden group">
      <div className="absolute inset-0 bg-gradient-hero opacity-90" />
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${backgroundImage})` }}
      />
      <div className="absolute top-4 right-4 z-10">
        <SectionSpeaker 
          getText={getText}
          sectionId="hero-section"
          ariaLabel="Read hero section"
          alwaysVisible
          className="text-primary-foreground hover:text-secondary"
        />
      </div>
      <div className="relative container mx-auto px-4 py-20 lg:py-32">
        <div className="max-w-3xl">
          {badge && (
            <Badge variant="secondary" className="px-4 py-2 text-sm font-medium mb-6">
              {badge}
            </Badge>
          )}
          <h1 className="text-4xl lg:text-6xl font-bold text-primary-foreground mb-6">
            {title}
          </h1>
          <p className="text-xl text-primary-foreground/90 leading-relaxed mb-8">
            {description}
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <Button variant="secondary" size="lg" className="text-lg px-8 py-6">
              {primaryAction.icon && <primaryAction.icon className="mr-2 h-5 w-5" />}
              {primaryAction.text}
            </Button>
            <Button variant="secondary" size="lg" className="text-lg px-8 py-6">
              {secondaryAction.icon && <secondaryAction.icon className="mr-2 h-5 w-5" />}
              {secondaryAction.text}
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};