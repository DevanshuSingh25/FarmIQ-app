import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { useNavigate } from "react-router-dom";
import {
  Beaker,
  Stethoscope,
  TrendingUp,
  HandHeart,
  ChevronRight
} from "lucide-react";

export function ActionButtons() {
  const navigate = useNavigate();

  const actions = [
    {
      id: 'soil',
      title: 'Soil analysis',
      description: 'Test soil health and get recommendations',
      icon: Beaker,
      color: 'success',
      route: '/soil-analysis'
    },
    {
      id: 'disease',
      title: 'Crop disease',
      description: 'Detect and prevent crop diseases',
      icon: Stethoscope,
      color: 'destructive',
      route: '/farmer/crop-disease'
    },
    {
      id: 'market',
      title: 'Market data',
      description: 'Get latest market prices and trends',
      icon: TrendingUp,
      color: 'primary',
      route: '/market-prices'
    },
    {
      id: 'ngo',
      title: 'Government scheme',
      description: 'Access government schemes and support',
      icon: HandHeart,
      color: 'accent',
      route: '/farmer/ngo-schemes'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {actions.map((action) => {
        const IconComponent = action.icon;

        const getColorClasses = (color: string) => {
          switch (color) {
            case 'success':
              return {
                bg: 'bg-success/10 dark:bg-success/20',
                border: 'border-success/20 dark:border-success/30',
                gradient: 'from-success to-success/80',
                icon: 'text-success'
              };
            case 'destructive':
              return {
                bg: 'bg-destructive/10 dark:bg-destructive/20',
                border: 'border-destructive/20 dark:border-destructive/30',
                gradient: 'from-destructive to-destructive/80',
                icon: 'text-destructive'
              };
            case 'primary':
              return {
                bg: 'bg-primary/10 dark:bg-primary/20',
                border: 'border-primary/20 dark:border-primary/30',
                gradient: 'from-primary to-primary/80',
                icon: 'text-primary'
              };
            case 'accent':
              return {
                bg: 'bg-accent/10 dark:bg-accent/20',
                border: 'border-accent/20 dark:border-accent/30',
                gradient: 'from-accent to-accent/80',
                icon: 'text-accent'
              };
            default:
              return {
                bg: 'bg-muted',
                border: 'border-border',
                gradient: 'from-muted to-muted/80',
                icon: 'text-muted-foreground'
              };
          }
        };

        const colorClasses = getColorClasses(action.color);

        return (
          <Card
            key={action.id}
            className={`${colorClasses.bg} ${colorClasses.border} border-2 hover:shadow-strong hover:scale-105 transition-all duration-300 cursor-pointer group overflow-hidden`}
            onClick={() => navigate(action.route)}
          >
            <CardContent className="p-6 relative group">
              <div className="absolute top-2 right-2">
                <SectionSpeaker
                  sectionId={`action-${action.id}`}
                  getText={() => `${action.title}. ${action.description}`}
                  ariaLabel={`Listen to ${action.title} information`}
                  alwaysVisible={true}
                />
              </div>

              <div className="flex flex-col items-center text-center space-y-4">
                <div className={`p-4 rounded-full bg-gradient-to-br ${colorClasses.gradient} shadow-medium group-hover:shadow-strong transition-all`}>
                  <IconComponent className="h-8 w-8 text-white" />
                </div>

                <div className="space-y-2">
                  <h3 className="text-lg font-semibold text-foreground group-hover:text-primary transition-colors">
                    {action.title}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {action.description}
                  </p>
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  className="w-full mt-4 hover:bg-primary hover:text-primary-foreground transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(action.route);
                  }}
                >
                  Get Started
                  <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}