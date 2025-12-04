import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { ScrollArea } from '@/components/ui/scroll-area';
import { CloudRain, Wind, Thermometer } from 'lucide-react';
import { HourlyForecast } from '@/types/weather';

interface HourlyForecastPanelProps {
  isOpen: boolean;
  onClose: () => void;
  hourlyData: HourlyForecast[];
  date: string;
  units: { temperature: 'C' | 'F'; wind: 'kph' | 'mph' };
}

export function HourlyForecastPanel({ 
  isOpen, 
  onClose, 
  hourlyData, 
  date, 
  units 
}: HourlyForecastPanelProps) {
  const convertTemp = (tempC: number) => {
    return units.temperature === 'F' ? Math.round((tempC * 9/5) + 32) : tempC;
  };
  
  const convertWind = (windKph: number) => {
    return units.wind === 'mph' ? Math.round(windKph * 0.621371) : windKph;
  };

  const formatTime = (dateTime: string) => {
    return new Date(dateTime).toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      hour12: true 
    });
  };

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent side="right" className="w-full sm:max-w-lg">
        <SheetHeader>
          <SheetTitle>Hourly Forecast</SheetTitle>
          <SheetDescription>
            {new Date(date).toLocaleDateString('en-US', { 
              weekday: 'long', 
              month: 'long', 
              day: 'numeric' 
            })}
          </SheetDescription>
        </SheetHeader>
        
        <ScrollArea className="h-[calc(100vh-120px)] mt-6">
          <div className="space-y-3 pr-4">
            {hourlyData.map((hour, index) => (
              <div 
                key={index}
                className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
              >
                <div className="flex items-center gap-4">
                  <div className="text-sm font-medium text-foreground min-w-[60px]">
                    {formatTime(hour.dateTime)}
                  </div>
                  
                  <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                      <Thermometer className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium text-foreground">
                        {convertTemp(hour.tempC)}°{units.temperature}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <CloudRain className="h-4 w-4 text-primary" />
                      <span className="text-sm text-muted-foreground">
                        {Math.round(hour.precipChance)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <Wind className="h-4 w-4 text-primary" />
                      <span className="text-sm text-muted-foreground">
                        {convertWind(hour.windKph)} {units.wind}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}