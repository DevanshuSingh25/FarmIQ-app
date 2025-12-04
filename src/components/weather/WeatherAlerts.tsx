import { useState } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  AlertTriangle, 
  CloudRain, 
  Thermometer, 
  Wind, 
  Zap, 
  X,
  Bell,
  BellOff 
} from 'lucide-react';
import { WeatherAlert } from '@/types/weather';

interface WeatherAlertsProps {
  alerts: WeatherAlert[];
  onDismiss: (alertCode: string) => void;
  onSaveReminder: (alert: WeatherAlert) => void;
}

const alertIcons = {
  heavy_rain: CloudRain,
  heat_wave: Thermometer,
  frost: Thermometer,
  high_wind: Wind,
  drought: Thermometer,
  storm: Zap
};

const severityStyles = {
  info: 'border-primary bg-primary/5 text-primary',
  warning: 'border-warning bg-warning/5 text-warning',
  danger: 'border-destructive bg-destructive/5 text-destructive'
};

const severityBadgeStyles = {
  info: 'bg-primary text-primary-foreground',
  warning: 'bg-warning text-warning-foreground',
  danger: 'bg-destructive text-destructive-foreground'
};

export function WeatherAlerts({ alerts, onDismiss, onSaveReminder }: WeatherAlertsProps) {
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());
  const [savedReminders, setSavedReminders] = useState<Set<string>>(new Set());

  const handleDismiss = (alertCode: string) => {
    setDismissedAlerts(prev => new Set(prev).add(alertCode));
    onDismiss(alertCode);
  };

  const handleSaveReminder = (alert: WeatherAlert) => {
    setSavedReminders(prev => new Set(prev).add(alert.code));
    onSaveReminder(alert);
  };

  const visibleAlerts = alerts.filter(alert => !dismissedAlerts.has(alert.code));

  if (visibleAlerts.length === 0) return null;

  return (
    <div className="space-y-3">
      {visibleAlerts.map((alert) => {
        const AlertIcon = alertIcons[alert.code] || AlertTriangle;
        const isReminderSaved = savedReminders.has(alert.code);
        
        return (
          <Alert 
            key={alert.code} 
            className={`${severityStyles[alert.severity]} border-l-4`}
          >
            <div className="flex items-start gap-3 w-full">
              <AlertIcon className="h-5 w-5 mt-0.5 flex-shrink-0" />
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <AlertTitle className="text-base font-semibold flex items-center gap-2">
                    {alert.code.replace('_', ' ').toUpperCase()}
                    <Badge className={`text-xs ${severityBadgeStyles[alert.severity]}`}>
                      {alert.severity}
                    </Badge>
                  </AlertTitle>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDismiss(alert.code)}
                    className="h-6 w-6 p-0 hover:bg-background/20"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                
                <AlertDescription className="text-sm mb-3">
                  {alert.message}
                </AlertDescription>
                
                <div className="flex items-center justify-between">
                  <div className="text-xs text-muted-foreground">
                    {new Date(alert.from).toLocaleDateString()} - {new Date(alert.to).toLocaleDateString()}
                  </div>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleSaveReminder(alert)}
                    disabled={isReminderSaved}
                    className="flex items-center gap-1 text-xs"
                  >
                    {isReminderSaved ? (
                      <>
                        <BellOff className="h-3 w-3" />
                        Saved
                      </>
                    ) : (
                      <>
                        <Bell className="h-3 w-3" />
                        Save reminder
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </Alert>
        );
      })}
    </div>
  );
}