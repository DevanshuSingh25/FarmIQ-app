import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Bell, BellOff, Smartphone, AlertTriangle, Sunrise, Sprout } from 'lucide-react';
import { NotificationSettings as NotificationSettingsType } from '@/types/weather';
import { useToast } from '@/hooks/use-toast';

interface NotificationSettingsProps {
  settings: NotificationSettingsType;
  onSettingsChange: (settings: NotificationSettingsType) => void;
}

export function NotificationSettings({ settings, onSettingsChange }: NotificationSettingsProps) {
  const { toast } = useToast();
  const [isEnabling, setIsEnabling] = useState(false);

  const handleEnableNotifications = async () => {
    setIsEnabling(true);
    
    try {
      // Request notification permission
      if ('Notification' in window) {
        const permission = await Notification.requestPermission();
        
        if (permission === 'granted') {
          onSettingsChange({ ...settings, enabled: true });
          toast({
            title: "Notifications enabled",
            description: "You'll receive weather alerts and farming updates",
          });
        } else {
          toast({
            title: "Notifications blocked",
            description: "Please enable notifications in your browser settings",
            variant: "destructive",
          });
        }
      } else {
        toast({
          title: "Notifications not supported",
          description: "Your browser doesn't support notifications",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Failed to enable notifications",
        description: "Please try again",
        variant: "destructive",
      });
    } finally {
      setIsEnabling(false);
    }
  };

  const handleSettingChange = (key: keyof NotificationSettingsType, value: boolean) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  const sendTestNotification = () => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('FarmIQ Weather Alert', {
        body: 'This is a test notification from your weather app.',
        icon: '/favicon.ico'
      });
      
      toast({
        title: "Test notification sent",
        description: "Check your system notifications",
      });
    } else {
      toast({
        title: "Notifications not enabled",
        description: "Please enable notifications first",
        variant: "destructive",
      });
    }
  };

  return (
    <Card className="shadow-soft">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {settings.enabled ? (
            <Bell className="h-5 w-5 text-success" />
          ) : (
            <BellOff className="h-5 w-5 text-muted-foreground" />
          )}
          Notification Settings
        </CardTitle>
        <CardDescription>
          Get timely alerts for weather changes and farming activities
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {!settings.enabled && (
          <div className="p-4 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-3 mb-3">
              <Smartphone className="h-5 w-5 text-primary" />
              <div>
                <h4 className="font-medium text-foreground">Enable Notifications</h4>
                <p className="text-sm text-muted-foreground">
                  Stay informed about important weather changes that affect your crops
                </p>
              </div>
            </div>
            <Button 
              onClick={handleEnableNotifications}
              disabled={isEnabling}
              className="w-full"
            >
              {isEnabling ? 'Requesting permission...' : 'Enable Notifications'}
            </Button>
          </div>
        )}

        {settings.enabled && (
          <div className="space-y-4">
            <div className="flex items-center justify-between space-x-2">
              <div className="flex items-center gap-3">
                <Sunrise className="h-4 w-4 text-accent" />
                <Label htmlFor="daily-summary" className="text-sm font-medium">
                  Daily Summary (6 AM)
                </Label>
              </div>
              <Switch
                id="daily-summary"
                checked={settings.dailySummary}
                onCheckedChange={(checked) => handleSettingChange('dailySummary', checked)}
              />
            </div>
            
            <div className="flex items-center justify-between space-x-2">
              <div className="flex items-center gap-3">
                <AlertTriangle className="h-4 w-4 text-warning" />
                <Label htmlFor="severe-weather" className="text-sm font-medium">
                  Severe Weather Alerts
                </Label>
              </div>
              <Switch
                id="severe-weather"
                checked={settings.severeWeather}
                onCheckedChange={(checked) => handleSettingChange('severeWeather', checked)}
              />
            </div>
            
            <div className="flex items-center justify-between space-x-2">
              <div className="flex items-center gap-3">
                <Sprout className="h-4 w-4 text-success" />
                <Label htmlFor="sowing-alerts" className="text-sm font-medium">
                  Sowing Window Alerts
                </Label>
              </div>
              <Switch
                id="sowing-alerts"
                checked={settings.sowingAlerts}
                onCheckedChange={(checked) => handleSettingChange('sowingAlerts', checked)}
              />
            </div>

            <div className="pt-4 border-t">
              <Button 
                variant="outline" 
                onClick={sendTestNotification}
                className="w-full"
              >
                Send Test Notification
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}