import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { RefreshCw, Thermometer, Droplets, Eye, AlertTriangle, Wifi, WifiOff, Settings, Lightbulb } from 'lucide-react';
import { iotService, Reading } from '@/services/iotService';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/contexts/AuthContext';

interface LiveReadingsProps {
  isInstalled: boolean;
}

export default function LiveReadings({ isInstalled }: LiveReadingsProps) {
  const { toast } = useToast();
  const { user } = useAuth();
  const [readings, setReadings] = useState<Reading[]>([]);

  const [isLoading, setIsLoading] = useState(false);
  const [isOffline, setIsOffline] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [bookingStatus, setBookingStatus] = useState<string | null>(null);
  const [isCheckingStatus, setIsCheckingStatus] = useState(true);
  const [settings, setSettings] = useState({
    temperatureUnit: 'C' as 'C' | 'F',
    dailySummary: true,
    alerts: true,
  });
  const [motorState, setMotorState] = useState<boolean>(false);
  const [isMotorLoading, setIsMotorLoading] = useState(false);

  // Check booking status from iot_reading table
  const checkBookingStatus = async () => {
    if (!user?.id) return;

    setIsCheckingStatus(true);
    try {
      const booking = await iotService.getBookingRequest(user.id);
      setBookingStatus(booking?.status || null);
    } catch (error) {
      console.error('Error checking booking status:', error);
      setBookingStatus(null);
    } finally {
      setIsCheckingStatus(false);
    }
  };

  // Load readings
  const loadData = async () => {
    if (!user?.id || bookingStatus !== 'active') return;

    setIsLoading(true);
    try {
      const readingsData = await iotService.getReadings(user.id);
      setReadings(readingsData);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading readings:', error);
      setIsOffline(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Check booking status on mount
  useEffect(() => {
    checkBookingStatus();
  }, [user?.id]);

  // Load motor state when booking status becomes active
  useEffect(() => {
    if (bookingStatus === 'active') {
      loadMotorState();
    }
  }, [bookingStatus]);

  // Load data when booking status becomes active
  useEffect(() => {
    if (bookingStatus === 'active') {
      loadData();
    }
  }, [bookingStatus]);

  // Auto-refresh every 20 seconds if active
  useEffect(() => {
    if (bookingStatus !== 'active') return;

    const interval = setInterval(loadData, 20 * 1000); // 20 seconds
    return () => clearInterval(interval);
  }, [bookingStatus]);

  // Check online status
  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleRefresh = () => {
    if (bookingStatus === 'active') {
      loadData();
      toast({
        title: 'Refreshed',
        description: 'Readings have been updated.',
      });
    } else {
      checkBookingStatus();
      toast({
        title: 'Status Checked',
        description: 'Booking status has been refreshed.',
      });
    }
  };

  const handleSettingsSave = () => {
    toast({
      title: 'Settings Saved',
      description: 'Your preferences have been updated.',
    });
  };

  const convertTemperature = (celsius: number) => {
    if (settings.temperatureUnit === 'F') {
      return Math.round((celsius * 9 / 5 + 32) * 10) / 10;
    }
    return celsius;
  };

  // Load motor state from Blynk
  const loadMotorState = async () => {
    try {
      const response = await iotService.getBlynkLedStatus();
      setMotorState(response.state);
      console.log('Motor state loaded:', response);
    } catch (error) {
      console.error('Error loading motor state:', error);
    }
  };

  // Toggle motor state via Blynk
  const handleMotorToggle = async (newState: boolean) => {
    setIsMotorLoading(true);
    try {
      const response = await iotService.setBlynkLedState(newState);
      setMotorState(response.state);
      toast({
        title: 'Motor Control',
        description: `Motor turned ${newState ? 'ON' : 'OFF'} successfully`,
      });
      console.log('Motor state updated:', response);
    } catch (error) {
      console.error('Error updating motor state:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to control motor';
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      });
    } finally {
      setIsMotorLoading(false);
    }
  };

  const getTemperatureUnit = () => {
    return settings.temperatureUnit === 'F' ? '°F' : '°C';
  };





  // Generate simple trend data (mock)
  const generateTrendData = (values: number[]) => {
    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min;

    return values.map(value => {
      const normalized = range > 0 ? (value - min) / range : 0.5;
      return Math.round(normalized * 100);
    });
  };

  // Show loading while checking status
  if (isCheckingStatus) {
    return (
      <div className="max-w-2xl mx-auto">
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p className="text-muted-foreground">Checking sensor status...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show "not booked" message if status is not active
  if (bookingStatus !== 'active') {
    return (
      <div className="max-w-2xl mx-auto">
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
              <Settings className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-medium mb-2">Sensor isn't booked yet</h3>
            <p className="text-muted-foreground text-center mb-4">
              Your IoT sensor needs to be booked and activated before you can view live readings.
            </p>
            <Button variant="outline" onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Check Status
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const currentReading = readings[0];
  const temperatureValues = readings.map(r => r.temperature);
  const humidityValues = readings.map(r => r.humidity);
  const moistureValues = readings.map(r => r.soil_moisture);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Offline Banner */}
      {isOffline && (
        <Card className="border-yellow-200 bg-yellow-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2">
              <WifiOff className="h-5 w-5 text-yellow-600" />
              <p className="text-sm text-yellow-800">
                You're offline. Showing last saved readings.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Live Farm Readings</h2>
          <p className="text-muted-foreground">
            Real-time monitoring of your farm conditions
          </p>
        </div>
        <div className="flex items-center gap-2">
          {lastUpdated && (
            <p className="text-sm text-muted-foreground">
              Last updated: {lastUpdated.toLocaleTimeString('en-GB', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
              })}
            </p>
          )}
          <Button
            onClick={handleRefresh}
            disabled={isLoading}
            size="sm"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Reading Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Soil Moisture */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Droplets className="h-5 w-5 text-blue-600" />
                <span className="text-sm font-medium">Soil Moisture</span>
              </div>
              <Badge variant="secondary">{currentReading?.soil_moisture || 0}%</Badge>
            </div>
            <div className="text-3xl font-bold mb-2">
              {currentReading?.soil_moisture || 0}%
            </div>
            <div className="text-xs text-muted-foreground">
              {currentReading?.soil_moisture && currentReading.soil_moisture < 30 ? 'Low' :
                currentReading?.soil_moisture && currentReading.soil_moisture > 70 ? 'High' : 'Normal'}
            </div>
            {/* Mini trend */}
            <div className="flex items-end gap-1 mt-2 h-8">
              {generateTrendData(moistureValues.slice(-12)).map((height, i) => (
                <div
                  key={i}
                  className="bg-blue-200 rounded-sm flex-1"
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Temperature */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Thermometer className="h-5 w-5 text-red-600" />
                <span className="text-sm font-medium">Temperature</span>
              </div>
              <Badge variant="secondary">
                {convertTemperature(currentReading?.temperature || 0)}{getTemperatureUnit()}
              </Badge>
            </div>
            <div className="text-3xl font-bold mb-2">
              {convertTemperature(currentReading?.temperature || 0)}{getTemperatureUnit()}
            </div>
            <div className="text-xs text-muted-foreground">
              {currentReading?.temperature && currentReading.temperature < 15 ? 'Cold' :
                currentReading?.temperature && currentReading.temperature > 35 ? 'Hot' : 'Normal'}
            </div>
            {/* Mini trend */}
            <div className="flex items-end gap-1 mt-2 h-8">
              {generateTrendData(temperatureValues.slice(-12)).map((height, i) => (
                <div
                  key={i}
                  className="bg-red-200 rounded-sm flex-1"
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Humidity */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Eye className="h-5 w-5 text-green-600" />
                <span className="text-sm font-medium">Humidity</span>
              </div>
              <Badge variant="secondary">{currentReading?.humidity || 0}%</Badge>
            </div>
            <div className="text-3xl font-bold mb-2">
              {currentReading?.humidity || 0}%
            </div>
            <div className="text-xs text-muted-foreground">
              {currentReading?.humidity && currentReading.humidity < 40 ? 'Dry' :
                currentReading?.humidity && currentReading.humidity > 80 ? 'Humid' : 'Normal'}
            </div>
            {/* Mini trend */}
            <div className="flex items-end gap-1 mt-2 h-8">
              {generateTrendData(humidityValues.slice(-12)).map((height, i) => (
                <div
                  key={i}
                  className="bg-green-200 rounded-sm flex-1"
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
          </CardContent>
        </Card>


      </div>

      {/* Motor Control Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5" />
            Control your motor in farm
          </CardTitle>
          <CardDescription>Control the LED/motor connected to your IoT device</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
            <div className="space-y-1">
              <Label htmlFor="motor-switch" className="text-base font-medium">
                Motor Control
              </Label>
              <p className="text-sm text-muted-foreground">
                {motorState ? 'Motor is currently ON' : 'Motor is currently OFF'}
              </p>
            </div>
            <div className="flex items-center gap-3">
              {isMotorLoading && (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary"></div>
              )}
              <Switch
                id="motor-switch"
                checked={motorState}
                onCheckedChange={handleMotorToggle}
                disabled={isMotorLoading}
              />
            </div>
          </div>
          <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
            <Wifi className="h-4 w-4" />
            <span>Connected to Blynk IoT Cloud</span>
          </div>
        </CardContent>
      </Card>

      {/* History */}
      <Card>
        <CardHeader>
          <CardTitle>24-Hour History</CardTitle>
          <CardDescription>Recent readings from your IoT sensor</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Time</th>
                  <th className="text-left p-2">Temperature</th>
                  <th className="text-left p-2">Humidity</th>
                  <th className="text-left p-2">Moisture</th>
                </tr>
              </thead>
              <tbody>
                {readings.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="p-4 text-center text-muted-foreground">
                      No sensor data available. Waiting for ThingSpeak updates...
                    </td>
                  </tr>
                ) : (
                  readings.slice(0, 12).map((reading, index) => (
                    <tr key={index} className="border-b">
                      <td className="p-2">
                        {new Date(reading.timestamp).toLocaleTimeString('en-GB', {
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
                          hour12: false
                        })}
                      </td>
                      <td className="p-2">
                        {convertTemperature(reading.temperature)}{getTemperatureUnit()}
                      </td>
                      <td className="p-2">{reading.humidity}%</td>
                      <td className="p-2">{reading.soil_moisture}%</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
