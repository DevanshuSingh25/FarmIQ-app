import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { SectionSpeaker } from '@/components/ui/section-speaker';
import { FarmIQNavbar } from '@/components/farmiq/FarmIQNavbar';
import { CloudRain, Wifi, WifiOff } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { LocationSelector } from '@/components/weather/LocationSelector';
import { WeatherCard } from '@/components/weather/WeatherCard';
import { HourlyForecastPanel } from '@/components/weather/HourlyForecastPanel';
import { WeeklyAdvice } from '@/components/weather/WeeklyAdvice';
import { WeatherAlerts } from '@/components/weather/WeatherAlerts';
import { NotificationSettings } from '@/components/weather/NotificationSettings';
import {
  DailyForecast,
  HourlyForecast,
  WeeklyAdvice as WeeklyAdviceType,
  WeatherAlert,
  WeatherFilters,
  NotificationSettings as NotificationSettingsType,
  WeatherIcon
} from '@/types/weather';
import { useToast } from '@/hooks/use-toast';

const OPENWEATHER_API_KEY = '2191c6793168af5d4899a607a940e6d9';

const capitalize = (text: string) => text ? text.charAt(0).toUpperCase() + text.slice(1) : '';

const mapWeatherIcon = (main?: string): WeatherIcon => {
  const key = (main || '').toLowerCase();
  if (key === 'clear') return 'clear';
  if (key === 'thunderstorm' || key === 'tornado') return 'storm';
  if (key === 'drizzle' || key === 'rain') return 'rain';
  if (key === 'snow') return 'snow';
  if (['mist', 'smoke', 'haze', 'dust', 'fog', 'sand', 'ash'].includes(key)) return 'fog';
  if (key === 'squall') return 'wind';
  if (key === 'clouds') return 'cloud';
  return 'cloud';
};

const degToCompass = (deg?: number): string => {
  if (deg === undefined || deg === null) return 'N';
  const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  return directions[Math.round(deg / 45) % 8];
};

const buildAlerts = (forecasts: DailyForecast[]): WeatherAlert[] => {
  const alerts: WeatherAlert[] = [];

  const heavyRainDays = forecasts.filter(day => day.precipChance >= 80);
  if (heavyRainDays.length) {
    alerts.push({
      code: 'heavy_rain',
      from: heavyRainDays[0].date,
      to: heavyRainDays[heavyRainDays.length - 1].date,
      severity: 'warning',
      message: 'Heavy rainfall expected. Prepare drainage and delay spraying operations.'
    });
  }

  const heatDays = forecasts.filter(day => day.tempMaxC >= 38);
  if (heatDays.length) {
    alerts.push({
      code: 'heat_wave',
      from: heatDays[0].date,
      to: heatDays[heatDays.length - 1].date,
      severity: 'danger',
      message: 'High temperature spell ahead. Increase irrigation and provide shade.'
    });
  }

  const windyDays = forecasts.filter(day => day.windKph >= 40);
  if (windyDays.length) {
    alerts.push({
      code: 'high_wind',
      from: windyDays[0].date,
      to: windyDays[windyDays.length - 1].date,
      severity: 'warning',
      message: 'Gusty winds forecasted. Secure light materials and avoid foliar sprays.'
    });
  }

  return alerts;
};

type FetchWeatherOptions = {
  city?: string;
  coords?: { lat: number; lon: number };
  label?: string;
};

const Weather = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  const [filters, setFilters] = useState<WeatherFilters>({
    location: null,
    units: {
      temperature: 'C',
      wind: 'kph'
    }
  });

  const [dailyForecast, setDailyForecast] = useState<DailyForecast[]>([]);
  const [hourlyForecast, setHourlyForecast] = useState<HourlyForecast[]>([]);
  const [weeklyAdvice, setWeeklyAdvice] = useState<WeeklyAdviceType | null>(null);
  const [weatherAlerts, setWeatherAlerts] = useState<WeatherAlert[]>([]);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [selectedDay, setSelectedDay] = useState<string>('');
  const [showHourlyPanel, setShowHourlyPanel] = useState(false);
  const [selectedCity, setSelectedCity] = useState('');
  const [notificationSettings, setNotificationSettings] = useState<NotificationSettingsType>({
    dailySummary: false,
    severeWeather: true,
    sowingAlerts: true,
    enabled: false
  });

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  // Monitor online status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const fetchWeatherData = async (options: FetchWeatherOptions = {}) => {
    const baseCity = options.city || selectedCity || filters.location?.district || filters.location?.name?.split(',')[0]?.trim();
    if (!baseCity && !options.coords) {
      toast({
        title: "Select a city",
        description: "Please choose a city before fetching weather data.",
        variant: "destructive"
      });
      return;
    }

    if (!isOnline) {
      toast({
        title: "Offline",
        description: "Connect to the internet to fetch the latest weather.",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      let endpoint = '';
      if (options.coords) {
        endpoint = `https://api.openweathermap.org/data/2.5/forecast?lat=${options.coords.lat}&lon=${options.coords.lon}&appid=${OPENWEATHER_API_KEY}&units=metric`;
      } else {
        endpoint = `https://api.openweathermap.org/data/2.5/forecast?q=${encodeURIComponent(baseCity)},IN&appid=${OPENWEATHER_API_KEY}&units=metric`;
      }
      const response = await fetch(endpoint);
      const data = await response.json();

      if (data.cod !== '200') {
        throw new Error('Unable to fetch weather data for this city.');
      }

      const dailyMap = new Map<string, any>();
      data.list.forEach((item: any) => {
        const dateKey = item.dt_txt.split(' ')[0];
        if (!dailyMap.has(dateKey)) {
          dailyMap.set(dateKey, item);
        }
      });

      const dailyData: DailyForecast[] = Array.from(dailyMap.entries())
        .slice(0, 5)
        .map(([date, item]) => ({
          date,
          summary: capitalize(item.weather?.[0]?.description || 'Forecast'),
          icon: mapWeatherIcon(item.weather?.[0]?.main),
          tempMinC: Math.round(item.main?.temp_min ?? 0),
          tempMaxC: Math.round(item.main?.temp_max ?? 0),
          precipChance: Math.round((item.pop ?? 0) * 100),
          windKph: Math.round((item.wind?.speed ?? 0) * 3.6),
          windDir: degToCompass(item.wind?.deg),
          humidityPct: Math.round(item.main?.humidity ?? 0),
          confidence: 'High',
          alerts: []
        }));

      const hourlyData: HourlyForecast[] = data.list.slice(0, 24).map((item: any) => ({
        dateTime: item.dt_txt,
        tempC: Math.round(item.main?.temp ?? 0),
        precipChance: Math.round((item.pop ?? 0) * 100),
        windKph: Math.round((item.wind?.speed ?? 0) * 3.6)
      }));

      setDailyForecast(dailyData);
      setHourlyForecast(hourlyData);
      setWeatherAlerts(buildAlerts(dailyData));
      setWeeklyAdvice(null);
      setLastUpdated(new Date().toISOString());
      const resolvedLabel = options.label || options.city || baseCity;
      if (resolvedLabel) {
        setSelectedCity(resolvedLabel);
      }
    } catch (error) {
      toast({
        title: "Failed to fetch weather data",
        description: error instanceof Error ? error.message : "Please try again later.",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleViewHourly = (date: string) => {
    setSelectedDay(date);
    setShowHourlyPanel(true);
  };

  const handleDismissAlert = (alertCode: string) => {
    setWeatherAlerts(prev => prev.filter(alert => alert.code !== alertCode));
  };

  const handleSaveReminder = (alert: WeatherAlert) => {
    // Mock notification scheduling
    toast({
      title: "Reminder saved",
      description: `You'll be notified about ${alert.code.replace('_', ' ')}`,
    });
  };

  const handleCitySelect = (city: string) => {
    const location = {
      lat: 0,
      lon: 0,
      name: `${city}, Punjab`,
      state: 'Punjab',
      district: city
    };

    setFilters(prev => ({
      ...prev,
      location
    }));

    fetchWeatherData({ city, label: city });
  };

  const handleUseCurrentLocation = async ({ lat, lon }: { lat: number; lon: number }) => {
    try {
      let label = 'Current Location';
      const reverseUrl = `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${OPENWEATHER_API_KEY}`;
      const reverseResponse = await fetch(reverseUrl);
      if (reverseResponse.ok) {
        const reverseData = await reverseResponse.json();
        if (reverseData?.[0]?.name) {
          label = reverseData[0].name;
        }
      }

      setFilters(prev => ({
        ...prev,
        location: {
          lat,
          lon,
          name: `${label}, Punjab`,
          state: 'Punjab',
          district: label
        }
      }));

      await fetchWeatherData({ coords: { lat, lon }, label });
    } catch (error) {
      toast({
        title: "Unable to use current location",
        description: error instanceof Error ? error.message : "Please try again later.",
        variant: "destructive"
      });
      throw error;
    }
  };

  const isEmpty = !filters.location || dailyForecast.length === 0;
  const isOfflineWithNoData = !isOnline && dailyForecast.length === 0;

  return (
    <div className="min-h-screen bg-background">
      <FarmIQNavbar
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />

      {/* Header */}
      <header className="bg-card border-b shadow-soft group relative">
        <div className="absolute top-4 right-4 z-10">
          <SectionSpeaker
            getText={() => "Weather and Crop Advice page. Get personalized weather insights and agricultural recommendations based on your location. View 5-day forecasts, hourly predictions, and farming advice."}
            sectionId="weather-page-header"
            ariaLabel="Read weather page information"
            alwaysVisible
          />
        </div>
        <div className="container mx-auto px-4 py-4 pt-20">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
                <CloudRain className="h-6 w-6 text-primary" />
                Weather & Crop Advice
              </h1>
              <div className="flex items-center gap-4 mt-1">
                <span className="text-sm text-muted-foreground">
                  Home / Weather
                </span>
                <Badge
                  variant={isOnline ? "secondary" : "destructive"}
                  className="text-xs"
                >
                  {isOnline ? (
                    <>
                      <Wifi className="h-3 w-3 mr-1" />
                      Online
                    </>
                  ) : (
                    <>
                      <WifiOff className="h-3 w-3 mr-1" />
                      Offline
                    </>
                  )}
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6 space-y-6">
        {/* Location Selector */}
        <LocationSelector
          filters={filters}
          onFiltersChange={setFilters}
          lastUpdated={lastUpdated}
          onRefresh={() => fetchWeatherData()}
          isLoading={loading}
          onCitySelect={handleCitySelect}
          onUseCurrentLocation={handleUseCurrentLocation}
        />

        {/* Offline Banner */}
        {!isOnline && (
          <Alert className="border-warning bg-warning/5">
            <WifiOff className="h-4 w-4" />
            <AlertDescription>
              You're offline. Showing last saved forecast
              {lastUpdated && ` (updated ${new Date(lastUpdated).toLocaleString()})`}.
            </AlertDescription>
          </Alert>
        )}

        {/* Weather Alerts */}
        {weatherAlerts.length > 0 && (
          <WeatherAlerts
            alerts={weatherAlerts}
            onDismiss={handleDismissAlert}
            onSaveReminder={handleSaveReminder}
          />
        )}

        {/* Empty State */}
        {isEmpty && (
          <div className="text-center py-12">
            <CloudRain className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-foreground mb-2">
              Choose a location to view forecast
            </h3>
            <p className="text-muted-foreground">
              Select your location above to get personalized weather insights and crop advice
            </p>
          </div>
        )}

        {/* Offline with no data */}
        {isOfflineWithNoData && (
          <div className="text-center py-12">
            <WifiOff className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-foreground mb-2">
              No cached data available
            </h3>
            <p className="text-muted-foreground">
              Connect to the internet to fetch weather data
            </p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {Array.from({ length: 7 }).map((_, i) => (
                <div key={i} className="relative overflow-hidden rounded-2xl shadow-2xl">
                  <div className="absolute inset-0 bg-gradient-to-b from-blue-600 via-purple-600 to-orange-500" />
                  <div className="relative p-6 space-y-4">
                    <div className="flex justify-between items-start">
                      <Skeleton className="h-6 w-20 bg-white/20" />
                      <Skeleton className="h-5 w-16 bg-white/20" />
                    </div>
                    <div className="flex justify-center">
                      <Skeleton className="h-20 w-20 rounded-full bg-white/20" />
                    </div>
                    <div className="text-center">
                      <Skeleton className="h-16 w-24 mx-auto bg-white/20" />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <Skeleton className="h-16 w-full bg-white/20 rounded-lg" />
                      <Skeleton className="h-16 w-full bg-white/20 rounded-lg" />
                    </div>
                    <Skeleton className="h-10 w-full bg-white/20 rounded-lg" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Weather Content */}
        {!isEmpty && !loading && dailyForecast.length > 0 && (
          <>
            {/* 5-Day Forecast Grid */}
            <div className="space-y-4 group relative">
              <div className="absolute top-2 right-2 z-10">
                <SectionSpeaker
                  getText={() => "5-Day weather forecast showing daily conditions, temperatures, precipitation, and farming advice for the selected location."}
                  sectionId="weather-forecast-grid"
                  ariaLabel="Read weather forecast information"
                  alwaysVisible
                />
              </div>
              <h2 className="text-xl font-semibold text-foreground">5-Day Forecast</h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {dailyForecast.map((forecast) => (
                  <WeatherCard
                    key={forecast.date}
                    forecast={forecast}
                    units={filters.units}
                    onViewHourly={() => handleViewHourly(forecast.date)}
                  />
                ))}
              </div>
            </div>

            {/* Weekly Advice */}
            {weeklyAdvice && (
              <WeeklyAdvice advice={weeklyAdvice} />
            )}

            {/* Notification Settings */}
            <NotificationSettings
              settings={notificationSettings}
              onSettingsChange={setNotificationSettings}
            />
          </>
        )}

        {/* Hourly Forecast Panel */}
        <HourlyForecastPanel
          isOpen={showHourlyPanel}
          onClose={() => setShowHourlyPanel(false)}
          hourlyData={hourlyForecast}
          date={selectedDay}
          units={filters.units}
        />
      </div>
    </div>
  );
};

export default Weather;