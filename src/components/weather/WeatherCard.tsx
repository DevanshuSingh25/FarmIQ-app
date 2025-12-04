import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Cloud, 
  CloudRain, 
  Sun, 
  Zap, 
  Wind, 
  CloudSnow, 
  Thermometer,
  Eye 
} from 'lucide-react';
import { DailyForecast, WeatherIcon } from '@/types/weather';

interface WeatherCardProps {
  forecast: DailyForecast;
  units: { temperature: 'C' | 'F'; wind: 'kph' | 'mph' };
  onViewHourly: () => void;
}

const weatherIcons: Record<WeatherIcon, React.ComponentType<{ className?: string }>> = {
  clear: Sun,
  rain: CloudRain,
  storm: Zap,
  cloud: Cloud,
  wind: Wind,
  fog: Cloud,
  snow: CloudSnow,
  heat: Thermometer
};

const confidenceColors = {
  High: 'bg-green-500/20 text-green-100 border-green-500/30',
  Medium: 'bg-yellow-500/20 text-yellow-100 border-yellow-500/30',
  Low: 'bg-red-500/20 text-red-100 border-red-500/30'
};

export function WeatherCard({ forecast, units, onViewHourly }: WeatherCardProps) {
  const WeatherIcon = weatherIcons[forecast.icon];
  const date = new Date(forecast.date);
  const dayName = date.toLocaleDateString('en-US', { weekday: 'long' }).toUpperCase();
  const dayDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toUpperCase();
  
  const convertTemp = (tempC: number) => {
    return units.temperature === 'F' ? Math.round((tempC * 9/5) + 32) : tempC;
  };
  
  const convertWind = (windKph: number) => {
    return units.wind === 'mph' ? Math.round(windKph * 0.621371) : windKph;
  };

  // Get gradient based on weather condition
  const getGradientClass = (icon: WeatherIcon) => {
    switch (icon) {
      case 'clear':
        return 'from-blue-600 via-purple-600 to-orange-500';
      case 'rain':
        return 'from-slate-600 via-blue-700 to-slate-800';
      case 'storm':
        return 'from-gray-800 via-purple-900 to-gray-900';
      case 'cloud':
        return 'from-gray-500 via-blue-600 to-gray-700';
      case 'wind':
        return 'from-blue-500 via-cyan-600 to-blue-700';
      case 'fog':
        return 'from-gray-400 via-gray-600 to-gray-800';
      case 'snow':
        return 'from-blue-200 via-blue-400 to-blue-600';
      case 'heat':
        return 'from-orange-500 via-red-600 to-yellow-600';
      default:
        return 'from-blue-600 via-purple-600 to-orange-500';
    }
  };

  return (
    <div className="relative overflow-hidden rounded-2xl shadow-2xl transition-all duration-300 hover:scale-105 hover:shadow-3xl">
      {/* Gradient Background */}
      <div className={`absolute inset-0 bg-gradient-to-b ${getGradientClass(forecast.icon)}`} />
      
      {/* Content */}
      <div className="relative p-6 text-white">
        {/* Header */}
        <div className="flex justify-between items-start mb-6">
          <div>
            <h3 className="text-2xl font-bold text-white mb-1">{dayName}</h3>
            <p className="text-sm text-white/80 font-medium">{dayDate}</p>
          </div>
          <Badge 
            className={`text-xs px-2 py-1 ${confidenceColors[forecast.confidence]} border backdrop-blur-sm`}
            variant="secondary"
          >
            {forecast.confidence}
          </Badge>
        </div>

        {/* Weather Icon - Large and Prominent */}
        <div className="flex justify-center mb-6">
          <div className="relative">
            <WeatherIcon className="h-20 w-20 text-white/90 drop-shadow-lg" />
            {/* Add some cloud elements for visual interest */}
            {forecast.icon === 'clear' && (
              <>
                <Cloud className="absolute -top-2 -right-2 h-8 w-8 text-white/60" />
                <Cloud className="absolute -bottom-1 -left-3 h-6 w-6 text-white/50" />
              </>
            )}
          </div>
        </div>

        {/* Temperature - Large and Prominent */}
        <div className="text-center mb-6">
          <p className="text-sm text-white/80 font-medium mb-1">TEMPERATURE</p>
          <div className="flex justify-center items-baseline gap-2">
            <span className="text-6xl font-bold text-white drop-shadow-lg">
              {convertTemp(forecast.tempMaxC)}°
            </span>
            <span className="text-2xl text-white/70 font-medium">
              {convertTemp(forecast.tempMinC)}°
            </span>
          </div>
        </div>

        {/* Weather Summary */}
        <div className="text-center mb-6">
          <p className="text-lg font-medium text-white/90 capitalize">{forecast.summary}</p>
        </div>

        {/* Weather Details - Compact Grid */}
        <div className="grid grid-cols-2 gap-3 text-sm mb-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3 text-center">
            <CloudRain className="h-5 w-5 text-white/80 mx-auto mb-1" />
            <p className="text-white/80 text-xs">Rain</p>
            <p className="text-white font-semibold">{forecast.precipChance}%</p>
          </div>
          
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3 text-center">
            <Wind className="h-5 w-5 text-white/80 mx-auto mb-1" />
            <p className="text-white/80 text-xs">Wind</p>
            <p className="text-white font-semibold">
              {convertWind(forecast.windKph)} {units.wind}
            </p>
          </div>
        </div>

        {/* Alerts */}
        {forecast.alerts.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {forecast.alerts.map((alert, index) => (
              <Badge 
                key={index} 
                className="text-xs bg-red-500/20 text-red-100 border-red-500/30 backdrop-blur-sm"
              >
                {alert.replace('_', ' ')}
              </Badge>
            ))}
          </div>
        )}

        {/* View Hourly Button */}
        <Button
          onClick={onViewHourly}
          className="w-full bg-white/20 hover:bg-white/30 text-white border-white/30 backdrop-blur-sm transition-all duration-200 font-medium"
          variant="outline"
        >
          <Eye className="h-4 w-4 mr-2" />
          View Hourly
        </Button>
      </div>
    </div>
  );
}