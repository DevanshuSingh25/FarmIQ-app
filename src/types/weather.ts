export interface DailyForecast {
  date: string; // ISO date
  summary: string;
  icon: WeatherIcon;
  tempMinC: number;
  tempMaxC: number;
  precipChance: number; // 0-100
  windKph: number;
  windDir: string; // N/NE/E/SE/S/SW/W/NW
  humidityPct: number;
  confidence: 'High' | 'Medium' | 'Low';
  alerts: AlertCode[];
}

export interface HourlyForecast {
  dateTime: string; // ISO string
  tempC: number;
  precipChance: number;
  windKph: number;
}

export interface WeeklyAdvice {
  sow: string[]; // crop names
  avoid: string[]; // crop names
  tips: string[]; // simple textual tips
}

export interface WeatherAlert {
  code: AlertCode;
  from: string;
  to: string;
  severity: 'info' | 'warning' | 'danger';
  message: string;
}

export type WeatherIcon = 
  | 'clear' 
  | 'rain' 
  | 'storm' 
  | 'cloud' 
  | 'wind' 
  | 'fog' 
  | 'snow' 
  | 'heat';

export type AlertCode = 
  | 'heavy_rain' 
  | 'heat_wave' 
  | 'frost' 
  | 'high_wind' 
  | 'drought'
  | 'storm';

export interface Location {
  lat: number;
  lon: number;
  name: string;
  state?: string;
  district?: string;
}

export interface WeatherFilters {
  location: Location | null;
  units: {
    temperature: 'C' | 'F';
    wind: 'kph' | 'mph';
  };
}

export interface NotificationSettings {
  dailySummary: boolean;
  severeWeather: boolean;
  sowingAlerts: boolean;
  enabled: boolean;
}