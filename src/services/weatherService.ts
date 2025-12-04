import { DailyForecast, HourlyForecast, WeeklyAdvice, WeatherAlert, Location } from '@/types/weather';

// Mock data - replace with real API calls
const MOCK_DAILY_FORECAST: DailyForecast[] = [
  {
    date: '2025-09-25',
    summary: 'Partly cloudy',
    icon: 'cloud',
    tempMinC: 22,
    tempMaxC: 32,
    precipChance: 15,
    windKph: 12,
    windDir: 'NE',
    humidityPct: 65,
    confidence: 'High',
    alerts: []
  },
  {
    date: '2025-09-26',
    summary: 'Light rain',
    icon: 'rain',
    tempMinC: 23,
    tempMaxC: 31,
    precipChance: 78,
    windKph: 22,
    windDir: 'SW',
    humidityPct: 80,
    confidence: 'High',
    alerts: ['heavy_rain']
  },
  {
    date: '2025-09-27',
    summary: 'Heavy rain',
    icon: 'storm',
    tempMinC: 21,
    tempMaxC: 28,
    precipChance: 95,
    windKph: 35,
    windDir: 'W',
    humidityPct: 90,
    confidence: 'Medium',
    alerts: ['heavy_rain', 'high_wind']
  },
  {
    date: '2025-09-28',
    summary: 'Thunderstorms',
    icon: 'storm',
    tempMinC: 20,
    tempMaxC: 29,
    precipChance: 85,
    windKph: 28,
    windDir: 'SW',
    humidityPct: 85,
    confidence: 'High',
    alerts: ['storm']
  },
  {
    date: '2025-09-29',
    summary: 'Sunny',
    icon: 'clear',
    tempMinC: 24,
    tempMaxC: 35,
    precipChance: 5,
    windKph: 8,
    windDir: 'N',
    humidityPct: 55,
    confidence: 'High',
    alerts: []
  },
  {
    date: '2025-09-30',
    summary: 'Hot and sunny',
    icon: 'heat',
    tempMinC: 26,
    tempMaxC: 38,
    precipChance: 0,
    windKph: 6,
    windDir: 'NE',
    humidityPct: 45,
    confidence: 'High',
    alerts: ['heat_wave']
  },
  {
    date: '2025-10-01',
    summary: 'Very hot',
    icon: 'heat',
    tempMinC: 28,
    tempMaxC: 41,
    precipChance: 0,
    windKph: 10,
    windDir: 'E',
    humidityPct: 40,
    confidence: 'Medium',
    alerts: ['heat_wave']
  }
];

const MOCK_HOURLY_FORECAST: HourlyForecast[] = Array.from({ length: 24 }, (_, i) => ({
  dateTime: new Date(2025, 8, 25, i).toISOString(),
  tempC: 22 + Math.sin(i * Math.PI / 12) * 8,
  precipChance: i >= 14 && i <= 18 ? 60 + Math.random() * 30 : Math.random() * 20,
  windKph: 8 + Math.random() * 12
}));

const MOCK_WEEKLY_ADVICE: WeeklyAdvice = {
  sow: ['Rice', 'Maize', 'Sugarcane'],
  avoid: ['Onion', 'Chickpea', 'Leafy greens'],
  tips: [
    'Plan irrigation on Wednesday evening before heavy rain',
    'Avoid pesticide spraying during high wind periods',
    'Harvest mature crops before Thursday storm',
    'Prepare drainage channels for excessive rainfall'
  ]
};

export const weatherService = {
  async getWeatherForecast(location: Location, units = 'metric'): Promise<{
    lastUpdated: string;
    daily: DailyForecast[];
    hourly: HourlyForecast[];
  }> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    return {
      lastUpdated: new Date().toISOString(),
      daily: MOCK_DAILY_FORECAST,
      hourly: MOCK_HOURLY_FORECAST
    };
  },

  async getWeatherAdvice(location: Location): Promise<WeeklyAdvice> {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Mock logic based on forecast
    const hasHeavyRain = MOCK_DAILY_FORECAST.some(day => 
      day.alerts.includes('heavy_rain') || day.precipChance > 80
    );
    const hasHeatWave = MOCK_DAILY_FORECAST.some(day => 
      day.alerts.includes('heat_wave') || day.tempMaxC >= 38
    );

    let advice = { ...MOCK_WEEKLY_ADVICE };

    if (hasHeavyRain) {
      advice.avoid = [...advice.avoid, 'Open field vegetables'];
      advice.tips = [...advice.tips, 'Ensure proper drainage in fields'];
    }

    if (hasHeatWave) {
      advice.avoid = [...advice.avoid, 'Heat-sensitive crops'];
      advice.tips = [...advice.tips, 'Increase irrigation frequency', 'Provide shade for sensitive plants'];
    }

    return advice;
  },

  async getWeatherAlerts(location: Location): Promise<WeatherAlert[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const alerts: WeatherAlert[] = [];

    // Check for heavy rain
    const heavyRainDays = MOCK_DAILY_FORECAST.filter(day => day.alerts.includes('heavy_rain'));
    if (heavyRainDays.length > 0) {
      alerts.push({
        code: 'heavy_rain',
        from: heavyRainDays[0].date,
        to: heavyRainDays[heavyRainDays.length - 1].date,
        severity: 'warning',
        message: 'Heavy rainfall expected in the next 48 hours. Prepare drainage and avoid field operations.'
      });
    }

    // Check for heat wave
    const heatWaveDays = MOCK_DAILY_FORECAST.filter(day => day.alerts.includes('heat_wave'));
    if (heatWaveDays.length > 0) {
      alerts.push({
        code: 'heat_wave',
        from: heatWaveDays[0].date,
        to: heatWaveDays[heatWaveDays.length - 1].date,
        severity: 'danger',
        message: 'Heat wave conditions expected. Increase irrigation and provide crop protection.'
      });
    }

    // Check for high wind
    const windyDays = MOCK_DAILY_FORECAST.filter(day => day.alerts.includes('high_wind'));
    if (windyDays.length > 0) {
      alerts.push({
        code: 'high_wind',
        from: windyDays[0].date,
        to: windyDays[windyDays.length - 1].date,
        severity: 'warning',
        message: 'Strong winds expected. Secure loose materials and avoid spraying operations.'
      });
    }

    return alerts;
  },

  // Cache management
  getCachedForecast: (): any => {
    const cached = localStorage.getItem('weather_forecast_cache');
    return cached ? JSON.parse(cached) : null;
  },

  setCachedForecast: (data: any): void => {
    localStorage.setItem('weather_forecast_cache', JSON.stringify({
      ...data,
      cachedAt: new Date().toISOString()
    }));
  },

  isCacheValid: (cachedData: any): boolean => {
    if (!cachedData || !cachedData.cachedAt) return false;
    
    const cacheAge = Date.now() - new Date(cachedData.cachedAt).getTime();
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    
    return cacheAge < maxAge;
  }
};

// Location service for states and districts
export const locationService = {
  getStates: (): string[] => [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
  ],

  getDistricts: (state: string): string[] => {
    // Mock districts for each state - simplified
    const districtMap: Record<string, string[]> = {
      'Punjab': ['Amritsar', 'Bathinda', 'Faridkot', 'Fatehgarh Sahib', 'Ferozepur', 'Gurdaspur', 'Hoshiarpur', 'Jalandhar', 'Kapurthala', 'Ludhiana', 'Mansa', 'Moga', 'Mohali', 'Muktsar', 'Pathankot', 'Patiala', 'Rupnagar', 'Sangrur', 'Shaheed Bhagat Singh Nagar', 'Tarn Taran'],
      'Haryana': ['Ambala', 'Bhiwani', 'Charkhi Dadri', 'Faridabad', 'Fatehabad', 'Gurgaon', 'Hisar', 'Jhajjar', 'Jind', 'Kaithal', 'Karnal', 'Kurukshetra', 'Mahendragarh', 'Nuh', 'Palwal', 'Panchkula', 'Panipat', 'Rewari', 'Rohtak', 'Sirsa', 'Sonipat', 'Yamunanagar'],
      'Uttar Pradesh': ['Agra', 'Aligarh', 'Allahabad', 'Ambedkar Nagar', 'Amethi', 'Amroha', 'Auraiya', 'Azamgarh', 'Baghpat', 'Bahraich', 'Ballia', 'Balrampur', 'Banda', 'Barabanki', 'Bareilly', 'Basti', 'Bhadohi', 'Bijnor', 'Budaun', 'Bulandshahr', 'Chandauli', 'Chitrakoot', 'Deoria', 'Etah', 'Etawah', 'Faizabad', 'Farrukhabad', 'Fatehpur', 'Firozabad', 'Gautam Buddha Nagar', 'Ghaziabad', 'Ghazipur', 'Gonda', 'Gorakhpur', 'Hamirpur', 'Hapur', 'Hardoi', 'Hathras', 'Jalaun', 'Jaunpur', 'Jhajjar', 'Kannauj', 'Kanpur Dehat', 'Kanpur Nagar', 'Kashmiri Kuram', 'Khajuraho', 'Kheri', 'Kushinagar', 'Lalitpur', 'Lucknow', 'Maharajganj', 'Mahoba', 'Mainpuri', 'Mathura', 'Mau', 'Meerut', 'Mirzapur', 'Moradabad', 'Muzaffarnagar', 'Pilibhit', 'Pratapgarh', 'Raebareli', 'Rampur', 'Saharanpur', 'Sambhal', 'Sant Kabir Nagar', 'Shahjahanpur', 'Shamali', 'Shravasti', 'Siddharthnagar', 'Sitapur', 'Sonbhadra', 'Sultanpur', 'Unnao', 'Varanasi']
    };

    return districtMap[state] || ['Select a state first'];
  },

  getCurrentLocation: (): Promise<Location> => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation not supported'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            name: 'Current Location'
          });
        },
        (error) => {
          reject(error);
        },
        { timeout: 10000, enableHighAccuracy: true }
      );
    });
  }
};