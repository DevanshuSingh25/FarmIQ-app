import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";
import { 
  Sun, 
  Droplets, 
  Thermometer, 
  Wind,
  Calendar,
  CloudRain,
  Cloud,
  Leaf
} from "lucide-react";

export function WeatherCard() {
  const navigate = useNavigate();
  
  const currentDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  return (
    <Card 
      className="relative p-4 bg-gradient-to-br from-sky-100 via-blue-50 to-amber-50 dark:from-sky-900/20 dark:via-blue-900/10 dark:to-amber-900/20 border-sky-200/50 dark:border-sky-800/50 shadow-medium hover:shadow-strong transition-all duration-300 cursor-pointer group overflow-hidden"
      onClick={() => navigate('/farmer/weather')}
    >
      {/* Background Weather Image */}
      <div className="absolute inset-0 opacity-15 dark:opacity-10">
        <div 
          className="w-full h-full bg-cover bg-center bg-no-repeat rounded-lg"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 200'%3E%3Cdefs%3E%3ClinearGradient id='sky' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23e0f2fe;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23fef3c7;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='200' fill='url(%23sky)'/%3E%3Cpath d='M0,120 Q100,80 200,100 T400,90 L400,200 L0,200 Z' fill='%23dcfce7' opacity='0.6'/%3E%3Cpath d='M0,140 Q150,100 300,120 T400,110 L400,200 L0,200 Z' fill='%23bbf7d0' opacity='0.4'/%3E%3Ccircle cx='80' cy='60' r='15' fill='%23fbbf24' opacity='0.3'/%3E%3Ccircle cx='320' cy='40' r='12' fill='%23e5e7eb' opacity='0.4'/%3E%3Ccircle cx='350' cy='50' r='8' fill='%23e5e7eb' opacity='0.3'/%3E%3C/svg%3E")`
          }}
        />
      </div>
      
      {/* Main Content */}
      <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Left section - Date and Weather */}
        <div className="flex items-center gap-4">
          <div className="flex flex-col">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <Calendar className="h-4 w-4" />
              {currentDate}
            </div>
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="flex items-center gap-1">
                  <Sun className="h-6 w-6 text-amber-500" />
                  <Cloud className="h-5 w-5 text-slate-400 -ml-1" />
                </div>
              </div>
              <div>
                <span className="text-2xl font-bold text-foreground">18°C</span>
                <div className="text-left">
                  <p className="font-medium text-foreground text-sm">Cloudy</p>
                  <p className="text-xs text-muted-foreground">Partly cloudy day</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right section - Indicators */}
        <div className="flex gap-4">
          <div className="flex flex-col items-center gap-1">
            <div className="p-2 bg-white/90 dark:bg-black/30 rounded-full shadow-sm backdrop-blur-sm">
              <Droplets className="h-4 w-4 text-blue-500" />
            </div>
            <div className="text-center">
              <p className="text-xs font-medium text-muted-foreground">Humidity</p>
              <Badge className="text-xs bg-blue-500 text-white px-2 py-0.5 rounded-full">Good</Badge>
            </div>
          </div>
          
          <div className="flex flex-col items-center gap-1">
            <div className="p-2 bg-white/90 dark:bg-black/30 rounded-full shadow-sm backdrop-blur-sm">
              <Leaf className="h-4 w-4 text-orange-500" />
            </div>
            <div className="text-center">
              <p className="text-xs font-medium text-muted-foreground">Soil</p>
              <Badge className="text-xs bg-orange-500 text-white px-2 py-0.5 rounded-full">Low</Badge>
            </div>
          </div>
          
          <div className="flex flex-col items-center gap-1">
            <div className="p-2 bg-white/90 dark:bg-black/30 rounded-full shadow-sm backdrop-blur-sm">
              <CloudRain className="h-4 w-4 text-green-500" />
            </div>
            <div className="text-center">
              <p className="text-xs font-medium text-muted-foreground">Rain</p>
              <Badge className="text-xs bg-green-500 text-white px-2 py-0.5 rounded-full">Good</Badge>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}