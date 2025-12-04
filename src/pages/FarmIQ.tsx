import { useState } from "react";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { WeatherCard } from "@/components/farmiq/WeatherCard";
import { ActionButtons } from "@/components/farmiq/ActionButtons";
// Removed FarmIQFooter to replace with dashboard widgets and recent activities
import { SectionSpeaker } from "@/components/ui/section-speaker";

export default function FarmIQ() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  return (
    <div className="min-h-screen bg-background">
      <FarmIQNavbar 
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />
      
      <main className="pt-20 pb-4 px-4">
        <div className="container mx-auto max-w-6xl">
          {/* Weather Card */}
          <div className="mb-12 relative">
            <div className="absolute top-2 right-2 z-10">
              <SectionSpeaker 
                getText={() => "Today's weather is 18 degrees celsius with cloudy conditions. Humidity is good at 65%, soil moisture is low at 35%, and there's light precipitation expected."}
                sectionId="weather-card"
                ariaLabel="Read weather information"
                alwaysVisible
              />
            </div>
            <WeatherCard />
          </div>
          
          {/* Action Buttons */}
          <div className="relative">
            <ActionButtons />
          </div>

          {/* Dashboard Widgets: Yield, Revenue, Active Crops */}
          <div className="mt-8 grid lg:grid-cols-3 gap-6">
            <div className="bg-background border border-border rounded-xl p-6 shadow-medium">
              <h3 className="text-sm font-medium text-muted-foreground">This Month's Yield</h3>
              <div className="mt-3 text-3xl font-bold text-foreground">2.5 tons</div>
              <p className="text-xs text-muted-foreground mt-1">+15% from last month</p>
            </div>
            <div className="bg-background border border-border rounded-xl p-6 shadow-medium">
              <h3 className="text-sm font-medium text-muted-foreground">Revenue</h3>
              <div className="mt-3 text-3xl font-bold text-foreground">₹1,25,000</div>
              <p className="text-xs text-muted-foreground mt-1">+8% from last month</p>
            </div>
            <div className="bg-background border border-border rounded-xl p-6 shadow-medium">
              <h3 className="text-sm font-medium text-muted-foreground">Active Crops</h3>
              <div className="mt-3 text-3xl font-bold text-foreground">5</div>
              <p className="text-xs text-muted-foreground mt-1">Rice, Wheat, Corn, Tomato, Onion</p>
            </div>
          </div>

          {/* Recent Activities */}
          <div className="mt-8 bg-background border border-border rounded-xl p-6 shadow-medium">
            <h3 className="text-lg font-semibold mb-4">Recent Activities</h3>
            <ul className="divide-y divide-border">
              <li className="py-3 flex items-center justify-between">
                <span className="text-sm">Soil test completed</span>
                <span className="text-xs text-muted-foreground">2 hours ago</span>
              </li>
              <li className="py-3 flex items-center justify-between">
                <span className="text-sm">Weather alert received</span>
                <span className="text-xs text-muted-foreground">4 hours ago</span>
              </li>
              <li className="py-3 flex items-center justify-between">
                <span className="text-sm">Disease scan performed</span>
                <span className="text-xs text-muted-foreground">1 day ago</span>
              </li>
              <li className="py-3 flex items-center justify-between">
                <span className="text-sm">Market price updated</span>
                <span className="text-xs text-muted-foreground">2 days ago</span>
              </li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}