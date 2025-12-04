import { Button } from "@/components/ui/button";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { Play } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

const Teaching = () => {
  const navigate = useNavigate();
  
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
      
      <div className="border-b bg-card">
        <div className="container mx-auto px-4 py-6 pt-24">
          <div className="flex items-center gap-4 mb-4">
          </div>
          <h1 className="text-3xl font-bold">Know about the website</h1>
          <p className="text-muted-foreground">Add your single YouTube video below.</p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="bg-card border rounded-xl p-4 shadow-medium">
          <AspectRatio ratio={16 / 9}>
            <div className="w-full h-full flex items-center justify-center rounded-md border-2 border-dashed border-border bg-muted">
              <div className="text-center">
                <Play className="h-10 w-10 mx-auto mb-2 text-muted-foreground" />
                <p className="text-muted-foreground">YouTube video placeholder</p>
              </div>
            </div>
          </AspectRatio>
        </div>
      </div>
    </div>
  );
};

export default Teaching;