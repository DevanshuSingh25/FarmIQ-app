import React, { useState } from "react";
import { setLanguage as setGoogleLanguage } from "@/lib/googleTranslate";
import { getTranslation, getCurrentLanguage } from "@/lib/translations";
import { Button } from "@/components/ui/button";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import {
  User,
  Sun,
  Moon,
  Globe,
  ChevronDown,
  UserCircle,
  History,
  Info,
  LogOut,
  QrCode,
  Menu,
  ChevronRight,
  FlaskConical,
  Bug,
  CloudRain,
  TrendingUp,
  Cpu,
  Home,
  ArrowLeft,
  Brain,
  MessageSquare
} from "lucide-react";

interface FarmIQNavbarProps {
  theme: 'light' | 'dark';
  language: 'English' | 'Hindi' | 'Punjabi';
  onThemeToggle: () => void;
  onLanguageChange: (lang: 'English' | 'Hindi' | 'Punjabi') => void;
}

export function FarmIQNavbar({ theme, language, onThemeToggle, onLanguageChange }: FarmIQNavbarProps) {
  const [activeLink, setActiveLink] = useState<string>('');
  const navigate = useNavigate();
  const location = useLocation();
  const { logout, user } = useAuth();
  const [currentLanguage, setCurrentLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>(language);

  const navLinks = [
    { label: 'Soil', href: '/soil-analysis' },
    { label: 'Crop', href: '/farmer/crop-disease' },
    { label: 'Market', href: '/market-prices' },
    { label: 'IoT', href: '/farmer/iot' },
    { label: 'Forum', href: '/farmer/forum' },
    { label: 'Consultancy', href: '/farmer/consultancy' },
    { label: 'EdgeAI', href: 'https://translationchatbotfinal.onrender.com/', external: true },
  ];

  const menuItems = [
    { label: "Dashboard", icon: Home, to: "/farmer/dashboard" },
    { label: "Soil analysis", icon: FlaskConical, to: "/soil-analysis" },
    { label: "Crop disease", icon: Bug, to: "/farmer/crop-disease" },
    { label: "Weather", icon: CloudRain, to: "/farmer/weather" },
    { label: "Market data", icon: TrendingUp, to: "/market-prices" },
    { label: "IoT Sensor", icon: Cpu, to: "/farmer/iot" },
    { label: "Forum", icon: MessageSquare, to: "/farmer/forum" },
    { label: "Consultancy", icon: UserCircle, to: "/farmer/consultancy" },
    { label: "EdgeAI", icon: Brain, to: "https://translationchatbotfinal.onrender.com/", external: true }
  ];

  const languages = ['English', 'Hindi', 'Punjabi'] as const;

  // Update current language when language prop changes
  React.useEffect(() => {
    setCurrentLanguage(language);
  }, [language]);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border shadow-soft">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left side - Burger Menu and Logo */}
          <div className="flex items-center space-x-6">
            {/* Burger Menu - Leftmost position */}
            <div>
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="outline" size="icon" className="rounded-full shadow-medium">
                    <Menu className="h-5 w-5" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-80">
                  <SheetHeader>
                    <SheetTitle>Navigation</SheetTitle>
                  </SheetHeader>
                  <div className="mt-6 space-y-2">
                    {menuItems.map((item) => {
                      const Icon = item.icon;
                      const active = location.pathname === item.to;
                      return (
                        <Button
                          key={item.to}
                          variant={active ? "default" : "ghost"}
                          className="w-full justify-between"
                          onClick={() => {
                            if (item.external) {
                              window.open(item.to, '_blank', 'noopener,noreferrer');
                            } else {
                              navigate(item.to);
                            }
                          }}
                        >
                          <span className="flex items-center gap-3">
                            <Icon className="h-4 w-4" />
                            {item.label}
                          </span>
                          <ChevronRight className="h-4 w-4 opacity-60" />
                        </Button>
                      );
                    })}
                  </div>
                </SheetContent>
              </Sheet>
            </div>

            {/* Logo with spacing */}
            <div className="flex-shrink-0">
              <button
                onClick={() => navigate(`/${user?.role || 'farmer'}/dashboard`)}
                className="text-2xl font-bold text-primary hover:text-primary/80 transition-colors cursor-pointer"
              >
                FarmIQ <span className="text-lg font-normal text-muted-foreground ml-3">
                  {user?.role === 'farmer' ? 'Farmer' :
                    user?.role === 'vendor' ? 'Vendor' :
                      user?.role === 'admin' ? 'Admin' : 'User'} dashboard
                </span>
              </button>
            </div>
          </div>

          {/* Navigation Links - Desktop Only */}
          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map((link) => (
              <button
                key={link.label}
                onClick={() => {
                  setActiveLink(link.href);
                  if (link.external) {
                    window.open(link.href, '_blank', 'noopener,noreferrer');
                  } else if (link.href.startsWith('/')) {
                    navigate(link.href);
                  }
                }}
                className={`relative text-foreground hover:text-primary transition-smooth font-medium ${activeLink === link.href ? 'text-primary' : ''
                  } after:content-[''] after:absolute after:w-full after:scale-x-0 after:h-0.5 after:bottom-0 after:left-0 after:bg-primary after:origin-bottom-right after:transition-transform after:duration-300 hover:after:scale-x-100 hover:after:origin-bottom-left`}
              >
                {link.label}
              </button>
            ))}
          </div>

          {/* Right side controls */}
          <div className="flex items-center space-x-3">
            {/* Language Switcher */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="h-9 w-9 p-0">
                  <Globe className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-popover border border-border shadow-medium">
                {languages.map((lang) => (
                  <DropdownMenuItem
                    key={lang}
                    onClick={() => { onLanguageChange(lang); setGoogleLanguage(lang); }}
                    className="cursor-pointer hover:bg-muted"
                  >
                    {lang}
                    {language === lang && <span className="ml-2 text-primary">✓</span>}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Theme Switcher */}
            <Button
              variant="ghost"
              size="sm"
              className="h-9 w-9 p-0"
              onClick={onThemeToggle}
            >
              {theme === 'light' ?
                <Moon className="h-4 w-4" /> :
                <Sun className="h-4 w-4" />
              }
            </Button>

            {/* QR Menu */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="h-9 w-9 p-0" aria-label="QR menu">
                  <QrCode className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-popover border border-border shadow-medium">
                <DropdownMenuItem
                  className="cursor-pointer hover:bg-muted"
                  onClick={() => navigate('/farmer/qr/generate')}
                  role="menuitem"
                >
                  {getTranslation('qr.menu.generate', currentLanguage)}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Profile Menu */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="h-9 w-9 p-0">
                  <User className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48 bg-popover border border-border shadow-medium">
                <DropdownMenuItem
                  className="cursor-pointer hover:bg-muted py-3"
                  onClick={() => navigate('/profile')}
                >
                  <UserCircle className="mr-3 h-4 w-4" />
                  Profile
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer hover:bg-muted py-3"
                  onClick={() => navigate('/yield-prediction')}
                >
                  <History className="mr-3 h-4 w-4" />
                  Crop history
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer hover:bg-muted py-3"
                  onClick={() => navigate('/farmer/teaching')}
                >
                  <Info className="mr-3 h-4 w-4" />
                  Know about the website
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer hover:bg-muted py-3 text-destructive"
                  onClick={async () => {
                    await logout();
                    navigate('/login');
                  }}
                >
                  <LogOut className="mr-3 h-4 w-4" />
                  Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </nav>
  );
}