import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useNavigate, Link } from 'react-router-dom';
import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import {
  Menu,
  Globe,
  Moon,
  Sun,
  User,
  DollarSign,
  Box,
  Sprout,
  TrendingUp,
  Search,
  BarChart3,
  Leaf,
  LogOut,
  Info,
  UserCircle,
  LayoutDashboard,
  QrCode,
  ChevronRight,
  MessageCircle,
  Phone,
  Brain
} from 'lucide-react';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { setLanguage as setGoogleLanguage } from "@/lib/googleTranslate";

export default function VendorDashboard() {
  const navigate = useNavigate();
  const { logout } = useAuth();

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');
  const languages = ['English', 'Hindi', 'Punjabi'] as const;

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  return (
    <div className="min-h-screen bg-[#F8F9FA] dark:bg-gray-900 font-sans text-gray-900 dark:text-gray-100">
      {/* Top Navigation Bar */}
      <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            {/* Left Side */}
            <div className="flex items-center gap-4">
              <Sheet>
                <SheetTrigger asChild>
                  <button className="p-2.5 bg-white rounded-full shadow-sm hover:shadow-md transition-all border border-gray-100 group">
                    <Menu className="h-5 w-5 text-gray-700 group-hover:text-gray-900" />
                  </button>
                </SheetTrigger>
                <SheetContent side="left" className="w-[300px] sm:w-[340px] p-6 bg-[#F8F9FA] dark:bg-gray-800">
                  <SheetHeader className="mb-8 flex flex-row items-center justify-between space-y-0">
                    <SheetTitle className="text-xl font-bold text-gray-900 dark:text-gray-100">Navigation</SheetTitle>
                  </SheetHeader>
                  <div className="space-y-3">
                    <Link to="/vendor/dashboard" className="flex items-center justify-between px-4 py-3 bg-[#1a5d1a] dark:bg-green-700 text-white rounded-full shadow-sm transition-all group">
                      <div className="flex items-center gap-3">
                        <LayoutDashboard className="h-5 w-5 stroke-[2.5]" />
                        <span className="font-medium text-sm">Dashboard</span>
                      </div>
                      <ChevronRight className="h-4 w-4 text-white/80" />
                    </Link>

                    <Link to="/vendor/farmer-search" className="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-100 hover:bg-green-50 dark:hover:bg-green-900 hover:text-green-900 dark:hover:text-green-100 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100 dark:hover:border-green-700">
                      <div className="flex items-center gap-3">
                        <Search className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                        <span className="font-medium text-sm">Farmer Search</span>
                      </div>
                      <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                    </Link>

                    <Link to="/vendor/market-prices" className="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-100 hover:bg-yellow-50 dark:hover:bg-yellow-900 hover:text-yellow-900 dark:hover:text-yellow-100 rounded-full transition-all group shadow-sm border border-transparent hover:border-yellow-100 dark:hover:border-yellow-700">
                      <div className="flex items-center gap-3">
                        <BarChart3 className="h-5 w-5 text-gray-500 group-hover:text-yellow-700 transition-colors" />
                        <span className="font-medium text-sm">Market Price</span>
                      </div>
                      <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-yellow-700 transition-colors" />
                    </Link>

                    <a href="https://translationchatbotfinal.onrender.com/" target="_blank" rel="noopener noreferrer" className="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-100 hover:bg-purple-50 dark:hover:bg-purple-900 hover:text-purple-900 dark:hover:text-purple-100 rounded-full transition-all group shadow-sm border border-transparent hover:border-purple-100 dark:hover:border-purple-700">
                      <div className="flex items-center gap-3">
                        <Brain className="h-5 w-5 text-gray-500 group-hover:text-purple-700 transition-colors" />
                        <span className="font-medium text-sm">EdgeAI</span>
                      </div>
                      <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-purple-700 transition-colors" />
                    </a>
                  </div>
                </SheetContent>
              </Sheet>
              <div className="flex items-center gap-2">
                <Leaf className="h-6 w-6 text-green-600" fill="currentColor" />
                <span className="text-xl font-bold text-green-600 tracking-tight">FarmIQ</span>
                <span className="text-gray-400 text-lg font-light">|</span>
                <span className="text-gray-500 text-sm font-medium">Vendor dashboard</span>
              </div>
            </div>

            {/* Center Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              <Link to="/vendor/dashboard" className="text-gray-900 font-medium border-b-2 border-green-500 px-1 py-5">
                Dashboard
              </Link>
              <Link to="/vendor/qr-scan" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
                QR Scan
              </Link>
              <Link to="/vendor/market-prices" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
                Market Prices
              </Link>
              <Link to="/vendor/farmer-search" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
                Farmer Search
              </Link>
              <a href="https://translationchatbotfinal.onrender.com/" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
                EdgeAI
              </a>
            </div>

            {/* Right Side Icons */}
            <div className="flex items-center gap-2">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors">
                    <Globe className="h-5 w-5" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="bg-white border border-gray-200 shadow-md">
                  {languages.map((lang) => (
                    <DropdownMenuItem
                      key={lang}
                      onClick={() => {
                        setLanguage(lang);
                        setGoogleLanguage(lang);
                      }}
                      className="cursor-pointer hover:bg-gray-50"
                    >
                      {lang}
                      {language === lang && <span className="ml-2 text-green-600">✓</span>}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>

              <button
                className="p-2 text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
                onClick={toggleTheme}
              >
                {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
              </button>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors">
                    <User className="h-5 w-5" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56 bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                  <DropdownMenuItem
                    className="cursor-pointer"
                    onClick={() => navigate('/profile')}
                  >
                    <UserCircle className="mr-2 h-4 w-4" />
                    <span>Profile</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="cursor-pointer"
                    onClick={() => navigate('/farmer/teaching')}
                  >
                    <Info className="mr-2 h-4 w-4" />
                    <span>Know about the website</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="cursor-pointer text-red-600 focus:text-red-600"
                    onClick={() => {
                      logout();
                      navigate('/login');
                    }}
                  >
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Logout</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Dashboard Layout */}
      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Dashboard Overview</h1>
        </div>

        {/* Statistic Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Total Purchases */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
            <div className="flex justify-between items-start mb-4">
              <div>
                <p className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-1">Total Purchases</p>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">₹2,45,000</h3>
              </div>
              <div className="p-2 bg-green-50 rounded-lg">
                <DollarSign className="h-5 w-5 text-green-600" />
              </div>
            </div>
            <div className="flex items-center text-sm">
              <span className="text-green-600 font-medium flex items-center">
                +12%
              </span>
              <span className="text-gray-400 ml-1">from last month</span>
            </div>
          </div>

          {/* Active Orders */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <div className="flex justify-between items-start mb-4">
              <div>
                <p className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-1">Active Orders</p>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">8</h3>
              </div>
              <div className="p-2 bg-blue-50 rounded-lg">
                <Box className="h-5 w-5 text-blue-600" />
              </div>
            </div>
            <div className="flex items-center text-sm">
              <span className="text-gray-400">3 pending verification</span>
            </div>
          </div>

          {/* Crops Available */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <div className="flex justify-between items-start mb-4">
              <div>
                <p className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-1">Crops Available</p>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">12</h3>
              </div>
              <div className="p-2 bg-green-50 rounded-lg">
                <Sprout className="h-5 w-5 text-green-600" />
              </div>
            </div>
            <div className="flex items-center text-sm">
              <span className="text-gray-400">5 new this week</span>
            </div>
          </div>

          {/* Growth Rate */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <div className="flex justify-between items-start mb-4">
              <div>
                <p className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-1">Growth Rate</p>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">+18%</h3>
              </div>
              <div className="p-2 bg-orange-50 rounded-lg">
                <TrendingUp className="h-5 w-5 text-orange-500" />
              </div>
            </div>
            <div className="flex items-center text-sm">
              <span className="text-gray-400">Compared to last quarter</span>
            </div>
          </div>
        </div>

        {/* Main Sections */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent QR Scans */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
            <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-6">Recent QR Scans</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-100 dark:border-gray-600 hover:border-green-200 dark:hover:border-green-600 transition-colors cursor-pointer">
                <div>
                  <h4 className="font-bold text-gray-900">Rice - Basmati</h4>
                  <p className="text-sm text-gray-500 mt-1">2.5 tons • Verified</p>
                </div>
                <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">
                  Verified
                </span>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-100 hover:border-green-200 transition-colors cursor-pointer">
                <div>
                  <h4 className="font-bold text-gray-900">Wheat - Premium</h4>
                  <p className="text-sm text-gray-500 mt-1">5 tons • Verified</p>
                </div>
                <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">
                  Verified
                </span>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-6">Quick Actions</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <button
                onClick={() => navigate('/vendor/qr-scan')}
                className="flex flex-col items-start p-4 bg-green-50 hover:bg-green-100 rounded-xl transition-colors text-left group"
              >
                <div className="p-2 bg-white rounded-lg mb-3 shadow-sm group-hover:scale-105 transition-transform">
                  <QrCode className="h-6 w-6 text-green-600" />
                </div>
                <span className="font-bold text-green-900">Scan QR</span>
                <span className="text-xs text-green-700 mt-1">Verify product</span>
              </button>

              <button
                onClick={() => navigate('/vendor/market-prices')}
                className="flex flex-col items-start p-4 bg-blue-50 hover:bg-blue-100 rounded-xl transition-colors text-left group"
              >
                <div className="p-2 bg-white rounded-lg mb-3 shadow-sm group-hover:scale-105 transition-transform">
                  <BarChart3 className="h-6 w-6 text-blue-600" />
                </div>
                <span className="font-bold text-blue-900">Market Prices</span>
                <span className="text-xs text-blue-700 mt-1">View rates</span>
              </button>

              <button
                onClick={() => navigate('/vendor/farmer-search')}
                className="flex flex-col items-start p-4 bg-emerald-50 hover:bg-emerald-100 rounded-xl transition-colors text-left group"
              >
                <div className="p-2 bg-white rounded-lg mb-3 shadow-sm group-hover:scale-105 transition-transform">
                  <Search className="h-6 w-6 text-emerald-600" />
                </div>
                <span className="font-bold text-emerald-900">Search Crops</span>
                <span className="text-xs text-emerald-700 mt-1">Find farmers</span>
              </button>

              <button className="flex flex-col items-start p-4 bg-yellow-50 hover:bg-yellow-100 rounded-xl transition-colors text-left group">
                <div className="p-2 bg-white rounded-lg mb-3 shadow-sm group-hover:scale-105 transition-transform">
                  <MessageCircle className="h-6 w-6 text-yellow-600" />
                </div>
                <span className="font-bold text-yellow-900">Chat Support</span>
                <span className="text-xs text-yellow-700 mt-1">Get help</span>
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
