import { Button } from "@/components/ui/button";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { Play, Menu, Globe, Moon, User, Leaf, LogOut, Info, UserCircle, LayoutDashboard, Search, BarChart3, ChevronRight, QrCode } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export default function VendorTeaching() {
    const navigate = useNavigate();
    const { logout } = useAuth();

    const [theme, setTheme] = useState<'light' | 'dark'>(() => {
        const saved = localStorage.getItem('farmiq-theme');
        return (saved as 'light' | 'dark') || 'light';
    });

    const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>(() => {
        const saved = localStorage.getItem('farmiq-language');
        return (saved as 'English' | 'Hindi' | 'Punjabi') || 'English';
    });

    const languages = ['English', 'Hindi', 'Punjabi'] as const;

    const toggleTheme = () => {
        const newTheme = theme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
        localStorage.setItem('farmiq-theme', newTheme);
        if (newTheme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    };

    const handleLanguageChange = (lang: 'English' | 'Hindi' | 'Punjabi') => {
        setLanguage(lang);
        localStorage.setItem('farmiq-language', lang);
    };

    // Apply theme on mount
    useEffect(() => {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, []);

    return (
        <div className="min-h-screen bg-[#F8F9FA] font-sans text-gray-900">
            {/* Vendor Dashboard Navbar */}
            <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
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
                                <SheetContent side="left" className="w-[300px] sm:w-[340px] p-6 bg-[#F8F9FA]">
                                    <SheetHeader className="mb-8 flex flex-row items-center justify-between space-y-0">
                                        <SheetTitle className="text-xl font-bold text-gray-900">Navigation</SheetTitle>
                                    </SheetHeader>
                                    <div className="space-y-3">
                                        <Link to="/vendor/dashboard" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <LayoutDashboard className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                                                <span className="font-medium text-sm">Dashboard</span>
                                            </div>
                                            <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                                        </Link>

                                        <Link to="/vendor/qr-scan" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <QrCode className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                                                <span className="font-medium text-sm">QR Scan</span>
                                            </div>
                                            <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                                        </Link>

                                        <Link to="/vendor/market-prices" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <BarChart3 className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                                                <span className="font-medium text-sm">Market Price</span>
                                            </div>
                                            <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                                        </Link>

                                        <Link to="/vendor/farmer-search" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <Search className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                                                <span className="font-medium text-sm">Farmer Search</span>
                                            </div>
                                            <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                                        </Link>
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
                            <Link to="/vendor/dashboard" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
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
                                            onClick={() => handleLanguageChange(lang)}
                                            className="cursor-pointer hover:bg-gray-50"
                                        >
                                            {lang}
                                            {language === lang && <span className="ml-2 text-green-600">✓</span>}
                                        </DropdownMenuItem>
                                    ))}
                                </DropdownMenuContent>
                            </DropdownMenu>

                            <button
                                className="p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors"
                                onClick={toggleTheme}
                            >
                                <Moon className="h-5 w-5" />
                            </button>

                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors">
                                        <User className="h-5 w-5" />
                                    </button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end" className="w-56">
                                    <DropdownMenuItem
                                        className="cursor-pointer"
                                        onClick={() => navigate('/profile')}
                                    >
                                        <UserCircle className="mr-2 h-4 w-4" />
                                        <span>Profile</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                        className="cursor-pointer"
                                        onClick={() => navigate('/vendor/teaching')}
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

            {/* Content */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <h1 className="text-3xl font-bold text-gray-900">Know about the website</h1>
                    <p className="text-gray-500 mt-1">Add your single YouTube video below.</p>
                </div>
            </div>

            <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
                    <AspectRatio ratio={16 / 9}>
                        <div className="w-full h-full flex items-center justify-center rounded-md border-2 border-dashed border-gray-300 bg-gray-50">
                            <div className="text-center">
                                <Play className="h-12 w-12 mx-auto mb-3 text-gray-400" />
                                <p className="text-gray-500">YouTube video placeholder</p>
                            </div>
                        </div>
                    </AspectRatio>
                </div>
            </main>
        </div>
    );
}
