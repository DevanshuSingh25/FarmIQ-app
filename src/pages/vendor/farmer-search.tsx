import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import {
    Menu,
    Globe,
    Moon,
    User,
    Leaf,
    Search,
    MapPin,
    Box,
    Phone,
    UserCircle,
    Info,
    LogOut,
    LayoutDashboard,
    BarChart3,
    ChevronRight,
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

interface Farmer {
    id: number;
    full_name: string;
    crops_grown: string | null;
    available_quantity: number | null;
    location: string | null;
    expected_price: string | null;
}

export default function VendorFarmerSearch() {
    const navigate = useNavigate();
    const { logout } = useAuth();
    const [searchQuery, setSearchQuery] = useState("");
    const [farmers, setFarmers] = useState<Farmer[]>([]);
    const [loading, setLoading] = useState(true);

    const [theme, setTheme] = useState<'light' | 'dark'>('light');
    const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');
    const languages = ['English', 'Hindi', 'Punjabi'] as const;

    const toggleTheme = () => {
        setTheme(prev => prev === 'light' ? 'dark' : 'light');
        document.documentElement.classList.toggle('dark');
    };

    // Fetch farmers from API
    useEffect(() => {
        const fetchFarmers = async () => {
            try {
                const response = await fetch('http://localhost:3001/api/profiles', {
                    credentials: 'include'
                });
                if (response.ok) {
                    const data = await response.json();
                    setFarmers(data);
                } else {
                    console.error('Failed to fetch farmers');
                }
            } catch (error) {
                console.error('Error fetching farmers:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchFarmers();
    }, []);

    const filteredFarmers = farmers.filter(farmer =>
        farmer.full_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (farmer.location && farmer.location.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (farmer.crops_grown && farmer.crops_grown.toLowerCase().includes(searchQuery.toLowerCase()))
    );

    return (
        <div className="min-h-screen bg-[#F8F9FA] font-sans text-gray-900">
            {/* Top Navigation Bar */}
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

                                        <Link to="/vendor/farmer-search" className="flex items-center justify-between px-4 py-3 bg-[#e0f2f1] text-[#004d40] rounded-full shadow-sm transition-all group border border-transparent">
                                            <div className="flex items-center gap-3">
                                                <Search className="h-5 w-5 stroke-[2.5]" />
                                                <span className="font-medium text-sm">Farmer Search</span>
                                            </div>
                                            <ChevronRight className="h-4 w-4 text-[#004d40]/80" />
                                        </Link>

                                        <Link to="/vendor/market-prices" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <BarChart3 className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                                                <span className="font-medium text-sm">Market Price</span>
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
                            <Link to="/vendor/farmer-search" className="text-gray-900 font-medium border-b-2 border-green-500 px-1 py-5">
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
                                            onClick={() => setLanguage(lang)}
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

            {/* Main Content */}
            <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900">Farmer Search</h1>
                </div>

                {/* Search Bar */}
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 mb-8">
                    <div className="flex gap-4">
                        <div className="flex-1 relative">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Search Farmer by Name or Location"
                                className="w-full pl-12 pr-4 py-3 bg-white border border-gray-200 rounded-lg text-gray-900 focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-500 transition-all"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                        <button className="px-8 py-3 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors">
                            Search
                        </button>
                    </div>
                </div>

                {/* Results Count */}
                <div className="mb-6">
                    <p className="text-gray-500">
                        {loading ? 'Loading farmers...' : `Found ${filteredFarmers.length} farmers ${searchQuery ? `matching "${searchQuery}"` : ''}`}
                    </p>
                </div>

                {/* Farmer Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {loading ? (
                        <div className="col-span-full text-center py-12 text-gray-500">
                            Loading farmers...
                        </div>
                    ) : filteredFarmers.length === 0 ? (
                        <div className="col-span-full text-center py-12 text-gray-500">
                            No farmers found {searchQuery ? `matching "${searchQuery}"` : ''}
                        </div>
                    ) : (
                        filteredFarmers.map((farmer) => (
                            <div key={farmer.id} className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-all">
                                <div className="flex justify-between items-start mb-4">
                                    <div>
                                        <h3 className="text-lg font-bold text-gray-900 mb-1">{farmer.full_name || 'Unknown'}</h3>
                                    </div>
                                </div>

                                <div className="space-y-4 mb-6">
                                    <div className="flex items-start gap-3">
                                        <Box className="h-5 w-5 text-gray-400 mt-0.5" />
                                        <div>
                                            <p className="text-xs text-gray-500 uppercase font-medium">Crops Grown</p>
                                            <p className="text-gray-900 font-medium">
                                                {farmer.crops_grown || '-'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex items-start gap-3">
                                        <Box className="h-5 w-5 text-gray-400 mt-0.5" />
                                        <div>
                                            <p className="text-xs text-gray-500 uppercase font-medium">Available Quantity</p>
                                            <p className="text-gray-900 font-medium">
                                                {farmer.available_quantity ? `${farmer.available_quantity} kg` : '-'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex items-start gap-3">
                                        <MapPin className="h-5 w-5 text-gray-400 mt-0.5" />
                                        <div>
                                            <p className="text-xs text-gray-500 uppercase font-medium">Location</p>
                                            <p className="text-gray-900 font-medium">
                                                {farmer.location || '-'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex items-start gap-3">
                                        <div className="h-5 w-5 flex items-center justify-center text-gray-400 font-serif font-bold">₹</div>
                                        <div>
                                            <p className="text-xs text-gray-500 uppercase font-medium">Expected Price</p>
                                            <p className="text-gray-900 font-bold">
                                                {farmer.expected_price || '-'}
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <button className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors">
                                    <Phone className="h-4 w-4" />
                                    <span>Connect with Farmer</span>
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </main>
        </div>
    );
}
