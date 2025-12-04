import { useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { Download, RefreshCw, Wifi, WifiOff, Menu, Globe, Moon, User, Leaf, LogOut, Info, UserCircle, LayoutDashboard, Search as SearchIcon, BarChart3, ChevronRight, QrCode } from "lucide-react";
import { PricesFilters } from "@/components/market/PricesFilters";
import { PricesTable } from "@/components/market/PricesTable";
import { marketPricesService, MarketPrice, MarketPriceFilters } from "@/services/marketPricesService";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export interface PaginationState {
    page: number;
    pageSize: number;
    total: number;
}

export interface SortState {
    field: keyof MarketPrice;
    direction: 'asc' | 'desc';
}

export default function VendorMarketPrices() {
    const [prices, setPrices] = useState<MarketPrice[]>([]);
    const [loading, setLoading] = useState(false);
    const [theme, setTheme] = useState<'light' | 'dark'>('light');
    const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');
    const languages = ['English', 'Hindi', 'Punjabi'] as const;

    const toggleTheme = () => {
        setTheme(prev => prev === 'light' ? 'dark' : 'light');
        document.documentElement.classList.toggle('dark');
    };

    const [filters, setFilters] = useState<MarketPriceFilters>({
        crop: "all",
        state: "all",
        district: "all"
    });
    const [pagination, setPagination] = useState<PaginationState>({
        page: 1,
        pageSize: 100,
        total: 0
    });
    const [sort, setSort] = useState<SortState>({
        field: 'commodity',
        direction: 'asc'
    });
    const [lastUpdated, setLastUpdated] = useState<string>("");
    const [latestPriceDate, setLatestPriceDate] = useState<string>("");
    const [nextRefreshIn, setNextRefreshIn] = useState<number>(0);
    const [isOnline, setIsOnline] = useState(navigator.onLine);
    const [error, setError] = useState<string>("");

    const { toast } = useToast();
    const navigate = useNavigate();
    const { logout } = useAuth();

    // Auto-refresh countdown
    useEffect(() => {
        if (nextRefreshIn > 0) {
            const timer = setInterval(() => {
                setNextRefreshIn(prev => {
                    if (prev <= 1) {
                        fetchPrices(filters, pagination.page, pagination.pageSize, sort);
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);
            return () => clearInterval(timer);
        }
    }, [nextRefreshIn, filters, pagination.page, pagination.pageSize, sort]);

    // Online/offline detection
    useEffect(() => {
        const handleOnline = () => setIsOnline(true);
        const handleOffline = () => setIsOnline(false);

        window.addEventListener('online', handleOnline);
        window.addEventListener('offline', handleOffline);

        return () => {
            window.removeEventListener('online', handleOnline);
            window.removeEventListener('offline', handleOffline);
        };
    }, []);

    const fetchPrices = useCallback(async (
        currentFilters: MarketPriceFilters,
        currentPage = 1,
        currentPageSize = 100,
        currentSort = sort
    ) => {
        setLoading(true);
        setError("");

        try {
            const response = await marketPricesService.getPrices(
                currentFilters,
                currentPage,
                currentPageSize,
                `${currentSort.field}:${currentSort.direction}`
            );

            setPrices(response.data);
            setPagination({
                page: response.page,
                pageSize: response.pageSize,
                total: response.total
            });
            setLastUpdated(response.lastUpdated);

            // Extract latest arrival date from the data
            if (response.data && response.data.length > 0) {
                const dates = response.data
                    .map(item => item.arrival_date)
                    .filter(date => date !== null);

                if (dates.length > 0) {
                    // Get the most recent date (assuming DD/MM/YYYY format)
                    const sortedDates = dates.sort((a, b) => {
                        const [dayA, monthA, yearA] = a!.split('/').map(Number);
                        const [dayB, monthB, yearB] = b!.split('/').map(Number);
                        return new Date(yearB, monthB - 1, dayB).getTime() - new Date(yearA, monthA - 1, dayA).getTime();
                    });
                    setLatestPriceDate(sortedDates[0]!);
                }
            }

            // Set next refresh countdown (5 minutes = 300 seconds)
            const nextUpdate = await marketPricesService.getLastUpdated();
            setNextRefreshIn(nextUpdate.nextRefreshInSeconds);

            // Show toast for auto-refresh
            if (currentPage === pagination.page && prices.length > 0) {
                toast({
                    title: "Prices updated",
                    description: "Market prices have been refreshed with latest data",
                });
            }

        } catch (err) {
            setError("Failed to fetch market prices. Please try again.");
            console.error("Error fetching prices:", err);
        } finally {
            setLoading(false);
        }
    }, [sort, pagination.page, prices.length, toast]);

    // Initial load
    useEffect(() => {
        fetchPrices(filters);
    }, []);

    const handleFiltersChange = (newFilters: MarketPriceFilters) => {
        setFilters(newFilters);
        setPagination(prev => ({ ...prev, page: 1 }));
        fetchPrices(newFilters, 1, pagination.pageSize, sort);

        // Save filters to localStorage
        localStorage.setItem('farmiq-market-filters', JSON.stringify(newFilters));
    };

    const handlePageChange = (page: number) => {
        setPagination(prev => ({ ...prev, page }));
        fetchPrices(filters, page, pagination.pageSize, sort);
    };

    const handleSortChange = (field: keyof MarketPrice) => {
        const newSort: SortState = {
            field,
            direction: sort.field === field && sort.direction === 'asc' ? 'desc' : 'asc'
        };
        setSort(newSort);
        fetchPrices(filters, pagination.page, pagination.pageSize, newSort);
    };

    const handleExportCSV = () => {
        try {
            marketPricesService.exportToCSV(prices, filters);
            toast({
                title: "Export successful",
                description: "Market prices data has been downloaded as CSV",
            });
        } catch (err) {
            toast({
                title: "Export failed",
                description: "Unable to export data. Please try again.",
                variant: "destructive",
            });
        }
    };

    const handleRetry = () => {
        fetchPrices(filters, pagination.page, pagination.pageSize, sort);
    };

    const formatTimeRemaining = (seconds: number) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    };

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

                                        <Link to="/vendor/market-prices" className="flex items-center justify-between px-4 py-3 bg-[#FFD700] text-[#5c4d00] rounded-full shadow-sm transition-all group">
                                            <div className="flex items-center gap-3">
                                                <BarChart3 className="h-5 w-5 stroke-[2.5]" />
                                                <span className="font-medium text-sm">Market Price</span>
                                            </div>
                                            <ChevronRight className="h-4 w-4 text-[#5c4d00]/80" />
                                        </Link>

                                        <Link to="/vendor/farmer-search" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <SearchIcon className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
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
                            <Link to="/vendor/market-prices" className="text-gray-900 font-medium border-b-2 border-green-500 px-1 py-5">
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

            {/* Header Section */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900">Market Prices</h1>
                            <p className="text-gray-500 mt-1">Official prices refreshed daily</p>
                            {latestPriceDate && (
                                <p className="text-sm text-gray-500 mt-1">
                                    Prices are updated up to {latestPriceDate}
                                </p>
                            )}
                        </div>

                        <div className="flex items-center gap-4">
                            {!isOnline && (
                                <Badge variant="destructive" className="gap-2">
                                    <WifiOff className="h-4 w-4" />
                                    Offline
                                </Badge>
                            )}

                            {lastUpdated && (
                                <div className="text-right">
                                    <Badge variant="secondary" className="gap-2">
                                        <Wifi className="h-4 w-4" />
                                        Last updated: {new Date(lastUpdated).toLocaleString()}
                                    </Badge>
                                    {nextRefreshIn > 0 && (
                                        <p className="text-xs text-gray-500 mt-1">
                                            Auto-refresh in: {formatTimeRemaining(nextRefreshIn)}
                                        </p>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Offline Banner */}
            {!isOnline && (
                <Alert className="mx-4 mt-4 max-w-[1600px] mx-auto">
                    <WifiOff className="h-4 w-4" />
                    <AlertDescription>
                        You're offline. Showing last saved prices.
                    </AlertDescription>
                </Alert>
            )}

            {/* Error Banner */}
            {error && (
                <Alert variant="destructive" className="mx-4 mt-4 max-w-[1600px] mx-auto">
                    <AlertDescription className="flex items-center justify-between">
                        {error}
                        <Button variant="outline" size="sm" onClick={handleRetry}>
                            <RefreshCw className="h-4 w-4 mr-2" />
                            Retry
                        </Button>
                    </AlertDescription>
                </Alert>
            )}

            {/* Main Content */}
            <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Filters */}
                <div className="sticky top-16 z-10 bg-white border border-gray-200 rounded-xl shadow-sm p-6 mb-6">
                    <PricesFilters
                        filters={filters}
                        onFiltersChange={handleFiltersChange}
                        loading={loading}
                    />
                </div>

                {/* Actions Bar */}
                <div className="flex items-center justify-between mb-6">
                    <p className="text-gray-500">
                        {pagination.total > 0 && (
                            <>Showing {((pagination.page - 1) * pagination.pageSize) + 1} to {Math.min(pagination.page * pagination.pageSize, pagination.total)} of {pagination.total} results</>
                        )}
                        {filters.crop === "all" && pagination.total === 0 && (
                            <span className="text-green-600 font-medium">
                                💡 Tip: Select a crop for precise results
                            </span>
                        )}
                    </p>

                    <Button
                        onClick={handleExportCSV}
                        disabled={prices.length === 0}
                        variant="outline"
                        className="gap-2"
                    >
                        <Download className="h-4 w-4" />
                        Export CSV
                    </Button>
                </div>

                {/* Table */}
                <PricesTable
                    prices={prices}
                    loading={loading}
                    pagination={pagination}
                    sort={sort}
                    onPageChange={handlePageChange}
                    onSortChange={handleSortChange}
                />
            </main>
        </div>
    );
}
