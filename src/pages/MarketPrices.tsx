import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import { Download, RefreshCw, Wifi, WifiOff, Home } from "lucide-react";
import { PricesFilters } from "@/components/market/PricesFilters";
import { PricesTable } from "@/components/market/PricesTable";
import { marketPricesService, MarketPrice, MarketPriceFilters } from "@/services/marketPricesService";

export interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

export interface SortState {
  field: keyof MarketPrice;
  direction: 'asc' | 'desc';
}

const MarketPrices = () => {
  const [prices, setPrices] = useState<MarketPrice[]>([]);
  const [loading, setLoading] = useState(false);

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

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

      // Set next refresh countdown (24 hours = 86400 seconds)
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

  // Initial load - NO auto-fetch, user must click "Search" button
  // useEffect removed so results are empty until search clicked

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
    <div className="min-h-screen bg-background">
      <FarmIQNavbar
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />

      {/* Header */}
      <div className="border-b bg-card group relative">
        <div className="absolute top-4 right-4 z-10">
          <SectionSpeaker
            getText={() => "Market Prices page. Get official crop prices refreshed daily. Filter by crop type, state, and district to find current market rates and make informed selling decisions."}
            sectionId="market-prices-header"
            ariaLabel="Read market prices page information"
            alwaysVisible
          />
        </div>
        <div className="container mx-auto px-4 py-6 pt-24">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-4 mb-2">
              </div>
              <h1 className="text-3xl font-bold">Market Prices</h1>
              <p className="text-muted-foreground mt-1">
                Official prices refreshed daily
              </p>
              {latestPriceDate && (
                <p className="text-sm text-muted-foreground mt-1">
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
                    <p className="text-xs text-muted-foreground mt-1">
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
        <Alert className="mx-4 mt-4">
          <WifiOff className="h-4 w-4" />
          <AlertDescription>
            You're offline. Showing last saved prices.
          </AlertDescription>
        </Alert>
      )}

      {/* Error Banner */}
      {error && (
        <Alert variant="destructive" className="mx-4 mt-4">
          <AlertDescription className="flex items-center justify-between">
            {error}
            <Button variant="outline" size="sm" onClick={handleRetry}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div className="container mx-auto px-4 py-6">
        {/* Filters */}
        <div className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm border rounded-lg p-4 mb-6">
          <PricesFilters
            filters={filters}
            onFiltersChange={handleFiltersChange}
            loading={loading}
          />
        </div>

        {/* Actions Bar */}
        <div className="flex items-center justify-between mb-6">
          <p className="text-muted-foreground">
            {pagination.total > 0 && (
              <>Showing {((pagination.page - 1) * pagination.pageSize) + 1} to {Math.min(pagination.page * pagination.pageSize, pagination.total)} of {pagination.total} results</>
            )}
            {filters.crop === "all" && pagination.total === 0 && (
              <span className="text-accent font-medium">
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
      </div>
    </div>
  );
};

export default MarketPrices;