// Market Prices Service - Backend Proxy Integration
// Calls server-side proxy at /api/market-prices which fetches from data.gov.in API

export interface MarketPrice {
  state: string | null;
  district: string | null;
  market: string | null;
  commodity: string | null;
  variety: string | null;
  min_price: number | null;
  max_price: number | null;
  modal_price: number | null;
  arrival_date: string | null;
}

export interface MarketPriceFilters {
  crop: string;
  state: string;
  district: string;
  // date field removed - using latest data from API
}

export interface MarketPricesResponse {
  data: MarketPrice[];
  page: number;
  pageSize: number;
  total: number;
  lastUpdated: string;
}

class MarketPricesService {

  /**
   * Fetch market prices from backend proxy
   */
  async getPrices(
    filters: MarketPriceFilters,
    page: number = 1,
    pageSize: number = 100,
    sort: string = 'commodity:asc'
  ): Promise<MarketPricesResponse> {
    try {
      // Build query params
      const params = new URLSearchParams();

      // Only add filters if they are NOT "all" values
      if (filters.state && filters.state !== 'all' && filters.state !== 'All States') {
        params.append('state', filters.state);
      }
      if (filters.district && filters.district !== 'all' && filters.district !== 'All Districts') {
        params.append('district', filters.district);
      }
      if (filters.crop && filters.crop !== 'all' && filters.crop !== 'All Crops') {
        params.append('commodity', filters.crop); // Map crop to commodity
      }

      // Calculate offset from page
      const offset = (page - 1) * pageSize;
      params.append('offset', String(offset));
      params.append('limit', String(pageSize));

      // Make request to backend proxy
      const response = await fetch(`/api/market-prices?${params.toString()}`, {
        credentials: 'include',
        headers: {
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend error ${response.status}: ${errorText}`);
      }

      const result = await response.json();

      // Transform backend response to match expected format
      const data = result.data || [];

      // Client-side sorting (since backend doesn't sort)
      const sortedData = this.sortData(data, sort);

      return {
        data: sortedData,
        page,
        pageSize,
        total: result.meta?.count || data.length,
        lastUpdated: new Date().toISOString()
      };

    } catch (error) {
      console.error('Error fetching market prices:', error);
      throw error;
    }
  }

  /**
   * Client-side sorting helper
   */
  private sortData(data: MarketPrice[], sortString: string): MarketPrice[] {
    const [field, direction] = sortString.split(':');
    const sortField = field as keyof MarketPrice;

    return [...data].sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      // Handle null values
      if (aVal === null && bVal === null) return 0;
      if (aVal === null) return 1;
      if (bVal === null) return -1;

      // Handle numeric fields
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return direction === 'asc' ? aVal - bVal : bVal - aVal;
      }

      // Handle string fields
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return direction === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      return 0;
    });
  }

  /**
   * Get last updated info - now just returns current time
   */
  async getLastUpdated(): Promise<{ lastUpdated: string; nextRefreshInSeconds: number }> {
    return {
      lastUpdated: new Date().toISOString(),
      nextRefreshInSeconds: 300 // 5 minutes (matches server cache TTL)
    };
  }

  /**
   * Export current data to CSV
   */
  exportToCSV(data: MarketPrice[], filters: MarketPriceFilters): void {
    if (data.length === 0) {
      throw new Error('No data to export');
    }

    // Create CSV headers - matching new column order
    const headers = [
      'State', 'District/Mandi', 'Market', 'Commodity', 'Variety',
      'Min Price (₹)', 'Max Price (₹)', 'Modal Price (₹)'
    ];

    // Create CSV rows
    const rows = data.map(item => [
      item.state || '—',
      item.district || '—',
      item.market || '—',
      item.commodity || '—',
      item.variety || '—',
      item.min_price !== null ? item.min_price.toString() : '—',
      item.max_price !== null ? item.max_price.toString() : '—',
      item.modal_price !== null ? item.modal_price.toString() : '—'
    ]);

    // Combine headers and rows
    const csvContent = [headers, ...rows]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n');

    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');

    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);

      // Generate filename with filters (no date)
      const filterParts = [];
      if (filters.crop && filters.crop !== 'all') filterParts.push(filters.crop);
      if (filters.state && filters.state !== 'all') filterParts.push(filters.state);
      if (filters.district && filters.district !== 'all') filterParts.push(filters.district);

      const filterString = filterParts.length > 0 ? `_${filterParts.join('_')}` : '';
      const today = new Date().toISOString().split('T')[0];
      const filename = `FarmIQ_MarketPrices${filterString}_${today}.csv`;

      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }
}

export const marketPricesService = new MarketPricesService();