import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Search, RotateCcw } from "lucide-react";
import { MarketPriceFilters } from "@/services/marketPricesService";

interface PricesFiltersProps {
  filters: MarketPriceFilters;
  onFiltersChange: (filters: MarketPriceFilters) => void;
  loading: boolean;
}

const CROPS = [
  "Wheat", "Rice", "Maize", "Potato", "Tomato", "Onion", "Cotton",
  "Sugarcane", "Soybean", "Mustard", "Groundnut", "Gram", "Turmeric"
];

const STATES = [
  "Maharashtra", "Karnataka", "Uttar Pradesh", "Rajasthan", "Gujarat",
  "Madhya Pradesh", "Punjab", "Haryana", "Tamil Nadu", "Andhra Pradesh"
];

const DISTRICTS = {
  "Maharashtra": ["Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad"],
  "Karnataka": ["Bengaluru", "Mysuru", "Belagavi", "Hubli-Dharwad", "Mangaluru"],
  "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
  "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"],
  "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda"],
  "": [] // Default empty state
};

export const PricesFilters = ({ filters, onFiltersChange, loading }: PricesFiltersProps) => {
  const [localFilters, setLocalFilters] = useState<MarketPriceFilters>(filters);
  const [availableDistricts, setAvailableDistricts] = useState<string[]>([]);

  // Don't load saved filters on mount - always start fresh with "all"
  // Filters only save after user clicks Search

  // Update available districts when state changes
  useEffect(() => {
    if (localFilters.state) {
      const districts = DISTRICTS[localFilters.state as keyof typeof DISTRICTS] || [];
      setAvailableDistricts(districts);

      // Clear district if it's not available in the new state
      if (localFilters.district && localFilters.district !== "all" && !districts.includes(localFilters.district)) {
        setLocalFilters(prev => ({ ...prev, district: "all" }));
      }
    } else {
      setAvailableDistricts([]);
      setLocalFilters(prev => ({ ...prev, district: "all" }));
    }
  }, [localFilters.state]);

  const handleFilterChange = (key: keyof MarketPriceFilters, value: string) => {
    setLocalFilters(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSearch = () => {
    onFiltersChange(localFilters);
  };

  const handleReset = () => {
    const resetFilters: MarketPriceFilters = {
      crop: "all",
      state: "all",
      district: "all"
    };
    setLocalFilters(resetFilters);
    onFiltersChange(resetFilters);
    localStorage.removeItem('farmiq-market-filters');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Crop Filter */}
        <div className="space-y-2">
          <Label htmlFor="crop">Crop</Label>
          <Select
            value={localFilters.crop}
            onValueChange={(value) => handleFilterChange('crop', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select crop" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Crops</SelectItem>
              {CROPS.map(crop => (
                <SelectItem key={crop} value={crop}>{crop}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* State Filter */}
        <div className="space-y-2">
          <Label htmlFor="state">State</Label>
          <Select
            value={localFilters.state}
            onValueChange={(value) => handleFilterChange('state', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select state" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All States</SelectItem>
              {STATES.map(state => (
                <SelectItem key={state} value={state}>{state}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* District Filter */}
        <div className="space-y-2">
          <Label htmlFor="district">District/Mandi</Label>
          <Select
            value={localFilters.district}
            onValueChange={(value) => handleFilterChange('district', value)}
            disabled={!localFilters.state}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select district" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Districts</SelectItem>
              {availableDistricts.map(district => (
                <SelectItem key={district} value={district}>{district}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-2">
        <Button
          onClick={handleSearch}
          disabled={loading}
          className="gap-2"
        >
          <Search className="h-4 w-4" />
          {loading ? "Searching..." : "Search"}
        </Button>

        <Button
          variant="outline"
          onClick={handleReset}
          disabled={loading}
          className="gap-2"
        >
          <RotateCcw className="h-4 w-4" />
          Reset
        </Button>
      </div>
    </div>
  );
};