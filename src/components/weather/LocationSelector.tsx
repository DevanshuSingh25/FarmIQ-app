import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { MapPin, RefreshCw, Clock } from 'lucide-react';
import { Location, WeatherFilters } from '@/types/weather';
import { useToast } from '@/hooks/use-toast';

interface LocationSelectorProps {
  filters: WeatherFilters;
  onFiltersChange: (filters: WeatherFilters) => void;
  lastUpdated?: string;
  onRefresh: () => void;
  isLoading: boolean;
  onCitySelect: (city: string) => void | Promise<void>;
  onUseCurrentLocation: (coords: { lat: number; lon: number }) => Promise<void>;
}

const PUNJAB_CITIES = [
  'Amritsar',
  'Jandiala',
  'Ajnala',
  'Rayya',
  'Majitha',
  'Raja Sansi',
  'Ramdas',
  'Barnala',
  'Tapa',
  'Dhanaula',
  'Bhadaur',
  'Handiaya',
  'Bathinda',
  'Rampura Phul',
  'Maur',
  'Raman',
  'Talwandi Sabo',
  'Mehraj',
  'Goniana',
  'Bhucho Mandi',
  'Bhai Rupa',
  'Ludhiana',
  'Khanna',
  'Patiala',
  'Rajpura',
  'Jalandhar',
  'Hoshiarpur',
  'Mohali',
  'Moga',
  'Batala',
  'Pathankot',
  'Abohar',
  'Malerkotla',
  'Muktsar',
  'Firozpur',
  'Kapurthala',
  'Phagwara',
  'Zirakpur'
] as const;

export function LocationSelector({
  filters,
  onFiltersChange,
  lastUpdated,
  onRefresh,
  isLoading,
  onCitySelect,
  onUseCurrentLocation
}: LocationSelectorProps) {
  const { toast } = useToast();
  const [selectedCity, setSelectedCity] = useState('');
  const [isGettingLocation, setIsGettingLocation] = useState(false);

  const handleCitySelection = () => {
    if (!selectedCity) {
      toast({
        title: "City required",
        description: "Please select a city from the list.",
        variant: "destructive"
      });
      return;
    }

    const location: Location = {
      lat: 0,
      lon: 0,
      name: `${selectedCity}, Punjab`,
      state: 'Punjab',
      district: selectedCity
    };

    onFiltersChange({
      ...filters,
      location
    });
    onCitySelect(selectedCity);
  };

  const handleCurrentLocation = async () => {
    if (!navigator.geolocation) {
      toast({
        title: "Geolocation not supported",
        description: "Your browser does not support accessing current location.",
        variant: "destructive"
      });
      return;
    }

    setIsGettingLocation(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        onUseCurrentLocation({
          lat: position.coords.latitude,
          lon: position.coords.longitude
        })
          .catch((error) => {
            toast({
              title: "Unable to use current location",
              description: error?.message || "Please try again later.",
              variant: "destructive"
            });
          })
          .finally(() => setIsGettingLocation(false));
      },
      (error) => {
        toast({
          title: "Unable to access location",
          description: error.message || "Please allow location access and try again.",
          variant: "destructive"
        });
        setIsGettingLocation(false);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000
      }
    );
  };

  return (
    <div className="bg-card rounded-lg p-6 shadow-soft border">
      <div className="flex flex-col space-y-6">
        {/* Location Selection */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground">Location</h3>

          <div className="space-y-2">
            <Label htmlFor="city-select" className="text-sm font-medium">Select City</Label>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <Button
                variant="outline"
                onClick={handleCurrentLocation}
                disabled={isGettingLocation || isLoading}
                className="flex items-center gap-2 h-11 px-4 sm:w-auto w-full whitespace-nowrap"
              >
                <MapPin className="h-4 w-4" />
                {isGettingLocation ? 'Detecting...' : 'Use Current Location'}
              </Button>
              <Select value={selectedCity} onValueChange={setSelectedCity}>
                <SelectTrigger id="city-select" className="w-full sm:w-[240px] h-11">
                  <SelectValue placeholder="Choose a city" />
                </SelectTrigger>
                <SelectContent>
                  {PUNJAB_CITIES.map(city => (
                    <SelectItem key={city} value={city}>
                      {city}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={handleCitySelection}
                disabled={isLoading}
                className="h-11 px-6 sm:w-auto w-full"
              >
                Set Location
              </Button>
            </div>
          </div>

          {filters.location && (
            <div className="flex items-center gap-2">
              <MapPin className="h-4 w-4 text-success" />
              <span className="text-sm text-foreground">{filters.location.name}</span>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pt-4 border-t">
          <div className="flex items-center gap-4">
            {lastUpdated && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span>Updated: {new Date(lastUpdated).toLocaleTimeString()}</span>
              </div>
            )}

            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              disabled={isLoading || !filters.location}
              className="flex items-center gap-2"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}