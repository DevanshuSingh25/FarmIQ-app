import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { MapPin, Phone, Calendar, Clock, User, FileText, CheckCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

const requestSchema = z.object({
  farmerName: z.string().min(1, 'Name is required'),
  phone: z.string().regex(/^\d{10}$/, 'Phone number must be exactly 10 digits'),
  location: z.object({
    lat: z.number().optional(),
    lon: z.number().optional(),
    state: z.string().optional(),
    district: z.string().optional(),
    village: z.string().optional(),
    landmark: z.string().optional(),
  }),
  preferredDate: z.string().min(1, 'Preferred date is required'),
  preferredWindow: z.enum(['Morning', 'Afternoon', 'Evening'], {
    required_error: 'Please select a time window',
  }),
  notes: z.string().max(250, 'Notes must be less than 250 characters').optional(),
  allowCalls: z.boolean(),
});

type RequestFormData = z.infer<typeof requestSchema>;

interface RequestFormProps {
  farmerName: string;
  onSubmit: (data: RequestFormData) => void;
}

const states = [
  'Punjab', 'Haryana', 'Rajasthan', 'Uttar Pradesh', 'Madhya Pradesh', 'Gujarat'
];

const districts = {
  'Punjab': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda'],
  'Haryana': ['Gurgaon', 'Faridabad', 'Hisar', 'Karnal', 'Panipat'],
  'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer'],
  'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi', 'Meerut'],
  'Madhya Pradesh': ['Bhopal', 'Indore', 'Gwalior', 'Jabalpur', 'Ujjain'],
  'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar']
};

const villages = {
  'Ludhiana': ['Dakha', 'Jagraon', 'Mullanpur', 'Sahnewal', 'Raikot'],
  'Amritsar': ['Ajnala', 'Beas', 'Majitha', 'Tarn Taran', 'Patti'],
  'Jalandhar': ['Nakodar', 'Phillaur', 'Shahkot', 'Nawanshahr', 'Kapurthala'],
  'Patiala': ['Nabha', 'Rajpura', 'Samana', 'Patran', 'Dudhansadhan'],
  'Bathinda': ['Mansa', 'Sangrur', 'Barnala', 'Muktsar', 'Faridkot']
};

export default function RequestForm({ farmerName, onSubmit }: RequestFormProps) {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [locationMethod, setLocationMethod] = useState<'manual' | 'gps'>('manual');
  const [selectedState, setSelectedState] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    reset
  } = useForm<RequestFormData>({
    resolver: zodResolver(requestSchema),
    defaultValues: {
      farmerName,
      phone: '',
      location: {},
      preferredDate: new Date().toISOString().split('T')[0],
      preferredWindow: 'Morning',
      notes: '',
      allowCalls: true,
    },
  });

  const handleLocationRequest = async () => {
    if (!navigator.geolocation) {
      toast({
        title: 'Location Not Supported',
        description: 'Your browser does not support geolocation. Please fill address manually.',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setValue('location.lat', position.coords.latitude);
        setValue('location.lon', position.coords.longitude);
        setLocationMethod('gps');
        toast({
          title: 'Location Captured',
          description: 'Your current location has been captured successfully.',
        });
        setIsLoading(false);
      },
      (error) => {
        console.error('Geolocation error:', error);
        toast({
          title: 'Location Access Denied',
          description: 'Could not access location. Please fill address manually.',
          variant: 'destructive',
        });
        setIsLoading(false);
      }
    );
  };

  const onFormSubmit = async (data: RequestFormData) => {
    setIsLoading(true);
    try {
      // Validate location
      if (locationMethod === 'gps') {
        if (!data.location.lat || !data.location.lon) {
          throw new Error('Location not captured properly');
        }
      } else {
        if (!data.location.state || !data.location.district || !data.location.village) {
          throw new Error('Please fill all location fields');
        }
      }

      await onSubmit(data);
      reset();
    } catch (error) {
      console.error('Form submission error:', error);
      toast({
        title: 'Validation Error',
        description: error instanceof Error ? error.message : 'Please check all required fields',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Request Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-primary" />
            Request Sensor Installation
          </CardTitle>
          <CardDescription>
            Get professional IoT sensors installed on your farm for real-time monitoring.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Benefits */}
          <div className="mb-6 p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">What you'll get:</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Real-time soil moisture, temperature, and humidity monitoring</li>
              <li>• Smart alerts for irrigation and crop management</li>
              <li>• Professional installation and setup by certified technicians</li>
            </ul>
          </div>

          <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-6">
            {/* Farmer Name */}
            <div className="space-y-2">
              <Label htmlFor="farmerName" className="flex items-center gap-2">
                <User className="h-4 w-4" />
                Farmer Name
              </Label>
              <Input
                id="farmerName"
                {...register('farmerName')}
                placeholder="Enter your full name"
                aria-invalid={errors.farmerName ? 'true' : 'false'}
              />
              {errors.farmerName && (
                <p className="text-sm text-destructive">{errors.farmerName.message}</p>
              )}
            </div>

            {/* Phone Number */}
            <div className="space-y-2">
              <Label htmlFor="phone" className="flex items-center gap-2">
                <Phone className="h-4 w-4" />
                Phone Number
              </Label>
              <Input
                id="phone"
                {...register('phone')}
                placeholder="Enter 10-digit phone number"
                aria-invalid={errors.phone ? 'true' : 'false'}
              />
              {errors.phone && (
                <p className="text-sm text-destructive">{errors.phone.message}</p>
              )}
            </div>

            {/* Location */}
            <div className="space-y-4">
              <Label className="flex items-center gap-2">
                <MapPin className="h-4 w-4" />
                Farm Location
              </Label>

              {/* Location Method Toggle */}
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant={locationMethod === 'gps' ? 'default' : 'outline'}
                  size="sm"
                  onClick={handleLocationRequest}
                  disabled={isLoading}
                >
                  {isLoading ? 'Getting Location...' : 'Use Current Location'}
                </Button>
                <Button
                  type="button"
                  variant={locationMethod === 'manual' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setLocationMethod('manual')}
                >
                  Enter Manually
                </Button>
              </div>

              {locationMethod === 'gps' ? (
                <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-sm text-green-800">
                    ✓ Location captured via GPS
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* State */}
                  <div className="space-y-2">
                    <Label>State</Label>
                    <Select
                      value={selectedState}
                      onValueChange={(value) => {
                        setSelectedState(value);
                        setSelectedDistrict('');
                        setValue('location.state', value);
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select State" />
                      </SelectTrigger>
                      <SelectContent>
                        {states.map((state) => (
                          <SelectItem key={state} value={state}>
                            {state}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* District */}
                  <div className="space-y-2">
                    <Label>District</Label>
                    <Select
                      value={selectedDistrict}
                      onValueChange={(value) => {
                        setSelectedDistrict(value);
                        setValue('location.district', value);
                      }}
                      disabled={!selectedState}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select District" />
                      </SelectTrigger>
                      <SelectContent>
                        {selectedState && districts[selectedState as keyof typeof districts]?.map((district) => (
                          <SelectItem key={district} value={district}>
                            {district}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Village */}
                  <div className="space-y-2">
                    <Label>Village/Mandi</Label>
                    <Select
                      onValueChange={(value) => setValue('location.village', value)}
                      disabled={!selectedDistrict}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select Village" />
                      </SelectTrigger>
                      <SelectContent>
                        {selectedDistrict && villages[selectedDistrict as keyof typeof villages]?.map((village) => (
                          <SelectItem key={village} value={village}>
                            {village}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {/* Landmark */}
              <div className="space-y-2">
                <Label htmlFor="landmark">Landmark (Optional)</Label>
                <Input
                  id="landmark"
                  {...register('location.landmark')}
                  placeholder="e.g., Near bus stand, opposite school"
                />
              </div>
            </div>

            {/* Preferred Date */}
            <div className="space-y-2">
              <Label htmlFor="preferredDate" className="flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                Preferred Visit Date
              </Label>
              <Input
                id="preferredDate"
                type="date"
                {...register('preferredDate')}
                min={new Date().toISOString().split('T')[0]}
                aria-invalid={errors.preferredDate ? 'true' : 'false'}
              />
              {errors.preferredDate && (
                <p className="text-sm text-destructive">{errors.preferredDate.message}</p>
              )}
            </div>

            {/* Preferred Time Window */}
            <div className="space-y-2">
              <Label className="flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Preferred Time Window
              </Label>
              <Select
                value={watch('preferredWindow')}
                onValueChange={(value) => setValue('preferredWindow', value as 'Morning' | 'Afternoon' | 'Evening')}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select time window" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Morning">Morning (8:00 AM - 11:00 AM)</SelectItem>
                  <SelectItem value="Afternoon">Afternoon (12:00 PM - 3:00 PM)</SelectItem>
                  <SelectItem value="Evening">Evening (3:00 PM - 6:00 PM)</SelectItem>
                </SelectContent>
              </Select>
              {errors.preferredWindow && (
                <p className="text-sm text-destructive">{errors.preferredWindow.message}</p>
              )}
            </div>

            {/* Notes */}
            <div className="space-y-2">
              <Label htmlFor="notes" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Notes for Technician (Optional)
              </Label>
              <Textarea
                id="notes"
                {...register('notes')}
                placeholder="Any special instructions or information for the technician..."
                maxLength={250}
                className="resize-none"
              />
              <p className="text-xs text-muted-foreground">
                {watch('notes')?.length || 0}/250 characters
              </p>
            </div>

            {/* Allow Calls Checkbox */}
            <div className="flex items-center space-x-2">
              <Checkbox
                id="allowCalls"
                {...register('allowCalls')}
              />
              <Label htmlFor="allowCalls" className="text-sm">
                I allow the technician to call me for coordination
              </Label>
            </div>

            {/* Submit Button */}
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? 'Submitting Request...' : 'Request Installation'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
