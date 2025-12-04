import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { FarmIQNavbar } from '@/components/farmiq/FarmIQNavbar';
import { Cpu, CheckCircle, MapPin, Phone, Calendar, User } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import LiveReadings from '@/components/iot/LiveReadings';
import { iotService, InstallationRequest, IotStatus } from '@/services/iotService';
import { useToast } from '@/hooks/use-toast';

const IoTSensor = () => {
  const { user, isAuthenticated, isLoading: authLoading } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [activeRequest, setActiveRequest] = useState<InstallationRequest | null>(null);
  const [iotStatus, setIotStatus] = useState<IotStatus | null>(null);
  const [isLoadingRequest, setIsLoadingRequest] = useState(true);
  const [activeTab, setActiveTab] = useState('request');

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    phone_number: '',
    location: '',
    state: '',
    district: '',
    preferred_visit_date: new Date().toISOString().split('T')[0],
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, authLoading, navigate]);

  // Load active request and status on mount
  useEffect(() => {
    if (isAuthenticated && user) {
      loadData();
    }
  }, [isAuthenticated, user]);

  const loadData = async () => {
    if (!user?.id) return;

    try {
      setIsLoadingRequest(true);

      // Load both booking and status
      const [booking, status] = await Promise.all([
        iotService.getBookingRequest(user.id),
        iotService.getStatus(user.id)
      ]);

      setActiveRequest(booking);
      setIotStatus(status);

      // If device is active, switch to readings tab
      if (status?.status === 'active') {
        setActiveTab('readings');
      }
    } catch (error) {
      console.error('Error loading IoT data:', error);
      toast({
        title: 'Error',
        description: 'Failed to load IoT data. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoadingRequest(false);
    }
  };

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!user?.id) return;

    // Validation
    if (!formData.name || !formData.phone_number) {
      toast({
        title: 'Validation Error',
        description: 'Name and phone number are required',
        variant: 'destructive',
      });
      return;
    }

    if (formData.phone_number.length < 7 || formData.phone_number.length > 20) {
      toast({
        title: 'Validation Error',
        description: 'Phone number must be between 7 and 20 characters',
        variant: 'destructive',
      });
      return;
    }

    try {
      setIsSubmitting(true);
      const newRequest = await iotService.createRequest(formData);
      setActiveRequest(newRequest);
      toast({
        title: 'Request Submitted',
        description: 'Your IoT sensor installation request has been submitted successfully.',
      });
    } catch (error) {
      console.error('Error creating request:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to submit request';
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  if (authLoading || !isAuthenticated) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <FarmIQNavbar
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />

      <div className="container mx-auto px-4 py-8 pt-24">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-3">
                <Cpu className="h-8 w-8 text-primary" />
                IoT Sensor
              </h1>
              <p className="text-muted-foreground mt-2">
                Request installation, track status, and view farm readings.
              </p>
            </div>

            {iotStatus && (
              <Badge variant={iotStatus.status === 'active' ? 'default' : 'secondary'}>
                Status: {iotStatus.status}
              </Badge>
            )}
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="request">Request & Status</TabsTrigger>
            <TabsTrigger value="readings">Live Readings</TabsTrigger>
          </TabsList>

          <TabsContent value="request" className="mt-6">
            {isLoadingRequest ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center ">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading status...</p>
                </div>
              </div>
            ) : activeRequest ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    Installation Request Submitted
                  </CardTitle>
                  <CardDescription>
                    Your request is being processed
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center gap-2">
                      <User className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">
                        <span className="font-medium">Name:</span> {activeRequest.name}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <Phone className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">
                        <span className="font-medium">Phone:</span> {activeRequest.phone_number}
                      </span>
                    </div>

                    {activeRequest.state && (
                      <div className="flex items-center gap-2">
                        <MapPin className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">
                          <span className="font-medium">Location:</span> {activeRequest.district}, {activeRequest.state}
                        </span>
                      </div>
                    )}

                    {activeRequest.preferred_visit_date && (
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">
                          <span className="font-medium">Preferred Date:</span> {activeRequest.preferred_visit_date}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="pt-4 border-t">
                    <div className="flex items-center gap-2">
                      <Badge variant={activeRequest.status === 'completed' ? 'default' : 'secondary'}>
                        {activeRequest.status.toUpperCase()}
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        Requested on {new Date(activeRequest.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle>Request IoT Sensor Installation</CardTitle>
                  <CardDescription>
                    Fill out the form below to request sensor installation on your farm
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleFormSubmit} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Full Name</Label>
                      <Input
                        id="name"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        placeholder="Enter your full name"
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="phone">Phone Number</Label>
                      <Input
                        id="phone"
                        value={formData.phone_number}
                        onChange={(e) => setFormData({ ...formData, phone_number: e.target.value })}
                        placeholder="Enter your phone number"
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="location">Location (Optional)</Label>
                      <Input
                        id="location"
                        value={formData.location}
                        onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                        placeholder="Landmark or GPS coordinates"
                      />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="state">State</Label>
                        <Input
                          id="state"
                          value={formData.state}
                          onChange={(e) => setFormData({ ...formData, state: e.target.value })}
                          placeholder="Enter your state"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="district">District</Label>
                        <Input
                          id="district"
                          value={formData.district}
                          onChange={(e) => setFormData({ ...formData, district: e.target.value })}
                          placeholder="Enter your district"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="date">Preferred Visit Date</Label>
                      <Input
                        id="date"
                        type="date"
                        value={formData.preferred_visit_date}
                        onChange={(e) => setFormData({ ...formData, preferred_visit_date: e.target.value })}
                        min={new Date().toISOString().split('T')[0]}
                      />
                    </div>

                    <Button type="submit" className="w-full" disabled={isSubmitting}>
                      {isSubmitting ? 'Submitting...' : 'Request Installation'}
                    </Button>
                  </form>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="readings" className="mt-6">
            <LiveReadings
              isInstalled={iotStatus?.status === 'active'}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default IoTSensor;
