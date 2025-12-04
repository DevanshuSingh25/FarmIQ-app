import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { Search, Building2, Calendar, MapPin, ExternalLink, Phone, FileText, Loader2, Filter, ChevronRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { filterEligibleSchemes, type NgoScheme, type EligibilityFilters } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

// List of Indian states
const INDIAN_STATES = [
  "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
  "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
  "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
  "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
  "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
  "Uttar Pradesh", "Uttarakhand", "West Bengal"
];

const CATEGORIES = ["SC", "ST", "OBC", "GEN"];

interface NGOScheme {
  id: string;
  title: string;
  organization: string;
  description: string;
  eligibility: string[];
  benefits: string[];
  applicationDeadline: string;
  location: string;
  category: string;
  contactPhone: string;
  documentsRequired: string[];
  status: 'Active' | 'Upcoming' | 'Closed';
  schemeType?: string;
  officialLink?: string;
  minLand?: number;
  maxLand?: number;
  requiredCategory?: string;
}

// Map DB schema to UI schema
function mapDBSchemeToUI(dbScheme: NgoScheme): NGOScheme {
  return {
    id: dbScheme.id.toString(),
    title: dbScheme.name || 'Untitled Scheme',
    organization: dbScheme.ministry || 'Government',
    description: dbScheme.benefit_text || 'No description available',
    eligibility: dbScheme.eligibility_text ? [dbScheme.eligibility_text] : [],
    benefits: dbScheme.benefit_text ? [dbScheme.benefit_text] : [],
    applicationDeadline: dbScheme.deadline || new Date().toISOString().split('T')[0],
    location: dbScheme.location || dbScheme.required_state || 'India',
    category: dbScheme.scheme_type === 'government' ? 'Government Scheme' : 'NGO Scheme',
    contactPhone: dbScheme.contact_number || 'Not available',
    documentsRequired: Array(dbScheme.no_of_docs_required || 0).fill('Document required'),
    status: (dbScheme.status === 'active' ? 'Active' : dbScheme.status === 'upcoming' ? 'Upcoming' : 'Closed') as 'Active' | 'Upcoming' | 'Closed',
    schemeType: dbScheme.required_state === 'ALL' ? 'Central' : dbScheme.required_state || 'Central',
    officialLink: dbScheme.official_link,
    minLand: dbScheme.min_land,
    maxLand: dbScheme.max_land,
    requiredCategory: dbScheme.required_category
  };
}

const NGOSchemes = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  // Form state
  const [state, setState] = useState<string>('');
  const [landSize, setLandSize] = useState<string>('');
  const [category, setCategory] = useState<string>('');
  const [age, setAge] = useState<string>('');

  // Results and UI state
  const [schemes, setSchemes] = useState<NGOScheme[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSchemeType, setSelectedSchemeType] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [hasSearched, setHasSearched] = useState(false);

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  const handleFindSchemes = async () => {
    // Validate input
    if (!state && !landSize && !category && !age) {
      toast({
        title: 'Input Required',
        description: 'Please fill at least one criteria to find eligible schemes',
        variant: 'destructive',
      });
      return;
    }

    try {
      setLoading(true);
      setHasSearched(true);

      const filters: EligibilityFilters = {
        state: state || undefined,
        land: landSize ? parseFloat(landSize) : undefined,
        category: category || undefined,
        age: age ? parseInt(age) : undefined,
      };

      console.log('🔍 Submitting filters:', filters);

      const eligibleSchemes = await filterEligibleSchemes(filters);
      const uiSchemes = eligibleSchemes.map(mapDBSchemeToUI);

      setSchemes(uiSchemes);

      toast({
        title: 'Success',
        description: `Found ${uiSchemes.length} eligible schemes`,
      });
    } catch (error) {
      console.error('Error fetching eligible schemes:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch eligible schemes. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const filteredSchemes = schemes.filter(scheme => {
    const matchesSearch = scheme.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      scheme.organization.toLowerCase().includes(searchTerm.toLowerCase()) ||
      scheme.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSchemeType = selectedSchemeType === 'all' ||
      (selectedSchemeType === 'Central' && scheme.schemeType === 'Central') ||
      (selectedSchemeType !== 'Central' && scheme.schemeType === selectedSchemeType);
    const matchesStatus = selectedStatus === 'all' || scheme.status === selectedStatus;

    return matchesSearch && matchesSchemeType && matchesStatus;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Active': return 'success';
      case 'Upcoming': return 'warning';
      case 'Closed': return 'destructive';
      default: return 'secondary';
    }
  };

  const schemeTypes = ['all', 'Central', 'Punjab'];
  const statuses = ['all', 'Active', 'Upcoming', 'Closed'];

  return (
    <div className="min-h-screen bg-background">
      <FarmIQNavbar
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />

      {/* Header */}
      <div className="border-b bg-card">
        <div className="container mx-auto px-4 py-6 pt-24">
          <div className="flex items-center gap-3 mb-2">
            <Building2 className="h-8 w-8 text-primary" />
            <h1 className="text-3xl font-bold">Government Schemes Eligibility Checker</h1>
          </div>
          <p className="text-muted-foreground">
            Find government schemes you're eligible for based on your profile
          </p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-6">
        {/* Eligibility Input Form */}
        <Card className="mb-6 border-2 border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Enter Your Details to Find Eligible Schemes
            </CardTitle>
            <CardDescription>
              Provide your information to discover schemes tailored for you
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="state">State *</Label>
                <Select value={state} onValueChange={setState}>
                  <SelectTrigger id="state">
                    <SelectValue placeholder="Select your state" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL">All India</SelectItem>
                    {INDIAN_STATES.map(s => (
                      <SelectItem key={s} value={s}>{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="landSize">Land Size (hectares)</Label>
                <Input
                  id="landSize"
                  type="number"
                  placeholder="e.g., 1.5"
                  value={landSize}
                  onChange={(e) => setLandSize(e.target.value)}
                  step="0.1"
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="category">Category</Label>
                <Select value={category} onValueChange={setCategory}>
                  <SelectTrigger id="category">
                    <SelectValue placeholder="Select category" />
                  </SelectTrigger>
                  <SelectContent>
                    {CATEGORIES.map(cat => (
                      <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="age">Age (years)</Label>
                <Input
                  id="age"
                  type="number"
                  placeholder="e.g., 30"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  min="0"
                  max="120"
                />
              </div>
            </div>

            <Button
              onClick={handleFindSchemes}
              className="w-full mt-6 gap-2"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Finding Schemes...
                </>
              ) : (
                <>
                  Find Eligible Schemes
                  <ChevronRight className="h-4 w-4" />
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section - Only show after search */}
        {hasSearched && (
          <>
            {/* Search and Filters */}
            <div className="bg-card border rounded-lg p-6 mb-6">
              <div className="grid gap-4 md:grid-cols-4">
                <div className="md:col-span-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search schemes..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                <Select value={selectedSchemeType} onValueChange={setSelectedSchemeType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Scheme Type" />
                  </SelectTrigger>
                  <SelectContent>
                    {schemeTypes.map(type => (
                      <SelectItem key={type} value={type}>
                        {type === 'all' ? 'All Types' : type}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select value={selectedStatus} onValueChange={setSelectedStatus}>
                  <SelectTrigger>
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    {statuses.map(status => (
                      <SelectItem key={status} value={status}>
                        {status === 'all' ? 'All Status' : status}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Results */}
            <div className="flex items-center justify-between mb-6">
              <p className="text-muted-foreground">
                Found {filteredSchemes.length} eligible schemes
              </p>
            </div>

            {/* Schemes Grid */}
            <div className="grid gap-6">
              {filteredSchemes.map((scheme) => {
                const getText = () => `${scheme.title}. ${scheme.description}. 
                  Organization: ${scheme.organization}. 
                  Location: ${scheme.location}. 
                  Deadline: ${new Date(scheme.applicationDeadline).toLocaleDateString()}`;

                return (
                  <Card key={scheme.id} className="overflow-hidden group hover:shadow-lg transition-shadow">
                    <CardHeader className="pb-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2 flex-wrap">
                            <CardTitle className="text-xl" data-tts="title">
                              {scheme.title}
                            </CardTitle>
                            <Badge variant={getStatusColor(scheme.status) as any}>
                              {scheme.status}
                            </Badge>
                            <Badge variant="outline" className="bg-primary/10">
                              {scheme.schemeType}
                            </Badge>
                            <SectionSpeaker
                              getText={getText}
                              sectionId={`ngo-scheme-${scheme.id}`}
                              ariaLabel={`Read ${scheme.title} scheme details`}
                              alwaysVisible
                            />
                          </div>
                          <div className="flex items-center gap-2 text-muted-foreground">
                            <Building2 className="h-4 w-4" />
                            <span>{scheme.organization}</span>
                          </div>
                        </div>
                        <Badge variant="outline">{scheme.category}</Badge>
                      </div>
                    </CardHeader>

                    <CardContent className="space-y-4">
                      <div>
                        <CardDescription className="mb-2" data-tts="desc">
                          {scheme.description}
                        </CardDescription>
                      </div>

                      {/* Eligibility Details */}
                      {(scheme.minLand || scheme.maxLand || scheme.requiredCategory) && (
                        <div className="bg-muted/50 p-3 rounded-md">
                          <h4 className="font-semibold text-sm mb-2">Eligibility Requirements</h4>
                          <div className="text-sm space-y-1">
                            {scheme.minLand && (
                              <p>• Minimum land: {scheme.minLand} hectares</p>
                            )}
                            {scheme.maxLand && (
                              <p>• Maximum land: {scheme.maxLand} hectares</p>
                            )}
                            {scheme.requiredCategory && scheme.requiredCategory !== 'ALL' && (
                              <p>• Category: {scheme.requiredCategory}</p>
                            )}
                          </div>
                        </div>
                      )}

                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-semibold mb-2">Eligibility</h4>
                          <ul className="space-y-1">
                            {scheme.eligibility.map((item, index) => (
                              <li key={index} className="text-sm text-muted-foreground">
                                • {item}
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div>
                          <h4 className="font-semibold mb-2">Benefits</h4>
                          <ul className="space-y-1">
                            {scheme.benefits.map((item, index) => (
                              <li key={index} className="text-sm text-muted-foreground">
                                • {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
                        <div className="flex items-center gap-2 text-sm">
                          <Calendar className="h-4 w-4 text-muted-foreground" />
                          <span>Deadline: {new Date(scheme.applicationDeadline).toLocaleDateString()}</span>
                        </div>

                        <div className="flex items-center gap-2 text-sm">
                          <MapPin className="h-4 w-4 text-muted-foreground" />
                          <span>{scheme.location}</span>
                        </div>

                        <div className="flex items-center gap-2 text-sm">
                          <Phone className="h-4 w-4 text-muted-foreground" />
                          <span>{scheme.contactPhone}</span>
                        </div>

                        <div className="flex items-center gap-2 text-sm">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span>{scheme.documentsRequired.length} docs required</span>
                        </div>
                      </div>

                      <div className="flex gap-2 pt-4">
                        {scheme.officialLink && (
                          <Button
                            variant="default"
                            className="gap-2"
                            onClick={() => window.open(scheme.officialLink, '_blank')}
                          >
                            <ExternalLink className="h-4 w-4" />
                            Visit Official Website
                          </Button>
                        )}
                        <Button variant="outline">
                          View Details
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>

            {filteredSchemes.length === 0 && (
              <div className="text-center py-12">
                <Building2 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No eligible schemes found</h3>
                <p className="text-muted-foreground">
                  Try adjusting your criteria or search filters
                </p>
              </div>
            )}
          </>
        )}

        {/* Initial State - Before Search */}
        {!hasSearched && (
          <div className="text-center py-12">
            <Building2 className="h-16 w-16 text-primary mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-semibold mb-2">Ready to Find Your Schemes?</h3>
            <p className="text-muted-foreground">
              Fill in your details above and click "Find Eligible Schemes" to discover programs tailored for you
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default NGOSchemes;