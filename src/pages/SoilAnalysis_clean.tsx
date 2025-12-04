import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Badge } from "@/components/ui/badge";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { Upload, Phone, MapPin, Clock, CheckCircle, Camera, FileText, Loader2, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { getSoilLabs, type SoilLab } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

const SoilAnalysis = () => {
  const navigate = useNavigate();
  const { user, isAuthenticated, isLoading } = useAuth();
  const { toast } = useToast();
  const [analysisMethod, setAnalysisMethod] = useState<"lab" | "self" | null>(null);
  const [labOption, setLabOption] = useState<"pickup" | "delivery" | null>(null);
  const [labs, setLabs] = useState<SoilLab[]>([]);
  const [labsLoading, setLabsLoading] = useState(false);

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  // Fetch labs when lab method is selected
  useEffect(() => {
    if (analysisMethod === 'lab') {
      fetchLabs();
    }
  }, [analysisMethod]);

  const fetchLabs = async () => {
    try {
      setLabsLoading(true);
      const data = await getSoilLabs();
      setLabs(data);
    } catch (error) {
      console.error('Error fetching soil labs:', error);
      toast({
        title: 'Error',
        description: 'Failed to load soil labs',
        variant: 'destructive',
      });
    } finally {
      setLabsLoading(false);
    }
  };

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, isLoading, navigate]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };
  const [soilImage, setSoilImage] = useState<File | null>(null);
  const [formData, setFormData] = useState({
    name: "",
    phone: "",
    address: "",
    cropType: "",
    soilType: "",
    issues: "",
    previousCrops: "",
    irrigation: "",
    fertilizers: ""
  });

  // Labs data now fetched from API

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSoilImage(file);
    }
  };

  const handleSelfAnalysis = () => {
    // Simulate ML analysis
    const analysisResult = {
      soilType: "Clay Loam",
      ph: "6.2 (Slightly Acidic)",
      nitrogen: "Low",
      phosphorus: "Medium",
      potassium: "High",
      organicMatter: "3.2%",
      recommendations: [
        "Add organic compost to improve nitrogen levels",
        "Apply lime to increase pH to neutral range (6.8-7.2)",
        "Reduce potassium-based fertilizers this season",
        "Consider crop rotation with legumes"
      ],
      suitableCrops: ["Wheat", "Rice", "Sugarcane", "Cotton"]
    };

    // Here you would normally send the data to ML service
    console.log("Analysis submitted:", { formData, soilImage, analysisResult });
  };

  if (analysisMethod === null) {
    const getText = () => "Soil Analysis options: Choose between Lab Testing for professional soil analysis by certified laboratories starting from 250 rupees, or Self Analysis for quick AI-powered analysis using your smartphone, available for free.";

    return (
      <div className="min-h-screen bg-background">
        <FarmIQNavbar
          theme={theme}
          language={language}
          onThemeToggle={toggleTheme}
          onLanguageChange={setLanguage}
        />

        <div className="container mx-auto max-w-4xl p-4 pt-24">
          <div className="flex items-center gap-4 mb-8 relative group">
            <div className="absolute top-0 right-0 z-10">
              <SectionSpeaker
                getText={getText}
                sectionId="soil-analysis-options"
                ariaLabel="Read soil analysis options"
                alwaysVisible
              />
            </div>
            <h1 className="text-3xl font-bold">Soil Analysis</h1>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <Card className="cursor-pointer hover:shadow-lg transition-smooth border-primary/20 hover:border-primary/50"
              onClick={() => setAnalysisMethod("lab")}>
              <CardHeader className="text-center">
                <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
                  <FileText className="h-8 w-8 text-primary" />
                </div>
                <CardTitle className="text-xl">Lab Testing</CardTitle>
                <CardDescription>
                  Professional soil analysis by certified laboratories
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">Highly accurate results</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">Professional recommendations</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">Door pickup available</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">Detailed nutrient analysis</span>
                  </div>
                </div>
                <div className="mt-6">
                  <Badge variant="secondary" className="w-full justify-center py-2">
                    Starting from ₹250
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-smooth border-primary/20 hover:border-primary/50"
              onClick={() => setAnalysisMethod("self")}>
              <CardHeader className="text-center">
                <div className="mx-auto w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mb-4">
                  <Camera className="h-8 w-8 text-accent" />
                </div>
                <CardTitle className="text-xl">Self Analysis</CardTitle>
                <CardDescription>
                  Quick AI-powered analysis using your smartphone
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">Instant results</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">No lab visit required</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">AI-powered recommendations</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm">Available 24/7</span>
                  </div>
                </div>
                <div className="mt-6">
                  <Badge variant="secondary" className="w-full justify-center py-2 bg-success/10 text-success">
                    Free
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    );
  }

  if (analysisMethod === "lab") {
    const getText = () => "Lab Testing Service: Choose from certified laboratories including Krishi Soil Testing Lab in Delhi at 300 rupees, AgriTech Labs in Mumbai at 450 rupees, and Bharat Soil Center in Bangalore at 250 rupees. All offer door pickup services and expert consultation.";

    return (
      <div className="min-h-screen bg-background">
        <FarmIQNavbar
          theme={theme}
          language={language}
          onThemeToggle={toggleTheme}
          onLanguageChange={setLanguage}
        />

        <div className="container mx-auto max-w-6xl p-4 pt-24">
          <div className="flex items-center gap-4 mb-8 relative group">
            <div className="absolute top-0 right-0 z-10">
              <SectionSpeaker
                getText={getText}
                sectionId="lab-testing-service"
                ariaLabel="Read lab testing service information"
                alwaysVisible
              />
            </div>
            <Button variant="ghost" size="sm" onClick={() => setAnalysisMethod(null)}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <h1 className="text-3xl font-bold">Lab Testing Service</h1>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Lab Selection */}
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold mb-4">Choose a Testing Lab</h2>
              {labsLoading ? (
                <div className="text-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin mx-auto" />
                </div>
              ) : labs.map((lab) => (
                <Card key={lab.id} className="hover:shadow-md transition-smooth">
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle className="text-lg">{lab.name}</CardTitle>
                        <CardDescription className="flex items-center gap-2 mt-1">
                          <MapPin className="h-4 w-4" />
                          {lab.location}
                        </CardDescription>
                      </div>
                      <Badge variant="secondary">⭐ {lab.rating || 'N/A'}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <Phone className="h-4 w-4 text-primary" />
                        <span className="font-medium">{lab.contact_number}</span>
                      </div>
                      <div className="flex justify-between items-center pt-3">
                        <span className="text-2xl font-bold text-primary">₹{lab.price}</span>
                        <Button size="sm">
                          <Phone className="h-4 w-4 mr-2" />
                          Contact Lab
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Contact Form */}
            <div>
              <Card>
                <CardHeader>
                  <CardTitle>Sample Collection Details</CardTitle>
                  <CardDescription>
                    Fill in your details for soil sample collection
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="name">Full Name</Label>
                      <Input
                        id="name"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        placeholder="Enter your name"
                      />
                    </div>
                    <div>
                      <Label htmlFor="phone">Phone Number</Label>
                      <Input
                        id="phone"
                        value={formData.phone}
                        onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                        placeholder="Enter phone number"
                      />
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="address">Farm Address</Label>
                    <Textarea
                      id="address"
                      value={formData.address}
                      onChange={(e) => setFormData({ ...formData, address: e.target.value })}
                      placeholder="Enter complete farm address"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="cropType">Current Crop</Label>
                      <Input
                        id="cropType"
                        value={formData.cropType}
                        onChange={(e) => setFormData({ ...formData, cropType: e.target.value })}
                        placeholder="e.g., Wheat, Rice"
                      />
                    </div>
                    <div>
                      <Label htmlFor="soilType">Soil Type (if known)</Label>
                      <Input
                        id="soilType"
                        value={formData.soilType}
                        onChange={(e) => setFormData({ ...formData, soilType: e.target.value })}
                        placeholder="e.g., Clay, Sandy"
                      />
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="issues">Current Soil Issues</Label>
                    <Textarea
                      id="issues"
                      value={formData.issues}
                      onChange={(e) => setFormData({ ...formData, issues: e.target.value })}
                      placeholder="Describe any problems you're facing"
                    />
                  </div>

                  <div>
                    <Label className="text-sm font-medium">Collection Method</Label>
                    <RadioGroup
                      value={labOption || ""}
                      onValueChange={(value) => setLabOption(value as "pickup" | "delivery")}
                      className="mt-2"
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="pickup" id="pickup" />
                        <Label htmlFor="pickup" className="flex items-center gap-2">
                          <Clock className="h-4 w-4" />
                          Door Pickup Service (+₹50)
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="delivery" id="delivery" />
                        <Label htmlFor="delivery" className="flex items-center gap-2">
                          <MapPin className="h-4 w-4" />
                          Self Delivery to Lab
                        </Label>
                      </div>
                    </RadioGroup>
                  </div>

                  <Button className="w-full" size="lg">
                    Book Soil Testing - ₹300
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Self Analysis Flow
  const selfAnalysisGetText = () => "Self Soil Analysis: Provide soil information including crop type, soil type, previous crops, irrigation method, and fertilizers used. Upload a soil image for AI-powered analysis and get instant recommendations.";

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // Don't render if not authenticated (will redirect)
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background">
      <FarmIQNavbar
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />

      <div className="container mx-auto max-w-4xl p-4 pt-24">
        <div className="flex items-center gap-4 mb-8 relative group">
          <div className="absolute top-0 right-0 z-10">
            <SectionSpeaker
              getText={selfAnalysisGetText}
              sectionId="self-analysis-form"
              ariaLabel="Read self analysis information"
              alwaysVisible
            />
          </div>
          <Button variant="ghost" size="sm" onClick={() => setAnalysisMethod(null)}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <h1 className="text-3xl font-bold">Self Soil Analysis</h1>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Questionnaire */}
          <Card>
            <CardHeader>
              <CardTitle>Soil Information</CardTitle>
              <CardDescription>
                Please provide basic information about your soil
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="cropType">What crop are you planning to grow?</Label>
                <Input
                  id="cropType"
                  value={formData.cropType}
                  onChange={(e) => setFormData({ ...formData, cropType: e.target.value })}
                  placeholder="e.g., Wheat, Rice, Cotton"
                />
              </div>

              <div>
                <Label htmlFor="soilType">What type of soil do you think it is?</Label>
                <Input
                  id="soilType"
                  value={formData.soilType}
                  onChange={(e) => setFormData({ ...formData, soilType: e.target.value })}
                  placeholder="e.g., Clay, Sandy, Loamy"
                />
              </div>

              <div>
                <Label htmlFor="previousCrops">What crops did you grow previously?</Label>
                <Input
                  id="previousCrops"
                  value={formData.previousCrops}
                  onChange={(e) => setFormData({ ...formData, previousCrops: e.target.value })}
                  placeholder="Last 2-3 crops grown"
                />
              </div>

              <div>
                <Label htmlFor="irrigation">How do you water your crops?</Label>
                <Input
                  id="irrigation"
                  value={formData.irrigation}
                  onChange={(e) => setFormData({ ...formData, irrigation: e.target.value })}
                  placeholder="e.g., Rain water, Tube well, Canal"
                />
              </div>

              <div>
                <Label htmlFor="fertilizers">What fertilizers did you use last season?</Label>
                <Textarea
                  id="fertilizers"
                  value={formData.fertilizers}
                  onChange={(e) => setFormData({ ...formData, fertilizers: e.target.value })}
                  placeholder="List fertilizers and quantities"
                />
              </div>

              <div>
                <Label htmlFor="issues">What problems are you facing?</Label>
                <Textarea
                  id="issues"
                  value={formData.issues}
                  onChange={(e) => setFormData({ ...formData, issues: e.target.value })}
                  placeholder="e.g., Poor growth, yellowing leaves, low yield"
                />
              </div>
            </CardContent>
          </Card>

          {/* Image Upload */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Upload Soil Image</CardTitle>
                <CardDescription>
                  Take a clear photo of your soil for AI analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center">
                  {soilImage ? (
                    <div className="space-y-4">
                      <img
                        src={URL.createObjectURL(soilImage)}
                        alt="Soil sample"
                        className="mx-auto max-w-full h-32 object-cover rounded-lg"
                      />
                      <p className="text-sm text-muted-foreground">{soilImage.name}</p>
                      <Button
                        variant="outline"
                        onClick={() => setSoilImage(null)}
                      >
                        Change Image
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
                      <div>
                        <Label htmlFor="soil-image" className="cursor-pointer">
                          <Button variant="outline" asChild>
                            <span>
                              <Camera className="h-4 w-4 mr-2" />
                              Upload Soil Photo
                            </span>
                          </Button>
                        </Label>
                        <Input
                          id="soil-image"
                          type="file"
                          accept="image/*"
                          onChange={handleImageUpload}
                          className="hidden"
                        />
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Upload a clear image of your soil sample
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Tips for Best Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success flex-shrink-0" />
                    <span>Take photo in good lighting</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success flex-shrink-0" />
                    <span>Remove any debris or stones</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success flex-shrink-0" />
                    <span>Show soil texture clearly</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success flex-shrink-0" />
                    <span>Answer all questions honestly</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Button
              className="w-full"
              size="lg"
              onClick={handleSelfAnalysis}
              disabled={!soilImage || !formData.cropType}
            >
              Analyze My Soil
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SoilAnalysis;