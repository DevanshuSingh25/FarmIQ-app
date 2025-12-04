import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { useToast } from "@/hooks/use-toast";
import {
  Upload,
  Camera,
  Loader2,
  CheckCircle,
  Clock,
  Eye,
  Trash2
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { callInferenceApi } from "@/services/predictionService";
import { getDiseaseInfo, computeDetectionStatus } from "@/utils/predictionUtils";

interface Detection {
  id: string;
  crop: string;
  image: string;
  note: string;
  result: string;
  cause: string;
  remedies: string;
  date: string;
}

const CropDiseaseDetection = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };
  const [activeTab, setActiveTab] = useState("detect");

  // Form state
  const [selectedCrop, setSelectedCrop] = useState("");
  const [note, setNote] = useState("");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentResult, setCurrentResult] = useState<Detection | null>(null);

  // History removed as per user request

  const crops = ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"];

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid file type",
        description: "Please upload a JPG or PNG image",
        variant: "destructive"
      });
      return;
    }

    // Validate file size (5MB limit)
    if (file.size > 5 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload an image smaller than 5MB",
        variant: "destructive"
      });
      return;
    }

    setSelectedImage(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleCameraCapture = () => {
    if (fileInputRef.current) {
      fileInputRef.current.setAttribute('capture', 'environment');
      fileInputRef.current.click();
    }
  };


  // Clean disease name - remove crop prefix and format
  const cleanDiseaseName = (name: string): string => {
    // Remove crop name prefix (e.g., "Apple___Apple_scab" -> "Apple scab")
    const parts = name.split('___');
    if (parts.length > 1) {
      // Take the second part, replace underscores with spaces
      return parts[1].replace(/_/g, ' ');
    }
    // If no prefix, just replace underscores
    return name.replace(/_/g, ' ');
  };

  const handleAnalyze = async () => {
    if (!selectedImage || !selectedCrop) {
      toast({
        title: "Missing information",
        description: "Please select an image and crop type",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      // Call real inference API - THIS IS NOT MOCK!
      console.log("🚀 [CropDiseaseDetection] Calling REAL inference API...");
      const response = await callInferenceApi(selectedImage);
      const { class_name } = response;

      // Corn is now supported - removed filter

      // Log the response to verify it's real
      console.log("✅ [CropDiseaseDetection] Received REAL prediction:", class_name);
      console.log("   ⚠️ This is from the REAL Keras model - NOT MOCK!");

      // Use existing mapping to get friendly title/cause/treatment
      const info = getDiseaseInfo(class_name);
      const cleanedDiseaseName = cleanDiseaseName(class_name);

      console.log(`🎯 [CropDiseaseDetection] Final result: ${info.title}`);

      // Create result object (no confidence)
      const result: Detection = {
        id: `D${Date.now()}`,
        crop: selectedCrop,
        image: imagePreview || "",
        note,
        result: cleanedDiseaseName,
        cause: info.cause,
        remedies: info.treatment,
        date: new Date().toISOString().split('T')[0]
      };

      setCurrentResult(result);

      toast({
        title: "Analysis complete",
        description: `Disease detected: ${info.title}`,
      });
    } catch (error) {
      console.error("Prediction error:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";

      // Check if it's a network/CORS error
      if (errorMessage.includes("fetch") || errorMessage.includes("Failed to fetch") || errorMessage.includes("CORS")) {
        toast({
          title: "Connection Error",
          description: "Cannot connect to prediction server. Please ensure the inference server is running on http://localhost:8000",
          variant: "destructive"
        });
      } else {
        toast({
          title: "Analysis failed",
          description: errorMessage.includes("Prediction failed")
            ? errorMessage
            : "Unable to process image. Please try again or contact support.",
          variant: "destructive"
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };


  const resetForm = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setSelectedCrop("");
    setNote("");
    setCurrentResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // History functions removed


  return (
    <div className="min-h-screen bg-background">
      <FarmIQNavbar
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />

      <div className="container mx-auto max-w-6xl p-4 pt-24">
        <div className="flex items-center gap-4 mb-8 group relative">
          <div className="absolute top-0 right-0 z-10">
            <SectionSpeaker
              getText={() => "Crop Disease Detection. Upload images of your crops to identify diseases using AI-powered analysis. Get instant diagnosis, treatment recommendations, and expert advice to protect your harvest."}
              sectionId="disease-detection-header"
              ariaLabel="Read crop disease detection information"
              alwaysVisible
            />
          </div>
          <h1 className="text-3xl font-bold">Crop Disease Detection</h1>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-1">
            <TabsTrigger value="detect">Upload & Detect</TabsTrigger>
          </TabsList>

          <TabsContent value="detect" className="space-y-6 group relative">
            <div className="absolute top-2 right-2 z-10">
              <SectionSpeaker
                getText={() => "Upload and Detect tab. Select your crop type, upload an image of affected plants, add optional notes, and click analyze to get AI-powered disease identification and treatment recommendations."}
                sectionId="detect-tab-content"
                ariaLabel="Read disease detection instructions"
                alwaysVisible
              />
            </div>
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Upload Section */}
              <Card>
                <CardHeader>
                  <CardTitle>Upload Plant Image</CardTitle>
                  <CardDescription>
                    Take or upload a clear photo of the affected plant part
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />

                  {imagePreview ? (
                    <div className="relative">
                      <img
                        src={imagePreview}
                        alt="Plant preview"
                        className="w-full h-64 object-cover rounded-lg border"
                      />
                      <Button
                        variant="destructive"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => {
                          setSelectedImage(null);
                          setImagePreview(null);
                          if (fileInputRef.current) fileInputRef.current.value = "";
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ) : (
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center">
                      <div className="flex flex-col items-center gap-4">
                        <div className="flex gap-4">
                          <Button
                            variant="outline"
                            onClick={() => fileInputRef.current?.click()}
                          >
                            <Upload className="h-4 w-4 mr-2" />
                            Upload Image
                          </Button>
                          <Button
                            variant="outline"
                            onClick={handleCameraCapture}
                          >
                            <Camera className="h-4 w-4 mr-2" />
                            Take Photo
                          </Button>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          JPG or PNG • Max 5MB
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="crop">Crop Type *</Label>
                      <Select value={selectedCrop} onValueChange={setSelectedCrop}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select crop type" />
                        </SelectTrigger>
                        <SelectContent>
                          {crops.map((crop) => (
                            <SelectItem key={crop} value={crop}>
                              {crop}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label htmlFor="note">Additional Notes (Optional)</Label>
                      <Textarea
                        id="note"
                        placeholder="Describe the symptoms you're seeing..."
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                        maxLength={250}
                        className="resize-none"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        {note.length}/250 characters
                      </p>
                    </div>

                    <Button
                      onClick={handleAnalyze}
                      disabled={!selectedImage || !selectedCrop || isAnalyzing}
                      className="w-full"
                      size="lg"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Analyzing Image...
                        </>
                      ) : (
                        <>
                          <Eye className="h-4 w-4 mr-2" />
                          Analyze Disease
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Results Section */}
              <div className="space-y-6">
                {currentResult ? (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <CheckCircle className="h-5 w-5 text-success" />
                        Detection Result
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <Label className="text-sm font-medium">Disease Detected</Label>
                        <p className="text-lg font-semibold mt-1">{currentResult.result}</p>
                      </div>

                      <div>
                        <Label className="text-sm font-medium">Cause</Label>
                        <p className="text-sm text-muted-foreground mt-1">{currentResult.cause}</p>
                      </div>

                      <div>
                        <Label className="text-sm font-medium">Recommended Treatment</Label>
                        <p className="text-sm text-muted-foreground mt-1">{currentResult.remedies}</p>
                      </div>

                      <div className="flex gap-2 pt-4">
                        <Button variant="outline" onClick={resetForm} className="w-full">
                          Analyze Another
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ) : (
                  <Card>
                    <CardContent className="flex flex-col items-center justify-center h-64 text-center">
                      <Eye className="h-12 w-12 text-muted-foreground mb-4" />
                      <h3 className="font-medium mb-2">Ready to Analyze</h3>
                      <p className="text-sm text-muted-foreground">
                        Upload an image and select your crop type to get started
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          {/* History tab removed as per user request */}
        </Tabs>
      </div>
    </div>
  );
};

export default CropDiseaseDetection;