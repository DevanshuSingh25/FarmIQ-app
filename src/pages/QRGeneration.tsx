import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { QrCode, Download, RefreshCw } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { getTranslation, getCurrentLanguage } from "@/lib/translations";

interface QRPayload {
  cropName: string;
  cropVariety: string;
  sowingDate: string;
  expectedDuration: number;
  expectedDurationUnit: "days" | "weeks" | "months";
  expectedPrice: number;
  generatedAt: string;
  version: "v1";
}

const QRGeneration = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');
  
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };
  
  const [formData, setFormData] = useState({
    cropName: "",
    cropVariety: "",
    sowingDate: "",
    expectedDuration: 0,
    expectedDurationUnit: "days" as "days" | "weeks" | "months",
    expectedPrice: 0
  });
  
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [generatedQR, setGeneratedQR] = useState<string | null>(null);
  const [qrPayload, setQrPayload] = useState<QRPayload | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Load from sessionStorage on mount and detect language
  useEffect(() => {
    const savedData = sessionStorage.getItem('qrFormData');
    if (savedData) {
      try {
        const parsed = JSON.parse(savedData);
        setFormData(parsed);
      } catch (error) {
        console.error('Failed to parse saved form data:', error);
      }
    }
    
    // Detect current language
    setLanguage(getCurrentLanguage());
  }, []);

  // Save to sessionStorage whenever form data changes
  useEffect(() => {
    sessionStorage.setItem('qrFormData', JSON.stringify(formData));
  }, [formData]);

  const cropOptions = [
    "Tomato", "Rice", "Wheat", "Corn", "Potato", "Onion", "Chili", "Cotton", 
    "Sugarcane", "Soybean", "Groundnut", "Sunflower", "Mustard", "Barley", "Millet"
  ];

  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.cropName.trim()) {
      newErrors.cropName = getTranslation('qr.validation.cropNameRequired', language);
    }
    
    if (!formData.sowingDate) {
      newErrors.sowingDate = getTranslation('qr.validation.sowingDateRequired', language);
    } else {
      const selectedDate = new Date(formData.sowingDate);
      const today = new Date();
      today.setHours(23, 59, 59, 999); // End of today
      
      if (selectedDate > today) {
        newErrors.sowingDate = getTranslation('qr.validation.sowingDateFuture', language);
      }
    }
    
    if (!formData.expectedDuration || formData.expectedDuration <= 0) {
      newErrors.expectedDuration = getTranslation('qr.validation.durationRequired', language);
    }
    
    if (!formData.expectedPrice || formData.expectedPrice <= 0) {
      newErrors.expectedPrice = "Expected price is required";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const generateRandomQR = () => {
    // Generate a random QR-like pattern (simplified representation)
    const size = 200;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return null;
    
    // White background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, size, size);
    
    // Generate random black squares to simulate QR pattern
    ctx.fillStyle = '#000000';
    const blockSize = 8;
    const blocksPerRow = size / blockSize;
    
    for (let i = 0; i < blocksPerRow; i++) {
      for (let j = 0; j < blocksPerRow; j++) {
        if (Math.random() > 0.5) {
          ctx.fillRect(i * blockSize, j * blockSize, blockSize, blockSize);
        }
      }
    }
    
    // Add corner markers (QR code style)
    const markerSize = 24;
    const positions = [
      [0, 0], [size - markerSize, 0], [0, size - markerSize]
    ];
    
    positions.forEach(([x, y]) => {
      // Outer square
      ctx.fillRect(x, y, markerSize, markerSize);
      // Inner white square
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(x + 4, y + 4, markerSize - 8, markerSize - 8);
      // Inner black square
      ctx.fillStyle = '#000000';
      ctx.fillRect(x + 8, y + 8, markerSize - 16, markerSize - 16);
    });
    
    return canvas.toDataURL();
  };

  const handleGenerateQR = async () => {
    if (!validateForm()) {
      toast({
        title: "Validation Error",
        description: getTranslation('qr.toast.validationError', language),
        variant: "destructive"
      });
      return;
    }

    setIsGenerating(true);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const payload: QRPayload = {
        cropName: formData.cropName,
        cropVariety: formData.cropVariety,
        sowingDate: formData.sowingDate,
        expectedDuration: formData.expectedDuration,
        expectedDurationUnit: formData.expectedDurationUnit,
        expectedPrice: formData.expectedPrice,
        generatedAt: new Date().toISOString(),
        version: "v1"
      };
      
      const qrImage = generateRandomQR();
      
      if (qrImage) {
        setGeneratedQR(qrImage);
        setQrPayload(payload);
        
        toast({
          title: "Success",
          description: getTranslation('qr.toast.generated', language),
        });
      } else {
        throw new Error("Failed to generate QR code");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: getTranslation('qr.toast.generationError', language),
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownloadPDF = () => {
    // Placeholder for PDF download functionality
    toast({
      title: "Coming Soon",
      description: getTranslation('qr.toast.pdfComingSoon', language),
    });
  };

  const handleRegenerate = () => {
    setGeneratedQR(null);
    setQrPayload(null);
  };

  const getText = () => getTranslation('qr.page.description', language);

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
              sectionId="qr-generation-page"
              ariaLabel="Read QR generation page information"
              alwaysVisible
            />
          </div>
          <h1 className="text-3xl font-bold">{getTranslation('qr.page.title', language)}</h1>
        </div>

        <div className="mb-6">
          <p className="text-muted-foreground">
            {getTranslation('qr.page.description', language)}
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Form */}
          <Card>
            <CardHeader>
              <CardTitle>{getTranslation('qr.form.cropName', language)} Information</CardTitle>
              <CardDescription>
                Provide details about your crop
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="cropName">{getTranslation('qr.form.cropName', language)} *</Label>
                <Select 
                  value={formData.cropName} 
                  onValueChange={(value) => setFormData({...formData, cropName: value})}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select crop name" />
                  </SelectTrigger>
                  <SelectContent>
                    {cropOptions.map((crop) => (
                      <SelectItem key={crop} value={crop}>
                        {crop}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.cropName && (
                  <p className="text-sm text-destructive mt-1">{errors.cropName}</p>
                )}
              </div>

              <div>
                <Label htmlFor="cropVariety">{getTranslation('qr.form.cropVariety', language)}</Label>
                <Input
                  id="cropVariety"
                  value={formData.cropVariety}
                  onChange={(e) => setFormData({...formData, cropVariety: e.target.value})}
                  placeholder="e.g., Hybrid, Organic, Traditional"
                />
              </div>

              <div>
                <Label htmlFor="sowingDate">{getTranslation('qr.form.sowingDate', language)} *</Label>
                <Input
                  id="sowingDate"
                  type="date"
                  value={formData.sowingDate}
                  onChange={(e) => setFormData({...formData, sowingDate: e.target.value})}
                  max={new Date().toISOString().split('T')[0]}
                />
                {errors.sowingDate && (
                  <p className="text-sm text-destructive mt-1">{errors.sowingDate}</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="expectedDuration">Expected delivery duration *</Label>
                  <Input
                    id="expectedDuration"
                    type="number"
                    min="1"
                    value={formData.expectedDuration}
                    onChange={(e) => setFormData({...formData, expectedDuration: parseInt(e.target.value) || 0})}
                    placeholder="90"
                  />
                  {errors.expectedDuration && (
                    <p className="text-sm text-destructive mt-1">{errors.expectedDuration}</p>
                  )}
                </div>
                <div>
                  <Label htmlFor="expectedDurationUnit">{getTranslation('qr.form.unit', language)}</Label>
                  <Select 
                    value={formData.expectedDurationUnit} 
                    onValueChange={(value: "days" | "weeks" | "months") => setFormData({...formData, expectedDurationUnit: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="days">{getTranslation('qr.unit.days', language)}</SelectItem>
                      <SelectItem value="weeks">{getTranslation('qr.unit.weeks', language)}</SelectItem>
                      <SelectItem value="months">{getTranslation('qr.unit.months', language)}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="expectedPrice">Expected price *</Label>
                <Input
                  id="expectedPrice"
                  type="number"
                  min="0"
                  step="0.01"
                  value={formData.expectedPrice}
                  onChange={(e) => setFormData({...formData, expectedPrice: parseFloat(e.target.value) || 0})}
                  placeholder="e.g., 5000"
                />
                {errors.expectedPrice && (
                  <p className="text-sm text-destructive mt-1">{errors.expectedPrice}</p>
                )}
              </div>

              <Button 
                className="w-full" 
                size="lg" 
                onClick={handleGenerateQR}
                disabled={isGenerating}
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <QrCode className="h-4 w-4 mr-2" />
                    {getTranslation('qr.action.generate', language)}
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* QR Preview */}
          <div className="space-y-6">
            {generatedQR && qrPayload ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <QrCode className="h-5 w-5" />
                    {getTranslation('qr.preview.title', language)}
                  </CardTitle>
                  <CardDescription>
                    {getTranslation('qr.preview.description', language)}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-center">
                    <div className="border-2 border-border rounded-lg p-4 bg-white">
                      <img 
                        src={generatedQR} 
                        alt="Generated QR Code" 
                        className="w-48 h-48"
                      />
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      onClick={handleRegenerate}
                      className="flex-1"
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      {getTranslation('qr.action.regenerate', language)}
                    </Button>
                    <Button 
                      onClick={handleDownloadPDF}
                      className="flex-1"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      {getTranslation('qr.action.downloadPdf', language)}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle>QR Preview</CardTitle>
                  <CardDescription>
                    Your generated QR will appear here
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center h-48 border-2 border-dashed border-muted-foreground/25 rounded-lg">
                    <div className="text-center text-muted-foreground">
                      <QrCode className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>{getTranslation('qr.preview.placeholder', language)}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

          </div>
        </div>
      </div>
    </div>
  );
};

export default QRGeneration;
