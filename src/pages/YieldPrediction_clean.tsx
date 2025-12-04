import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { TrendingUp, BarChart3, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { createCrop } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

const YieldPrediction = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  // Form state
  const [formData, setFormData] = useState({
    cropName: "",
    costPrice: "",
    sellingPrice: "",
    kgProduced: ""
  });
  const [showResults, setShowResults] = useState(false);
  const [saving, setSaving] = useState(false);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  // Supported crops and mock monthly yield data (past year)
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const cropToMonthlyYield: Record<string, number[]> = {
    rice: [2.2, 2.5, 2.8, 3.0, 3.1, 2.9, 2.6, 2.4, 2.3, 2.1, 1.9, 1.8],
    wheat: [1.1, 1.3, 1.7, 2.0, 2.2, 2.1, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9],
    tomato: [0.8, 1.0, 1.2, 1.6, 1.9, 2.1, 2.3, 2.0, 1.7, 1.3, 1.0, 0.8],
    corn: [1.3, 1.5, 1.8, 2.2, 2.5, 2.7, 2.6, 2.4, 2.2, 1.9, 1.6, 1.4]
  };

  const normalizeCropKey = (name: string) => name.trim().toLowerCase();
  const getCropKey = (name: string) => {
    const key = normalizeCropKey(name);
    if (cropToMonthlyYield[key]) return key;
    // Simple mapping for common variants
    if (key.includes('paddy')) return 'rice';
    if (key.includes('maize')) return 'corn';
    return 'rice';
  };

  const buildYieldSeries = (cropName: string, kgProduced: number) => {
    const key = getCropKey(cropName);
    const yields = cropToMonthlyYield[key] || cropToMonthlyYield.rice;
    // Scale the mock data based on user's actual production
    const scaleFactor = kgProduced > 0 ? kgProduced / 1000 : 1; // Convert kg to tons and scale
    return months.map((m, i) => ({ month: m, yield: Math.round(yields[i] * scaleFactor * 100) / 100 }));
  };

  const buildProfitSeries = (cropName: string, costPrice: number, sellingPrice: number, kgProduced: number) => {
    const key = getCropKey(cropName);
    const yields = cropToMonthlyYield[key] || cropToMonthlyYield.rice;
    const unitMargin = Math.max(0, sellingPrice - costPrice);
    // Scale the mock data based on user's actual production
    const scaleFactor = kgProduced > 0 ? kgProduced / 1000 : 1; // Convert kg to tons and scale
    return months.map((m, i) => ({ month: m, profit: Math.round(unitMargin * yields[i] * scaleFactor * 100) }));
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.cropName.trim()) return;
    if (!formData.costPrice || !formData.sellingPrice || !formData.kgProduced) return;

    // Save to database before showing results
    try {
      setSaving(true);
      await createCrop({
        crop_name: formData.cropName,
        crop_price: parseFloat(formData.costPrice),
        selling_price: parseFloat(formData.sellingPrice),
        crop_produced_kg: parseFloat(formData.kgProduced)
      });
      toast({
        title: 'Success',
        description: 'Crop record saved to database',
      });
      setShowResults(true);
    } catch (error) {
      console.error('Error saving crop:', error);
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to save crop',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const getText = () => "Crop history analytics. Enter crop name, cost price and selling price to view yield and profit trends over the past year for your crop category.";

  return (
    <div className="min-h-screen bg-background">
      <FarmIQNavbar
        theme={theme}
        language={language}
        onThemeToggle={toggleTheme}
        onLanguageChange={setLanguage}
      />

      <div className="container mx-auto px-4 py-8 pt-24">
        <div className="flex items-center gap-4 mb-8 relative group">
          <div className="absolute top-0 right-0 z-10">
            <SectionSpeaker
              getText={getText}
              sectionId="crop-history-page"
              ariaLabel="Read crop history page information"
              alwaysVisible
            />
          </div>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl lg:text-4xl font-bold mb-4">Crop History</h1>
            <p className="text-xl text-muted-foreground">
              Enter your crop and prices to view yield and profit trends over the past year
            </p>
          </div>

          {!showResults ? (
            <Card className="max-w-2xl mx-auto">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Basic Questionnaire
                </CardTitle>
                <CardDescription>
                  Provide crop name, cost price, selling price and production quantity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={onSubmit} className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="cropName">Crop name</Label>
                    <Input
                      id="cropName"
                      placeholder="e.g., Rice, Wheat, Tomato, Corn"
                      value={formData.cropName}
                      onChange={(e) => setFormData({ ...formData, cropName: e.target.value })}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="costPrice">Cost price (per unit)</Label>
                    <Input
                      id="costPrice"
                      type="number"
                      step="0.01"
                      placeholder="e.g., 1500"
                      value={formData.costPrice}
                      onChange={(e) => setFormData({ ...formData, costPrice: e.target.value })}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="sellingPrice">Selling price (per unit)</Label>
                    <Input
                      id="sellingPrice"
                      type="number"
                      step="0.01"
                      placeholder="e.g., 2000"
                      value={formData.sellingPrice}
                      onChange={(e) => setFormData({ ...formData, sellingPrice: e.target.value })}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="kgProduced">Crop produced (kg)</Label>
                    <Input
                      id="kgProduced"
                      type="number"
                      min="0"
                      step="0.1"
                      placeholder="e.g., 2500"
                      value={formData.kgProduced}
                      onChange={(e) => setFormData({ ...formData, kgProduced: e.target.value })}
                      required
                    />
                  </div>
                  <Button type="submit" className="w-full" size="lg" disabled={saving}>
                    {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    {saving ? 'Saving...' : 'View Crop Trends'}
                  </Button>
                </form>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-8">
              <div className="grid md:grid-cols-2 gap-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Yield over past year — {formData.cropName}
                    </CardTitle>
                    <CardDescription>
                      Monthly yields for the selected crop category
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={buildYieldSeries(formData.cropName, parseFloat(formData.kgProduced))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey="yield"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          name="Yield (tons)"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Profit over past year — {formData.cropName}
                    </CardTitle>
                    <CardDescription>
                      Computed from your cost and selling price
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={buildProfitSeries(formData.cropName, parseFloat(formData.costPrice), parseFloat(formData.sellingPrice), parseFloat(formData.kgProduced))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Bar
                          dataKey="profit"
                          fill="hsl(var(--success))"
                          name="Profit (₹)"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              <div className="text-center">
                <Button variant="outline" className="mr-4" onClick={() => setShowResults(false)}>
                  Modify answers
                </Button>
                <Button>
                  Download report
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default YieldPrediction;