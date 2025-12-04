import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Loader2, TrendingUp } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { createCrop } from '@/services/api';

export default function CropHistoryForm() {
    const navigate = useNavigate();
    const { toast } = useToast();
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        crop_name: '',
        crop_price: '',
        selling_price: '',
        crop_produced_kg: '',
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setLoading(true);
            await createCrop({
                crop_name: formData.crop_name,
                crop_price: parseFloat(formData.crop_price) || 0,
                selling_price: parseFloat(formData.selling_price) || 0,
                crop_produced_kg: parseFloat(formData.crop_produced_kg) || 0,
            });

            toast({
                title: 'Success',
                description: 'Crop record added successfully',
            });

            // Navigate to graphs page
            navigate('/crop-history/graphs');
        } catch (error) {
            console.error('Error creating crop:', error);
            toast({
                title: 'Error',
                description: error instanceof Error ? error.message : 'Failed to add crop record',
                variant: 'destructive',
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background p-8">
            <div className="max-w-2xl mx-auto">
                <Card>
                    <CardHeader>
                        <CardTitle className="text-2xl">Add Crop Record</CardTitle>
                        <CardDescription>
                            Enter your crop details to track yield and profit trends
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="crop_name">Crop Name*</Label>
                                <Input
                                    id="crop_name"
                                    value={formData.crop_name}
                                    onChange={(e) => setFormData({ ...formData, crop_name: e.target.value })}
                                    placeholder="e.g., Wheat, Rice, Corn"
                                    required
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="crop_price">Cost Price (₹/kg)*</Label>
                                    <Input
                                        id="crop_price"
                                        type="number"
                                        step="0.01"
                                        value={formData.crop_price}
                                        onChange={(e) => setFormData({ ...formData, crop_price: e.target.value })}
                                        placeholder="0.00"
                                        required
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="selling_price">Selling Price (₹/kg)*</Label>
                                    <Input
                                        id="selling_price"
                                        type="number"
                                        step="0.01"
                                        value={formData.selling_price}
                                        onChange={(e) => setFormData({ ...formData, selling_price: e.target.value })}
                                        placeholder="0.00"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="crop_produced_kg">Crop Produced (kg)*</Label>
                                <Input
                                    id="crop_produced_kg"
                                    type="number"
                                    step="0.01"
                                    value={formData.crop_produced_kg}
                                    onChange={(e) => setFormData({ ...formData, crop_produced_kg: e.target.value })}
                                    placeholder="0.00"
                                    required
                                />
                            </div>

                            <Button
                                type="submit"
                                className="w-full bg-green-600 hover:bg-green-700"
                                disabled={loading}
                            >
                                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                                <TrendingUp className="mr-2 h-4 w-4" />
                                View Crop Trends
                            </Button>
                        </form>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
