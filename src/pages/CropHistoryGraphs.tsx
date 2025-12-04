import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, Edit, Download, Plus } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { getCrops, type CropHistory } from '@/services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface MonthlyData {
    month: string;
    yield_tons: number;
    profit: number;
}

export default function CropHistoryGraphs() {
    const navigate = useNavigate();
    const { toast } = useToast();
    const [crops, setCrops] = useState<CropHistory[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchCrops();
    }, []);

    const fetchCrops = async () => {
        try {
            setLoading(true);
            const data = await getCrops();
            setCrops(data);
        } catch (error) {
            console.error('Error fetching crops:', error);
            toast({
                title: 'Error',
                description: 'Failed to load crop data',
                variant: 'destructive',
            });
        } finally {
            setLoading(false);
        }
    };

    // Aggregate data by month
    const monthlyData: MonthlyData[] = useMemo(() => {
        const grouped: Record<string, { yield_tons: number; profit: number }> = {};

        crops.forEach(crop => {
            const date = new Date(crop.created_at);
            const monthYear = date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });

            if (!grouped[monthYear]) {
                grouped[monthYear] = { yield_tons: 0, profit: 0 };
            }

            // Convert kg to tons
            const yield_tons = crop.crop_produced_kg / 1000;

            // Calculate profit: (selling_price - crop_price) * quantity
            const profit = (crop.selling_price - crop.crop_price) * crop.crop_produced_kg;

            grouped[monthYear].yield_tons += yield_tons;
            grouped[monthYear].profit += profit;
        });

        // Convert to array and sort by date
        return Object.entries(grouped)
            .map(([month, data]) => ({
                month,
                yield_tons: parseFloat(data.yield_tons.toFixed(2)),
                profit: parseFloat(data.profit.toFixed(2)),
            }))
            .sort((a, b) => new Date(a.month).getTime() - new Date(b.month).getTime())
            .slice(-12); // Last 12 months
    }, [crops]);

    const downloadReport = () => {
        const csv = [
            ['Crop Name', 'Cost Price', 'Selling Price', 'Quantity (kg)', 'Profit', 'Date'].join(','),
            ...crops.map(crop => [
                crop.crop_name,
                crop.crop_price,
                crop.selling_price,
                crop.crop_produced_kg,
                ((crop.selling_price - crop.crop_price) * crop.crop_produced_kg).toFixed(2),
                new Date(crop.created_at).toLocaleDateString()
            ].join(','))
        ].join('\\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `crop-history-${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-background p-8">
                <div className="flex items-center justify-center min-h-[400px]">
                    <Loader2 className="h-8 w-8 animate-spin" />
                </div>
            </div>
        );
    }

    if (crops.length === 0) {
        return (
            <div className="min-h-screen bg-background p-8">
                <div className="max-w-4xl mx-auto text-center py-12">
                    <h2 className="text-2xl font-bold mb-4">No Crop Data Yet</h2>
                    <p className="text-muted-foreground mb-6">
                        Add your first crop record to start tracking yield and profit trends
                    </p>
                    <Button onClick={() => navigate('/crop-history/form')} className="bg-green-600 hover:bg-green-700">
                        <Plus className="h-4 w-4 mr-2" />
                        Add First Crop
                    </Button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background p-8">
            <div className="max-w-7xl mx-auto">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-3xl font-bold">Crop History Trends</h1>
                        <p className="text-muted-foreground mt-2">
                            Yield and profit analysis based on your crop records
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <Button onClick={() => navigate('/crop-history/form')} variant="outline">
                            <Edit className="h-4 w-4 mr-2" />
                            Modify Answers
                        </Button>
                        <Button onClick={downloadReport} variant="outline">
                            <Download className="h-4 w-4 mr-2" />
                            Download Report
                        </Button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    {/* Yield Chart */}
                    <Card>
                        <CardHeader>
                            <CardTitle>Yield Trends</CardTitle>
                            <CardDescription>Total crop yield over time (in tons)</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={monthlyData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="month" />
                                    <YAxis />
                                    <Tooltip formatter={(value) => `${value} tons`} />
                                    <Legend />
                                    <Line type="monotone" dataKey="yield_tons" stroke="#22c55e" strokeWidth={2} name="Yield (tons)" />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>

                    {/* Profit Chart */}
                    <Card>
                        <CardHeader>
                            <CardTitle>Profit Trends</CardTitle>
                            <CardDescription>Total profit over time (₹)</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={monthlyData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="month" />
                                    <YAxis />
                                    <Tooltip formatter={(value) => `₹${value.toLocaleString()}`} />
                                    <Legend />
                                    <Line type="monotone" dataKey="profit" stroke="#3b82f6" strokeWidth={2} name="Profit (₹)" />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </div>

                {/* Recent Records Table */}
                <Card>
                    <CardHeader>
                        <CardTitle>Recent Crop Records</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b">
                                        <th className="text-left p-2">Crop</th>
                                        <th className="text-left p-2">Cost (₹/kg)</th>
                                        <th className="text-left p-2">Selling (₹/kg)</th>
                                        <th className="text-left p-2">Quantity (kg)</th>
                                        <th className="text-left p-2">Profit (₹)</th>
                                        <th className="text-left p-2">Date</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {crops.slice(0, 10).map(crop => (
                                        <tr key={crop.id} className="border-b">
                                            <td className="p-2">{crop.crop_name}</td>
                                            <td className="p-2">{crop.crop_price.toFixed(2)}</td>
                                            <td className="p-2">{crop.selling_price.toFixed(2)}</td>
                                            <td className="p-2">{crop.crop_produced_kg.toFixed(2)}</td>
                                            <td className="p-2 text-green-600 font-medium">
                                                ₹{((crop.selling_price - crop.crop_price) * crop.crop_produced_kg).toFixed(2)}
                                            </td>
                                            <td className="p-2 text-muted-foreground">{new Date(crop.created_at).toLocaleDateString()}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
