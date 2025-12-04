import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Loader2, Phone, MapPin, DollarSign, Star, Plus } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { getSoilLabs, type SoilLab } from '@/services/api';
import { useNavigate } from 'react-router-dom';

export default function SoilLabsList() {
    const { user } = useAuth();
    const { toast } = useToast();
    const navigate = useNavigate();
    const [labs, setLabs] = useState<SoilLab[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchLabs();
    }, []);

    const fetchLabs = async () => {
        try {
            setLoading(true);
            const data = await getSoilLabs();
            setLabs(data);
        } catch (error) {
            console.error('Failed to fetch soil labs:', error);
            toast({
                title: 'Error',
                description: 'Failed to load soil labs',
                variant: 'destructive',
            });
        } finally {
            setLoading(false);
        }
    };

    const isAdmin = user?.role === 'admin';

    if (loading) {
        return (
            <div className="min-h-screen bg-background p-8">
                <div className="flex items-center justify-center min-h-[400px]">
                    <Loader2 className="h-8 w-8 animate-spin" />
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background p-8">
            <div className="max-w-7xl mx-auto">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-3xl font-bold">Soil Testing Labs</h1>
                        <p className="text-muted-foreground mt-2">
                            Find nearby soil testing laboratories
                        </p>
                    </div>
                    {isAdmin && (
                        <Button onClick={() => navigate('/admin/soil-labs/new')}>
                            <Plus className="h-4 w-4 mr-2" />
                            Add Lab
                        </Button>
                    )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {labs.map((lab) => (
                        <Card key={lab.id} className="hover:shadow-lg transition-shadow">
                            <CardHeader>
                                <div className="flex justify-between items-start">
                                    <CardTitle className="text-lg">{lab.name}</CardTitle>
                                    {lab.tag && (
                                        <Badge variant="secondary">{lab.tag}</Badge>
                                    )}
                                </div>
                                {lab.location && (
                                    <CardDescription className="flex items-center gap-1">
                                        <MapPin className="h-3 w-3" />
                                        {lab.location}
                                    </CardDescription>
                                )}
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {lab.contact_number && (
                                    <div className="flex items-center gap-2 text-sm">
                                        <Phone className="h-4 w-4 text-muted-foreground" />
                                        <span>{lab.contact_number}</span>
                                    </div>
                                )}

                                <div className="flex items-center justify-between pt-2">
                                    {lab.price !== undefined && lab.price > 0 && (
                                        <div className="flex items-center gap-2 text-sm font-medium">
                                            <DollarSign className="h-4 w-4 text-green-600" />
                                            <span>₹{lab.price}</span>
                                        </div>
                                    )}

                                    {lab.rating !== undefined && lab.rating > 0 && (
                                        <div className="flex items-center gap-1 text-sm">
                                            <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                                            <span className="font-medium">{lab.rating.toFixed(1)}</span>
                                        </div>
                                    )}
                                </div>

                                {isAdmin && (
                                    <div className="mt-4 pt-3 border-t flex gap-2">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => navigate(`/admin/soil-labs/${lab.id}`)}
                                            className="flex-1"
                                        >
                                            Edit
                                        </Button>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    ))}
                </div>

                {labs.length === 0 && (
                    <div className="text-center py-12">
                        <p className="text-muted-foreground">No soil labs available yet.</p>
                        {isAdmin && (
                            <Button onClick={() => navigate('/admin/soil-labs/new')} className="mt-4">
                                <Plus className="h-4 w-4 mr-2" />
                                Add First Lab
                            </Button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
