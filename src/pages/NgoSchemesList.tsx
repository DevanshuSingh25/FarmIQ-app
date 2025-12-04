import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Loader2, Phone, MapPin, Calendar, FileText, Plus } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { getNgoSchemes, type NgoScheme } from '@/services/api';
import { useNavigate } from 'react-router-dom';

export default function NgoSchemesList() {
    const { user } = useAuth();
    const { toast } = useToast();
    const navigate = useNavigate();
    const [schemes, setSchemes] = useState<NgoScheme[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchSchemes();
    }, []);

    const fetchSchemes = async () => {
        try {
            setLoading(true);
            const data = await getNgoSchemes();
            setSchemes(data);
        } catch (error) {
            console.error('Failed to fetch NGO schemes:', error);
            toast({
                title: 'Error',
                description: 'Failed to load NGO schemes',
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
                        <h1 className="text-3xl font-bold">NGO Schemes</h1>
                        <p className="text-muted-foreground mt-2">
                            Browse government and NGO schemes available for farmers
                        </p>
                    </div>
                    {isAdmin && (
                        <Button onClick={() => navigate('/admin/ngo-schemes/new')}>
                            <Plus className="h-4 w-4 mr-2" />
                            Add Scheme
                        </Button>
                    )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {schemes.map((scheme) => (
                        <Card key={scheme.id} className="hover:shadow-lg transition-shadow">
                            <CardHeader>
                                <div className="flex justify-between items-start">
                                    <CardTitle className="text-lg">{scheme.name}</CardTitle>
                                    <Badge variant={scheme.status === 'active' ? 'default' : 'secondary'}>
                                        {scheme.status || 'active'}
                                    </Badge>
                                </div>
                                {scheme.ministry && (
                                    <CardDescription>{scheme.ministry}</CardDescription>
                                )}
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {scheme.location && (
                                    <div className="flex items-center gap-2 text-sm">
                                        <MapPin className="h-4 w-4 text-muted-foreground" />
                                        <span>{scheme.location}</span>
                                    </div>
                                )}

                                {scheme.deadline && (
                                    <div className="flex items-center gap-2 text-sm">
                                        <Calendar className="h-4 w-4 text-muted-foreground" />
                                        <span>Deadline: {scheme.deadline}</span>
                                    </div>
                                )}

                                {scheme.contact_number && (
                                    <div className="flex items-center gap-2 text-sm">
                                        <Phone className="h-4 w-4 text-muted-foreground" />
                                        <span>{scheme.contact_number}</span>
                                    </div>
                                )}

                                {scheme.no_of_docs_required !== undefined && scheme.no_of_docs_required > 0 && (
                                    <div className="flex items-center gap-2 text-sm">
                                        <FileText className="h-4 w-4 text-muted-foreground" />
                                        <span>{scheme.no_of_docs_required} documents required</span>
                                    </div>
                                )}

                                {scheme.benefit_text && (
                                    <div className="mt-3 pt-3 border-t">
                                        <p className="text-sm font-medium mb-1">Benefits:</p>
                                        <p className="text-sm text-muted-foreground">{scheme.benefit_text}</p>
                                    </div>
                                )}

                                {scheme.eligibility_text && (
                                    <div className="mt-2">
                                        <p className="text-sm font-medium mb-1">Eligibility:</p>
                                        <p className="text-sm text-muted-foreground">{scheme.eligibility_text}</p>
                                    </div>
                                )}

                                {isAdmin && (
                                    <div className="mt-4 pt-3 border-t flex gap-2">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => navigate(`/admin/ngo-schemes/${scheme.id}`)}
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

                {schemes.length === 0 && (
                    <div className="text-center py-12">
                        <p className="text-muted-foreground">No NGO schemes available yet.</p>
                        {isAdmin && (
                            <Button onClick={() => navigate('/admin/ngo-schemes/new')} className="mt-4">
                                <Plus className="h-4 w-4 mr-2" />
                                Add First Scheme
                            </Button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
