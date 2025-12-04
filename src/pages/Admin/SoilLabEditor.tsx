import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Loader2, Save, Trash2, ArrowLeft } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { getSoilLabById, createSoilLab, updateSoilLab, deleteSoilLab, type SoilLab } from '@/services/api';

export default function SoilLabEditor() {
    const { id } = useParams();
    const navigate = useNavigate();
    const { toast } = useToast();
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        location: '',
        contact_number: '',
        price: 0,
        rating: 0,
        tag: '',
    });

    const isEdit = Boolean(id);

    useEffect(() => {
        if (isEdit && id) {
            fetchLab();
        }
    }, [id]);

    const fetchLab = async () => {
        try {
            setLoading(true);
            const lab = await getSoilLabById(Number(id));
            setFormData({
                name: lab.name || '',
                location: lab.location || '',
                contact_number: lab.contact_number || '',
                price: lab.price || 0,
                rating: lab.rating || 0,
                tag: lab.tag || '',
            });
        } catch (error) {
            console.error('Failed to fetch lab:', error);
            toast({
                title: 'Error',
                description: 'Failed to load lab',
                variant: 'destructive',
            });
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setSaving(true);
            if (isEdit && id) {
                await updateSoilLab(Number(id), formData);
                toast({
                    title: 'Success',
                    description: 'Soil lab updated successfully',
                });
            } else {
                await createSoilLab(formData);
                toast({
                    title: 'Success',
                    description: 'Soil lab created successfully',
                });
            }
            navigate('/soil-labs');
        } catch (error) {
            console.error('Save error:', error);
            toast({
                title: 'Error',
                description: 'Failed to save lab',
                variant: 'destructive',
            });
        } finally {
            setSaving(false);
        }
    };

    const handleDelete = async () => {
        if (!confirm('Are you sure you want to delete this lab?')) return;

        try {
            setSaving(true);
            await deleteSoilLab(Number(id));
            toast({
                title: 'Success',
                description: 'Soil lab deleted successfully',
            });
            navigate('/soil-labs');
        } catch (error) {
            console.error('Delete error:', error);
            toast({
                title: 'Error',
                description: 'Failed to delete lab',
                variant: 'destructive',
            });
            setSaving(false);
        }
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

    return (
        <div className="min-h-screen bg-background p-8">
            <div className="max-w-3xl mx-auto">
                <Button variant="ghost" onClick={() => navigate('/soil-labs')} className="mb-4">
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back to Soil Labs
                </Button>

                <Card>
                    <CardHeader>
                        <CardTitle>{isEdit ? 'Edit' : 'Create'} Soil Lab</CardTitle>
                        <CardDescription>
                            {isEdit ? 'Update' : 'Add'} soil testing laboratory information
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="name">Lab Name*</Label>
                                <Input
                                    id="name"
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    required
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="location">Location</Label>
                                <Input
                                    id="location"
                                    value={formData.location}
                                    onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="contact_number">Contact Number</Label>
                                <Input
                                    id="contact_number"
                                    value={formData.contact_number}
                                    onChange={(e) => setFormData({ ...formData, contact_number: e.target.value })}
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="price">Price (₹)</Label>
                                    <Input
                                        id="price"
                                        type="number"
                                        min="0"
                                        step="0.01"
                                        value={formData.price}
                                        onChange={(e) => setFormData({ ...formData, price: parseFloat(e.target.value) || 0 })}
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="rating">Rating (0-5)</Label>
                                    <Input
                                        id="rating"
                                        type="number"
                                        min="0"
                                        max="5"
                                        step="0.1"
                                        value={formData.rating}
                                        onChange={(e) => setFormData({ ...formData, rating: parseFloat(e.target.value) || 0 })}
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="tag">Tag</Label>
                                <Input
                                    id="tag"
                                    value={formData.tag}
                                    onChange={(e) => setFormData({ ...formData, tag: e.target.value })}
                                    placeholder="e.g., Government Certified, Fast Service"
                                />
                            </div>

                            <div className="flex gap-2 pt-4">
                                <Button type="submit" disabled={saving} className="flex-1">
                                    {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                                    <Save className="h-4 w-4 mr-2" />
                                    {isEdit ? 'Update' : 'Create'} Lab
                                </Button>
                                {isEdit && (
                                    <Button
                                        type="button"
                                        variant="destructive"
                                        onClick={handleDelete}
                                        disabled={saving}
                                    >
                                        <Trash2 className="h-4 w-4 mr-2" />
                                        Delete
                                    </Button>
                                )}
                            </div>
                        </form>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
