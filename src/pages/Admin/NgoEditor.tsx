import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Save, Trash2, ArrowLeft } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { getNgoSchemeById, createNgoScheme, updateNgoScheme, deleteNgoScheme, type NgoScheme } from '@/services/api';

export default function NgoEditor() {
    const { id } = useParams();
    const navigate = useNavigate();
    const { toast } = useToast();
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        ministry: '',
        deadline: '',
        location: '',
        contact_number: '',
        no_of_docs_required: 0,
        status: 'active',
        benefit_text: '',
        eligibility_text: '',
    });

    const isEdit = Boolean(id);

    useEffect(() => {
        if (isEdit && id) {
            fetchScheme();
        }
    }, [id]);

    const fetchScheme = async () => {
        try {
            setLoading(true);
            const scheme = await getNgoSchemeById(Number(id));
            setFormData({
                name: scheme.name || '',
                ministry: scheme.ministry || '',
                deadline: scheme.deadline || '',
                location: scheme.location || '',
                contact_number: scheme.contact_number || '',
                no_of_docs_required: scheme.no_of_docs_required || 0,
                status: scheme.status || 'active',
                benefit_text: scheme.benefit_text || '',
                eligibility_text: scheme.eligibility_text || '',
            });
        } catch (error) {
            console.error('Failed to fetch scheme:', error);
            toast({
                title: 'Error',
                description: 'Failed to load scheme',
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
                await updateNgoScheme(Number(id), formData);
                toast({
                    title: 'Success',
                    description: 'NGO scheme updated successfully',
                });
            } else {
                await createNgoScheme(formData);
                toast({
                    title: 'Success',
                    description: 'NGO scheme created successfully',
                });
            }
            navigate('/ngo-schemes');
        } catch (error) {
            console.error('Save error:', error);
            toast({
                title: 'Error',
                description: 'Failed to save scheme',
                variant: 'destructive',
            });
        } finally {
            setSaving(false);
        }
    };

    const handleDelete = async () => {
        if (!confirm('Are you sure you want to delete this scheme?')) return;

        try {
            setSaving(true);
            await deleteNgoScheme(Number(id));
            toast({
                title: 'Success',
                description: 'NGO scheme deleted successfully',
            });
            navigate('/ngo-schemes');
        } catch (error) {
            console.error('Delete error:', error);
            toast({
                title: 'Error',
                description: 'Failed to delete scheme',
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
                <Button variant="ghost" onClick={() => navigate('/ngo-schemes')} className="mb-4">
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back to NGO Schemes
                </Button>

                <Card>
                    <CardHeader>
                        <CardTitle>{isEdit ? 'Edit' : 'Create'} NGO Scheme</CardTitle>
                        <CardDescription>
                            {isEdit ? 'Update' : 'Add'} scheme information for farmers
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="name">Scheme Name*</Label>
                                <Input
                                    id="name"
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    required
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="ministry">Ministry/Department</Label>
                                <Input
                                    id="ministry"
                                    value={formData.ministry}
                                    onChange={(e) => setFormData({ ...formData, ministry: e.target.value })}
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="deadline">Deadline</Label>
                                    <Input
                                        id="deadline"
                                        type="date"
                                        value={formData.deadline}
                                        onChange={(e) => setFormData({ ...formData, deadline: e.target.value })}
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="status">Status</Label>
                                    <select
                                        id="status"
                                        value={formData.status}
                                        onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                                        className="w-full p-2 border rounded-md"
                                    >
                                        <option value="active">Active</option>
                                        <option value="closed">Closed</option>
                                        <option value="upcoming">Upcoming</option>
                                    </select>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="location">Location</Label>
                                <Input
                                    id="location"
                                    value={formData.location}
                                    onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="contact_number">Contact Number</Label>
                                    <Input
                                        id="contact_number"
                                        value={formData.contact_number}
                                        onChange={(e) => setFormData({ ...formData, contact_number: e.target.value })}
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="no_of_docs_required">Documents Required</Label>
                                    <Input
                                        id="no_of_docs_required"
                                        type="number"
                                        min="0"
                                        value={formData.no_of_docs_required}
                                        onChange={(e) => setFormData({ ...formData, no_of_docs_required: parseInt(e.target.value) || 0 })}
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="benefit_text">Benefits</Label>
                                <Textarea
                                    id="benefit_text"
                                    rows={3}
                                    value={formData.benefit_text}
                                    onChange={(e) => setFormData({ ...formData, benefit_text: e.target.value })}
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="eligibility_text">Eligibility Criteria</Label>
                                <Textarea
                                    id="eligibility_text"
                                    rows={3}
                                    value={formData.eligibility_text}
                                    onChange={(e) => setFormData({ ...formData, eligibility_text: e.target.value })}
                                />
                            </div>

                            <div className="flex gap-2 pt-4">
                                <Button type="submit" disabled={saving} className="flex-1">
                                    {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                                    <Save className="h-4 w-4 mr-2" />
                                    {isEdit ? 'Update' : 'Create'} Scheme
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
