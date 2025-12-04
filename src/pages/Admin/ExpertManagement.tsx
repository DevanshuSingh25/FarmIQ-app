import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/hooks/use-toast';
import { Search, Edit, Trash2, Loader2, Plus, Star } from 'lucide-react';
import { getAPIBaseURL } from '@/utils/api';

interface Expert {
    id: number;
    name: string;
    experience_years: number;
    specializations: string;
    phone_number: string;
    rating: number;
    consultation_count: number;
}

export default function ExpertManagement() {
    const [experts, setExperts] = useState<Expert[]>([]);
    const [filteredExperts, setFilteredExperts] = useState<Expert[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [editExpert, setEditExpert] = useState<Expert | null>(null);
    const [deleteExpert, setDeleteExpert] = useState<Expert | null>(null);
    const [isNew, setIsNew] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const { toast } = useToast();
    const API_BASE_URL = getAPIBaseURL();

    useEffect(() => {
        fetchExperts();
    }, []);

    useEffect(() => {
        const filtered = experts.filter(
            (e) =>
                e.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                e.specializations?.toLowerCase().includes(searchTerm.toLowerCase())
        );
        setFilteredExperts(filtered);
    }, [experts, searchTerm]);

    const fetchExperts = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/admin/experts`, { credentials: 'include' });
            if (!response.ok) throw new Error('Failed to fetch experts');
            const data = await response.json();
            setExperts(data);
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to load experts', variant: 'destructive' });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!editExpert) return;
        setSubmitting(true);
        try {
            const url = isNew ? `${API_BASE_URL}/admin/experts` : `${API_BASE_URL}/admin/experts/${editExpert.id}`;
            const response = await fetch(url, {
                method: isNew ? 'POST' : 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(editExpert),
            });
            if (!response.ok) throw new Error('Failed to save expert');
            toast({ title: 'Success', description: `Expert ${isNew ? 'created' : 'updated'} successfully` });
            setEditExpert(null);
            setIsNew(false);
            fetchExperts();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to save expert', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleDelete = async () => {
        if (!deleteExpert) return;
        setSubmitting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/admin/experts/${deleteExpert.id}`, {
                method: 'DELETE',
                credentials: 'include',
            });
            if (!response.ok) throw new Error('Failed to delete expert');
            toast({ title: 'Success', description: 'Expert deleted successfully' });
            setDeleteExpert(null);
            fetchExperts();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to delete expert', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleNew = () => {
        setEditExpert({ id: 0, name: '', experience_years: 0, specializations: '', phone_number: '', rating: 0, consultation_count: 0 });
        setIsNew(true);
    };

    if (loading) {
        return <div className="flex items-center justify-center h-64"><Loader2 className="h-8 w-8 animate-spin text-green-600" /></div>;
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Expert Management</h2>
                <p className="text-gray-600">Manage agricultural experts and consultants</p>
            </div>

            <Card>
                <CardHeader>
                    <div className="flex justify-between items-center">
                        <CardTitle>Search</CardTitle>
                        <Button onClick={handleNew}><Plus className="h-4 w-4 mr-2" />Add Expert</Button>
                    </div>
                </CardHeader>
                <CardContent>
                    <div className="relative">
                        <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                        <Input placeholder="Search experts..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="pl-10" />
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Experts ({filteredExperts.length})</CardTitle>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>ID</TableHead>
                                <TableHead>Name</TableHead>
                                <TableHead>Experience</TableHead>
                                <TableHead>Specializations</TableHead>
                                <TableHead>Phone</TableHead>
                                <TableHead>Rating</TableHead>
                                <TableHead>Consultations</TableHead>
                                <TableHead className="text-right">Actions</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {filteredExperts.map((expert) => (
                                <TableRow key={expert.id}>
                                    <TableCell>{expert.id}</TableCell>
                                    <TableCell className="font-medium">{expert.name}</TableCell>
                                    <TableCell>{expert.experience_years} years</TableCell>
                                    <TableCell className="max-w-xs truncate">{expert.specializations}</TableCell>
                                    <TableCell>{expert.phone_number}</TableCell>
                                    <TableCell>
                                        <div className="flex items-center">
                                            {expert.rating} <Star className="h-4 w-4 ml-1 text-yellow-500 fill-yellow-500" />
                                        </div>
                                    </TableCell>
                                    <TableCell>{expert.consultation_count}</TableCell>
                                    <TableCell className="text-right">
                                        <Button size="sm" variant="outline" onClick={() => { setEditExpert(expert); setIsNew(false); }} className="mr-2">
                                            <Edit className="h-4 w-4" />
                                        </Button>
                                        <Button size="sm" variant="destructive" onClick={() => setDeleteExpert(expert)}>
                                            <Trash2 className="h-4 w-4" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Dialog open={!!editExpert} onOpenChange={() => { setEditExpert(null); setIsNew(false); }}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>{isNew ? 'Add New Expert' : 'Edit Expert'}</DialogTitle>
                    </DialogHeader>
                    {editExpert && (
                        <div className="grid gap-4">
                            <div><Label>Name</Label><Input value={editExpert.name} onChange={(e) => setEditExpert({ ...editExpert, name: e.target.value })} /></div>
                            <div><Label>Experience (years)</Label><Input type="number" value={editExpert.experience_years} onChange={(e) => setEditExpert({ ...editExpert, experience_years: parseInt(e.target.value) })} /></div>
                            <div><Label>Specializations</Label><Input value={editExpert.specializations} onChange={(e) => setEditExpert({ ...editExpert, specializations: e.target.value })} placeholder="e.g., Soil Health, Pest Management" /></div>
                            <div><Label>Phone Number</Label><Input value={editExpert.phone_number} onChange={(e) => setEditExpert({ ...editExpert, phone_number: e.target.value })} /></div>
                            <div><Label>Rating</Label><Input type="number" step="0.1" max="5" value={editExpert.rating} onChange={(e) => setEditExpert({ ...editExpert, rating: parseFloat(e.target.value) })} /></div>
                        </div>
                    )}
                    <DialogFooter>
                        <Button variant="outline" onClick={() => { setEditExpert(null); setIsNew(false); }}>Cancel</Button>
                        <Button onClick={handleSave} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}Save
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog open={!!deleteExpert} onOpenChange={() => setDeleteExpert(null)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete Expert</DialogTitle>
                        <DialogDescription>Are you sure you want to delete {deleteExpert?.name}?</DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDeleteExpert(null)}>Cancel</Button>
                        <Button variant="destructive" onClick={handleDelete} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}Delete
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
