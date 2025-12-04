import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/hooks/use-toast';
import { Search, Edit, Trash2, Loader2, Plus } from 'lucide-react';
import { getAPIBaseURL } from '@/utils/api';

interface Lab {
    id: number;
    name: string;
    location: string;
    contact_number: string;
    price: number;
    rating: number;
    tag: string;
}

export default function LabManagement() {
    const [labs, setLabs] = useState<Lab[]>([]);
    const [filteredLabs, setFilteredLabs] = useState<Lab[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [editLab, setEditLab] = useState<Lab | null>(null);
    const [deleteLab, setDeleteLab] = useState<Lab | null>(null);
    const [isNew, setIsNew] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const { toast } = useToast();
    const API_BASE_URL = getAPIBaseURL();

    useEffect(() => {
        fetchLabs();
    }, []);

    useEffect(() => {
        const filtered = labs.filter(
            (lab) =>
                lab.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                lab.location?.toLowerCase().includes(searchTerm.toLowerCase())
        );
        setFilteredLabs(filtered);
    }, [labs, searchTerm]);

    const fetchLabs = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/admin/labs`, { credentials: 'include' });
            if (!response.ok) throw new Error('Failed to fetch labs');
            const data = await response.json();
            setLabs(data);
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to load labs', variant: 'destructive' });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!editLab) return;
        setSubmitting(true);
        try {
            const url = isNew ? `${API_BASE_URL}/admin/labs` : `${API_BASE_URL}/admin/labs/${editLab.id}`;
            const response = await fetch(url, {
                method: isNew ? 'POST' : 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(editLab),
            });
            if (!response.ok) throw new Error('Failed to save lab');
            toast({ title: 'Success', description: `Lab ${isNew ? 'created' : 'updated'} successfully` });
            setEditLab(null);
            setIsNew(false);
            fetchLabs();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to save lab', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleDelete = async () => {
        if (!deleteLab) return;
        setSubmitting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/admin/labs/${deleteLab.id}`, {
                method: 'DELETE',
                credentials: 'include',
            });
            if (!response.ok) throw new Error('Failed to delete lab');
            toast({ title: 'Success', description: 'Lab deleted successfully' });
            setDeleteLab(null);
            fetchLabs();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to delete lab', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleNew = () => {
        setEditLab({ id: 0, name: '', location: '', contact_number: '', price: 0, rating: 0, tag: '' });
        setIsNew(true);
    };

    if (loading) {
        return <div className="flex items-center justify-center h-64"><Loader2 className="h-8 w-8 animate-spin text-green-600" /></div>;
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Lab Management</h2>
                <p className="text-gray-600">Manage soil testing laboratories</p>
            </div>

            <Card>
                <CardHeader>
                    <div className="flex justify-between items-center">
                        <CardTitle>Search</CardTitle>
                        <Button onClick={handleNew}><Plus className="h-4 w-4 mr-2" />Add Lab</Button>
                    </div>
                </CardHeader>
                <CardContent>
                    <div className="relative">
                        <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                        <Input placeholder="Search labs..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="pl-10" />
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Soil Labs ({filteredLabs.length})</CardTitle>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>ID</TableHead>
                                <TableHead>Name</TableHead>
                                <TableHead>Location</TableHead>
                                <TableHead>Contact</TableHead>
                                <TableHead>Price</TableHead>
                                <TableHead>Rating</TableHead>
                                <TableHead>Tag</TableHead>
                                <TableHead className="text-right">Actions</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {filteredLabs.map((lab) => (
                                <TableRow key={lab.id}>
                                    <TableCell>{lab.id}</TableCell>
                                    <TableCell className="font-medium">{lab.name}</TableCell>
                                    <TableCell>{lab.location}</TableCell>
                                    <TableCell>{lab.contact_number}</TableCell>
                                    <TableCell>₹{lab.price}</TableCell>
                                    <TableCell>{lab.rating}⭐</TableCell>
                                    <TableCell><span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">{lab.tag}</span></TableCell>
                                    <TableCell className="text-right">
                                        <Button size="sm" variant="outline" onClick={() => { setEditLab(lab); setIsNew(false); }} className="mr-2">
                                            <Edit className="h-4 w-4" />
                                        </Button>
                                        <Button size="sm" variant="destructive" onClick={() => setDeleteLab(lab)}>
                                            <Trash2 className="h-4 w-4" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Dialog open={!!editLab} onOpenChange={() => { setEditLab(null); setIsNew(false); }}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>{isNew ? 'Add New Lab' : 'Edit Lab'}</DialogTitle>
                    </DialogHeader>
                    {editLab && (
                        <div className="grid gap-4">
                            <div><Label>Name</Label><Input value={editLab.name} onChange={(e) => setEditLab({ ...editLab, name: e.target.value })} /></div>
                            <div><Label>Location</Label><Input value={editLab.location} onChange={(e) => setEditLab({ ...editLab, location: e.target.value })} /></div>
                            <div><Label>Contact Number</Label><Input value={editLab.contact_number} onChange={(e) => setEditLab({ ...editLab, contact_number: e.target.value })} /></div>
                            <div><Label>Price</Label><Input type="number" value={editLab.price} onChange={(e) => setEditLab({ ...editLab, price: parseFloat(e.target.value) })} /></div>
                            <div><Label>Rating</Label><Input type="number" step="0.1" max="5" value={editLab.rating} onChange={(e) => setEditLab({ ...editLab, rating: parseFloat(e.target.value) })} /></div>
                            <div><Label>Tag</Label><Input value={editLab.tag} onChange={(e) => setEditLab({ ...editLab, tag: e.target.value })} /></div>
                        </div>
                    )}
                    <DialogFooter>
                        <Button variant="outline" onClick={() => { setEditLab(null); setIsNew(false); }}>Cancel</Button>
                        <Button onClick={handleSave} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}Save
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog open={!!deleteLab} onOpenChange={() => setDeleteLab(null)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete Lab</DialogTitle>
                        <DialogDescription>Are you sure you want to delete {deleteLab?.name}?</DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDeleteLab(null)}>Cancel</Button>
                        <Button variant="destructive" onClick={handleDelete} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}Delete
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
