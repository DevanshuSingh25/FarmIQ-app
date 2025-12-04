import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/hooks/use-toast';
import { Search, Edit, Trash2, Loader2, Plus } from 'lucide-react';
import { getAPIBaseURL } from '@/utils/api';

interface Scheme {
    id: number;
    name: string;
    ministry: string;
    deadline: string;
    location: string;
    contact_number: string;
    no_of_docs_required: number;
    status: string;
    benefit_text: string;
    eligibility_text: string;
}

export default function SchemeManagement() {
    const [schemes, setSchemes] = useState<Scheme[]>([]);
    const [filteredSchemes, setFilteredSchemes] = useState<Scheme[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [editScheme, setEditScheme] = useState<Scheme | null>(null);
    const [deleteScheme, setDeleteScheme] = useState<Scheme | null>(null);
    const [isNew, setIsNew] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const { toast } = useToast();
    const API_BASE_URL = getAPIBaseURL();

    useEffect(() => {
        fetchSchemes();
    }, []);

    useEffect(() => {
        let filtered = schemes.filter(
            (s) =>
                s.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                s.ministry?.toLowerCase().includes(searchTerm.toLowerCase())
        );
        if (statusFilter !== 'all') {
            filtered = filtered.filter((s) => s.status === statusFilter);
        }
        setFilteredSchemes(filtered);
    }, [schemes, searchTerm, statusFilter]);

    const fetchSchemes = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/admin/schemes`, { credentials: 'include' });
            if (!response.ok) throw new Error('Failed to fetch schemes');
            const data = await response.json();
            setSchemes(data);
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to load schemes', variant: 'destructive' });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!editScheme) return;
        setSubmitting(true);
        try {
            const url = isNew ? `${API_BASE_URL}/admin/schemes` : `${API_BASE_URL}/admin/schemes/${editScheme.id}`;
            const response = await fetch(url, {
                method: isNew ? 'POST' : 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(editScheme),
            });
            if (!response.ok) throw new Error('Failed to save scheme');
            toast({ title: 'Success', description: `Scheme ${isNew ? 'created' : 'updated'} successfully` });
            setEditScheme(null);
            setIsNew(false);
            fetchSchemes();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to save scheme', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleDelete = async () => {
        if (!deleteScheme) return;
        setSubmitting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/admin/schemes/${deleteScheme.id}`, {
                method: 'DELETE',
                credentials: 'include',
            });
            if (!response.ok) throw new Error('Failed to delete scheme');
            toast({ title: 'Success', description: 'Scheme deleted successfully' });
            setDeleteScheme(null);
            fetchSchemes();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to delete scheme', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleNew = () => {
        setEditScheme({
            id: 0,
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
        setIsNew(true);
    };

    if (loading) {
        return <div className="flex items-center justify-center h-64"><Loader2 className="h-8 w-8 animate-spin text-green-600" /></div>;
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Scheme Management</h2>
                <p className="text-gray-600">Manage government schemes and programs</p>
            </div>

            <Card>
                <CardHeader>
                    <div className="flex justify-between items-center">
                        <CardTitle>Search & Filter</CardTitle>
                        <Button onClick={handleNew}><Plus className="h-4 w-4 mr-2" />Add Scheme</Button>
                    </div>
                </CardHeader>
                <CardContent className="flex gap-4">
                    <div className="flex-1 relative">
                        <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                        <Input placeholder="Search schemes..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="pl-10" />
                    </div>
                    <Select value={statusFilter} onValueChange={setStatusFilter}>
                        <SelectTrigger className="w-48">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Status</SelectItem>
                            <SelectItem value="active">Active</SelectItem>
                            <SelectItem value="inactive">Inactive</SelectItem>
                        </SelectContent>
                    </Select>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Schemes ({filteredSchemes.length})</CardTitle>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>ID</TableHead>
                                <TableHead>Name</TableHead>
                                <TableHead>Ministry</TableHead>
                                <TableHead>Deadline</TableHead>
                                <TableHead>Location</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead className="text-right">Actions</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {filteredSchemes.map((scheme) => (
                                <TableRow key={scheme.id}>
                                    <TableCell>{scheme.id}</TableCell>
                                    <TableCell className="font-medium">{scheme.name}</TableCell>
                                    <TableCell>{scheme.ministry}</TableCell>
                                    <TableCell>{scheme.deadline}</TableCell>
                                    <TableCell>{scheme.location}</TableCell>
                                    <TableCell>
                                        <span className={`px-2 py-1 rounded text-xs font-medium ${scheme.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
                                            }`}>
                                            {scheme.status}
                                        </span>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <Button size="sm" variant="outline" onClick={() => { setEditScheme(scheme); setIsNew(false); }} className="mr-2">
                                            <Edit className="h-4 w-4" />
                                        </Button>
                                        <Button size="sm" variant="destructive" onClick={() => setDeleteScheme(scheme)}>
                                            <Trash2 className="h-4 w-4" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Dialog open={!!editScheme} onOpenChange={() => { setEditScheme(null); setIsNew(false); }}>
                <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                    <DialogHeader>
                        <DialogTitle>{isNew ? 'Add New Scheme' : 'Edit Scheme'}</DialogTitle>
                    </DialogHeader>
                    {editScheme && (
                        <div className="grid grid-cols-2 gap-4">
                            <div className="col-span-2"><Label>Scheme Name</Label><Input value={editScheme.name} onChange={(e) => setEditScheme({ ...editScheme, name: e.target.value })} /></div>
                            <div><Label>Ministry</Label><Input value={editScheme.ministry} onChange={(e) => setEditScheme({ ...editScheme, ministry: e.target.value })} /></div>
                            <div><Label>Deadline</Label><Input type="date" value={editScheme.deadline} onChange={(e) => setEditScheme({ ...editScheme, deadline: e.target.value })} /></div>
                            <div><Label>Location</Label><Input value={editScheme.location} onChange={(e) => setEditScheme({ ...editScheme, location: e.target.value })} /></div>
                            <div><Label>Contact Number</Label><Input value={editScheme.contact_number} onChange={(e) => setEditScheme({ ...editScheme, contact_number: e.target.value })} /></div>
                            <div><Label>Documents Required</Label><Input type="number" value={editScheme.no_of_docs_required} onChange={(e) => setEditScheme({ ...editScheme, no_of_docs_required: parseInt(e.target.value) })} /></div>
                            <div>
                                <Label>Status</Label>
                                <Select value={editScheme.status} onValueChange={(value) => setEditScheme({ ...editScheme, status: value })}>
                                    <SelectTrigger><SelectValue /></SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="active">Active</SelectItem>
                                        <SelectItem value="inactive">Inactive</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="col-span-2"><Label>Benefits</Label><Textarea value={editScheme.benefit_text} onChange={(e) => setEditScheme({ ...editScheme, benefit_text: e.target.value })} rows={3} /></div>
                            <div className="col-span-2"><Label>Eligibility</Label><Textarea value={editScheme.eligibility_text} onChange={(e) => setEditScheme({ ...editScheme, eligibility_text: e.target.value })} rows={3} /></div>
                        </div>
                    )}
                    <DialogFooter>
                        <Button variant="outline" onClick={() => { setEditScheme(null); setIsNew(false); }}>Cancel</Button>
                        <Button onClick={handleSave} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}Save
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog open={!!deleteScheme} onOpenChange={() => setDeleteScheme(null)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete Scheme</DialogTitle>
                        <DialogDescription>Are you sure you want to delete {deleteScheme?.name}?</DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDeleteScheme(null)}>Cancel</Button>
                        <Button variant="destructive" onClick={handleDelete} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}Delete
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
