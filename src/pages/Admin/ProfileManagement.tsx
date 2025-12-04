import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/hooks/use-toast';
import { Search, Edit, Trash2, Key, Loader2, UserPlus, Eye, EyeOff } from 'lucide-react';
import { getAPIBaseURL } from '@/utils/api';

interface User {
    id: number;
    email: string;
    role: string;
    phone: string;
    full_name: string;
    phone_number: string;
    location: string;
    crops_grown: string;
    available_quantity: number;
    expected_price: string;
    created_at: string;
}

export default function ProfileManagement() {
    const [users, setUsers] = useState<User[]>([]);
    const [filteredUsers, setFilteredUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [roleFilter, setRoleFilter] = useState('all');
    const [editUser, setEditUser] = useState<User | null>(null);
    const [deleteUser, setDeleteUser] = useState<User | null>(null);
    const [resetPasswordUser, setResetPasswordUser] = useState<User | null>(null);
    const [newPassword, setNewPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const { toast } = useToast();
    const API_BASE_URL = getAPIBaseURL();

    useEffect(() => {
        fetchUsers();
    }, []);

    useEffect(() => {
        filterUsers();
    }, [users, searchTerm, roleFilter]);

    const fetchUsers = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/admin/users`, { credentials: 'include' });
            if (!response.ok) throw new Error('Failed to fetch users');
            const data = await response.json();
            setUsers(data);
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to load users', variant: 'destructive' });
        } finally {
            setLoading(false);
        }
    };

    const filterUsers = () => {
        let filtered = users;
        if (searchTerm) {
            filtered = filtered.filter(
                (u) =>
                    u.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                    u.email?.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }
        if (roleFilter !== 'all') {
            filtered = filtered.filter((u) => u.role === roleFilter);
        }
        setFilteredUsers(filtered);
    };

    const handleUpdate = async () => {
        if (!editUser) return;
        setSubmitting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/admin/users/${editUser.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(editUser),
            });
            if (!response.ok) throw new Error('Failed to update user');
            toast({ title: 'Success', description: 'User updated successfully' });
            setEditUser(null);
            fetchUsers();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to update user', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleDelete = async () => {
        if (!deleteUser) return;
        setSubmitting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/admin/users/${deleteUser.id}`, {
                method: 'DELETE',
                credentials: 'include',
            });
            if (!response.ok) throw new Error('Failed to delete user');
            toast({ title: 'Success', description: 'User deleted successfully' });
            setDeleteUser(null);
            fetchUsers();
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to delete user', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    const handleResetPassword = async () => {
        if (!resetPasswordUser || !newPassword) return;
        setSubmitting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/admin/users/${resetPasswordUser.id}/reset-password`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ newPassword }),
            });
            if (!response.ok) throw new Error('Failed to reset password');
            toast({ title: 'Success', description: 'Password reset successfully' });
            setResetPasswordUser(null);
            setNewPassword('');
            setShowPassword(false);
        } catch (error) {
            toast({ title: 'Error', description: 'Failed to reset password', variant: 'destructive' });
        } finally {
            setSubmitting(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="h-8 w-8 animate-spin text-green-600" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Profile Management</h2>
                <p className="text-gray-600">Manage user accounts and profiles</p>
            </div>

            {/* Filters */}
            <Card>
                <CardHeader>
                    <CardTitle>Search & Filter</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col sm:flex-row gap-4">
                    <div className="flex-1">
                        <div className="relative">
                            <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                            <Input
                                placeholder="Search by name or email..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="pl-10"
                            />
                        </div>
                    </div>
                    <Select value={roleFilter} onValueChange={setRoleFilter}>
                        <SelectTrigger className="w-full sm:w-48">
                            <SelectValue placeholder="Filter by role" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Roles</SelectItem>
                            <SelectItem value="farmer">Farmers</SelectItem>
                            <SelectItem value="vendor">Vendors</SelectItem>
                            <SelectItem value="admin">Admins</SelectItem>
                        </SelectContent>
                    </Select>
                </CardContent>
            </Card>

            {/* Users Table */}
            <Card>
                <CardHeader>
                    <div className="flex justify-between items-center">
                        <div>
                            <CardTitle>Users ({filteredUsers.length})</CardTitle>
                            <CardDescription>All registered users on the platform</CardDescription>
                        </div>
                    </div>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>ID</TableHead>
                                    <TableHead>Name</TableHead>
                                    <TableHead>Email</TableHead>
                                    <TableHead>Role</TableHead>
                                    <TableHead>Phone</TableHead>
                                    <TableHead>Location</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {filteredUsers.map((user) => (
                                    <TableRow key={user.id}>
                                        <TableCell className="font-medium">{user.id}</TableCell>
                                        <TableCell>{user.full_name || '-'}</TableCell>
                                        <TableCell>{user.email}</TableCell>
                                        <TableCell>
                                            <span
                                                className={`px-2 py-1 rounded-full text-xs font-medium ${user.role === 'admin'
                                                    ? 'bg-purple-100 text-purple-700'
                                                    : user.role === 'vendor'
                                                        ? 'bg-blue-100 text-blue-700'
                                                        : 'bg-green-100 text-green-700'
                                                    }`}
                                            >
                                                {user.role}
                                            </span>
                                        </TableCell>
                                        <TableCell>{user.phone_number || user.phone || '-'}</TableCell>
                                        <TableCell>{user.location || '-'}</TableCell>
                                        <TableCell className="text-right">
                                            <div className="flex justify-end gap-2">
                                                <Button size="sm" variant="outline" onClick={() => setEditUser(user)}>
                                                    <Edit className="h-4 w-4" />
                                                </Button>
                                                <Button size="sm" variant="outline" onClick={() => setResetPasswordUser(user)}>
                                                    <Key className="h-4 w-4" />
                                                </Button>
                                                <Button
                                                    size="sm"
                                                    variant="destructive"
                                                    onClick={() => setDeleteUser(user)}
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                </CardContent>
            </Card>

            {/* Edit Dialog */}
            <Dialog open={!!editUser} onOpenChange={() => setEditUser(null)}>
                <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                    <DialogHeader>
                        <DialogTitle>Edit User</DialogTitle>
                        <DialogDescription>Update user information and profile details</DialogDescription>
                    </DialogHeader>
                    {editUser && (
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <Label>Full Name</Label>
                                <Input
                                    value={editUser.full_name || ''}
                                    onChange={(e) => setEditUser({ ...editUser, full_name: e.target.value })}
                                />
                            </div>
                            <div>
                                <Label>Email</Label>
                                <Input
                                    type="email"
                                    value={editUser.email}
                                    onChange={(e) => setEditUser({ ...editUser, email: e.target.value })}
                                />
                            </div>
                            <div>
                                <Label>Role</Label>
                                <Select
                                    value={editUser.role}
                                    onValueChange={(value) => setEditUser({ ...editUser, role: value })}
                                >
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="farmer">Farmer</SelectItem>
                                        <SelectItem value="vendor">Vendor</SelectItem>
                                        <SelectItem value="admin">Admin</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            <div>
                                <Label>Phone</Label>
                                <Input
                                    value={editUser.phone_number || editUser.phone || ''}
                                    onChange={(e) => setEditUser({ ...editUser, phone_number: e.target.value, phone: e.target.value })}
                                />
                            </div>
                            <div>
                                <Label>Location</Label>
                                <Input
                                    value={editUser.location || ''}
                                    onChange={(e) => setEditUser({ ...editUser, location: e.target.value })}
                                />
                            </div>
                            <div>
                                <Label>Crops Grown</Label>
                                <Input
                                    value={editUser.crops_grown || ''}
                                    onChange={(e) => setEditUser({ ...editUser, crops_grown: e.target.value })}
                                />
                            </div>
                            <div>
                                <Label>Available Quantity</Label>
                                <Input
                                    type="number"
                                    value={editUser.available_quantity || 0}
                                    onChange={(e) =>
                                        setEditUser({ ...editUser, available_quantity: parseInt(e.target.value) || 0 })
                                    }
                                />
                            </div>
                            <div>
                                <Label>Expected Price</Label>
                                <Input
                                    value={editUser.expected_price || ''}
                                    onChange={(e) => setEditUser({ ...editUser, expected_price: e.target.value })}
                                />
                            </div>
                        </div>
                    )}
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setEditUser(null)}>
                            Cancel
                        </Button>
                        <Button onClick={handleUpdate} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                            Save Changes
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Delete Dialog */}
            <Dialog open={!!deleteUser} onOpenChange={() => setDeleteUser(null)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete User</DialogTitle>
                        <DialogDescription>
                            Are you sure you want to delete {deleteUser?.full_name || deleteUser?.email}? This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDeleteUser(null)}>
                            Cancel
                        </Button>
                        <Button variant="destructive" onClick={handleDelete} disabled={submitting}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                            Delete
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Reset Password Dialog */}
            <Dialog open={!!resetPasswordUser} onOpenChange={() => setResetPasswordUser(null)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Reset Password</DialogTitle>
                        <DialogDescription>
                            Enter a new password for {resetPasswordUser?.full_name || resetPasswordUser?.email}
                        </DialogDescription>
                    </DialogHeader>
                    <div>
                        <Label>New Password</Label>
                        <div className="relative">
                            <Input
                                type={showPassword ? "text" : "password"}
                                placeholder="Enter new password (min 6 characters)"
                                value={newPassword}
                                onChange={(e) => setNewPassword(e.target.value)}
                                className="pr-10"
                            />
                            <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                                onClick={() => setShowPassword(!showPassword)}
                            >
                                {showPassword ? (
                                    <EyeOff className="h-4 w-4 text-gray-400" />
                                ) : (
                                    <Eye className="h-4 w-4 text-gray-400" />
                                )}
                            </Button>
                        </div>
                    </div>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setResetPasswordUser(null)}>
                            Cancel
                        </Button>
                        <Button onClick={handleResetPassword} disabled={submitting || newPassword.length < 6}>
                            {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                            Reset Password
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
