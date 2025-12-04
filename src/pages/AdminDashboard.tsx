import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft, Shield, Users, FlaskConical, FileText, UserCircle,
  BarChart3, LogOut, Menu, X
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import ProfileManagement from './admin/ProfileManagement';
import LabManagement from './admin/LabManagement';
import SchemeManagement from './admin/SchemeManagement';
import ExpertManagement from './admin/ExpertManagement';

export default function AdminDashboard() {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const navItems = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'profiles', label: 'Profile Management', icon: Users },
    { id: 'labs', label: 'Lab Management', icon: FlaskConical },
    { id: 'schemes', label: 'Scheme Management', icon: FileText },
    { id: 'experts', label: 'Expert Management', icon: UserCircle },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b shadow-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden p-2 hover:bg-gray-100 rounded-lg"
              >
                {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
              <Shield className="h-8 w-8 text-green-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
                <p className="text-sm text-gray-600">Manage FarmIQ Platform</p>
              </div>
            </div>
            <Button onClick={handleLogout} variant="ghost" className="flex items-center space-x-2">
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Logout</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-white border-b shadow-lg">
          <nav className="container mx-auto px-4 py-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveTab(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg mb-1 transition-colors ${activeTab === item.id
                    ? 'bg-green-50 text-green-700 font-medium'
                    : 'text-gray-700 hover:bg-gray-100'
                    }`}
                >
                  <Icon className="h-5 w-5" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      )}

      <div className="container mx-auto px-4 py-6">
        <div className="flex gap-6">
          {/* Sidebar Navigation - Desktop */}
          <aside className="hidden lg:block w-64 flex-shrink-0">
            <Card className="sticky top-24">
              <CardHeader>
                <CardTitle className="text-lg">Navigation</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <nav className="space-y-1 p-2">
                  {navItems.map((item) => {
                    const Icon = item.icon;
                    return (
                      <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${activeTab === item.id
                          ? 'bg-green-50 text-green-700 font-medium'
                          : 'text-gray-700 hover:bg-gray-100'
                          }`}
                      >
                        <Icon className="h-5 w-5" />
                        <span>{item.label}</span>
                      </button>
                    );
                  })}
                </nav>
              </CardContent>
            </Card>
          </aside>

          {/* Main Content Area */}
          <div className="flex-1">
            {activeTab === 'overview' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-3xl font-bold text-gray-900 mb-2">Dashboard Overview</h2>
                  <p className="text-gray-600">Manage all platform data and settings</p>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {navItems.slice(1).map((item) => {
                    const Icon = item.icon;
                    return (
                      <Card
                        key={item.id}
                        className="hover:shadow-lg transition-shadow cursor-pointer"
                        onClick={() => setActiveTab(item.id)}
                      >
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                          <CardTitle className="text-sm font-medium">
                            {item.label}
                          </CardTitle>
                          <Icon className="h-4 w-4 text-green-600" />
                        </CardHeader>
                        <CardContent>
                          <div className="text-2xl font-bold">-</div>
                          <p className="text-xs text-muted-foreground">
                            Click to manage
                          </p>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>

                {/* Quick Actions */}
                <Card>
                  <CardHeader>
                    <CardTitle>Quick Actions</CardTitle>
                    <CardDescription>Common administrative tasks</CardDescription>
                  </CardHeader>
                  <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <Button onClick={() => setActiveTab('profiles')} className="w-full">
                      <Users className="mr-2 h-4 w-4" />
                      Manage Users
                    </Button>
                    <Button onClick={() => setActiveTab('labs')} variant="outline" className="w-full">
                      <FlaskConical className="mr-2 h-4 w-4" />
                      Add Lab
                    </Button>
                    <Button onClick={() => setActiveTab('schemes')} variant="outline" className="w-full">
                      <FileText className="mr-2 h-4 w-4" />
                      Add Scheme
                    </Button>
                    <Button onClick={() => setActiveTab('experts')} variant="outline" className="w-full">
                      <UserCircle className="mr-2 h-4 w-4" />
                      Add Expert
                    </Button>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'profiles' && <ProfileManagement />}
            {activeTab === 'labs' && <LabManagement />}
            {activeTab === 'schemes' && <SchemeManagement />}
            {activeTab === 'experts' && <ExpertManagement />}
          </div>
        </div>
      </div>
    </div>
  );
}
