import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { SectionSpeaker } from "@/components/ui/section-speaker";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { Edit, Save, X, Camera, User, Phone, Loader2, Mail, Menu, Globe, Moon, Leaf, LogOut, Info, UserCircle, LayoutDashboard, Search as SearchIcon, BarChart3, ChevronRight, QrCode } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { authService, ProfileData } from "@/services/authService";
import { Link } from "react-router-dom";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const VendorNavbar = ({ theme, onThemeToggle, language, onLanguageChange }: any) => {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const languages = ['English', 'Hindi', 'Punjabi'] as const;

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Left Side */}
          <div className="flex items-center gap-4">
            <Sheet>
              <SheetTrigger asChild>
                <button className="p-2.5 bg-white rounded-full shadow-sm hover:shadow-md transition-all border border-gray-100 group">
                  <Menu className="h-5 w-5 text-gray-700 group-hover:text-gray-900" />
                </button>
              </SheetTrigger>
              <SheetContent side="left" className="w-[300px] sm:w-[340px] p-6 bg-[#F8F9FA]">
                <SheetHeader className="mb-8 flex flex-row items-center justify-between space-y-0">
                  <SheetTitle className="text-xl font-bold text-gray-900">Navigation</SheetTitle>
                </SheetHeader>
                <div className="space-y-3">
                  <Link to="/vendor/dashboard" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                    <div className="flex items-center gap-3">
                      <LayoutDashboard className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                      <span className="font-medium text-sm">Dashboard</span>
                    </div>
                    <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                  </Link>
                  <Link to="/vendor/qr-scan" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                    <div className="flex items-center gap-3">
                      <QrCode className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                      <span className="font-medium text-sm">QR Scan</span>
                    </div>
                    <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                  </Link>
                  <Link to="/vendor/market-prices" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                    <div className="flex items-center gap-3">
                      <BarChart3 className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                      <span className="font-medium text-sm">Market Prices</span>
                    </div>
                    <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                  </Link>
                  <Link to="/vendor/farmer-search" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                    <div className="flex items-center gap-3">
                      <SearchIcon className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                      <span className="font-medium text-sm">Farmer Search</span>
                    </div>
                    <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                  </Link>
                </div>
              </SheetContent>
            </Sheet>
            <div className="flex items-center gap-2">
              <Leaf className="h-6 w-6 text-green-600" fill="currentColor" />
              <span className="text-xl font-bold text-green-600 tracking-tight">FarmIQ</span>
              <span className="text-gray-400 text-lg font-light">|</span>
              <span className="text-gray-500 text-sm font-medium">Vendor dashboard</span>
            </div>
          </div>

          {/* Center Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link to="/vendor/dashboard" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
              Dashboard
            </Link>
            <Link to="/vendor/qr-scan" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
              QR Scan
            </Link>
            <Link to="/vendor/market-prices" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
              Market Prices
            </Link>
            <Link to="/vendor/farmer-search" className="text-gray-500 hover:text-gray-900 font-medium px-1 py-5 transition-colors">
              Farmer Search
            </Link>
          </div>

          {/* Right Side Icons */}
          <div className="flex items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors">
                  <Globe className="h-5 w-5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-white border border-gray-200 shadow-md">
                {languages.map((lang) => (
                  <DropdownMenuItem
                    key={lang}
                    onClick={() => onLanguageChange(lang)}
                    className="cursor-pointer hover:bg-gray-50"
                  >
                    {lang}
                    {language === lang && <span className="ml-2 text-green-600">✓</span>}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            <button
              className="p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors"
              onClick={onThemeToggle}
            >
              <Moon className="h-5 w-5" />
            </button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors">
                  <User className="h-5 w-5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuItem
                  className="cursor-pointer"
                  onClick={() => navigate('/profile')}
                >
                  <UserCircle className="mr-2 h-4 w-4" />
                  <span>Profile</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer"
                  onClick={() => navigate('/farmer/teaching')}
                >
                  <Info className="mr-2 h-4 w-4" />
                  <span>Know about the website</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer text-red-600 focus:text-red-600"
                  onClick={() => {
                    logout();
                    navigate('/login');
                  }}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Logout</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </nav>
  );
};

const Profile = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { user } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [profileImage, setProfileImage] = useState<string | null>(null);

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

  const [profileData, setProfileData] = useState<ProfileData | null>(null);
  const [editData, setEditData] = useState<Partial<ProfileData>>({});

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark');
  };

  // Fetch profile data from backend
  useEffect(() => {
    const fetchProfile = async () => {
      if (!user) {
        navigate('/login');
        return;
      }

      try {
        setIsLoading(true);
        const profile = await authService.getProfile();
        setProfileData(profile);
        setEditData(profile);
      } catch (error) {
        console.error('Failed to fetch profile:', error);
        toast({
          title: "Error",
          description: "Failed to load profile data",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchProfile();
  }, [user, navigate, toast]);

  const handleEdit = () => {
    setEditData(profileData || {});
    setIsEditing(true);
  };

  const handleSave = async () => {
    try {
      setIsSaving(true);
      // Only send allowed fields: full_name, phone_number, language_pref
      await authService.updateProfile({
        full_name: editData.full_name,
        phone_number: editData.phone_number,
        language_pref: editData.language_pref
      });

      // Refetch profile data
      const updatedProfile = await authService.getProfile();
      setProfileData(updatedProfile);
      setIsEditing(false);

      toast({
        title: "Profile Updated",
        description: "Your profile has been successfully updated.",
      });
    } catch (error) {
      console.error('Failed to update profile:', error);
      toast({
        title: "Error",
        description: "Failed to update profile",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    setEditData(profileData || {});
    setIsEditing(false);
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setProfileImage(e.target?.result as string);
        toast({
          title: "Photo Updated",
          description: "Your profile photo has been updated.",
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const getInitials = (name?: string) => {
    if (!name) return "U";
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const getText = () => "Farmer Profile. View and edit your personal information including name, phone number, and email.";

  if (isLoading || !profileData) {
    return (
      <div className="min-h-screen bg-background">
        {user?.role === 'vendor' ? (
          <VendorNavbar
            theme={theme}
            language={language}
            onThemeToggle={toggleTheme}
            onLanguageChange={setLanguage}
          />
        ) : (
          <FarmIQNavbar
            theme={theme}
            language={language}
            onThemeToggle={toggleTheme}
            onLanguageChange={setLanguage}
          />
        )}
        <div className="container mx-auto px-4 py-8 pt-24">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
              <p className="text-muted-foreground">Loading profile...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {user?.role === 'vendor' ? (
        <VendorNavbar
          theme={theme}
          language={language}
          onThemeToggle={toggleTheme}
          onLanguageChange={setLanguage}
        />
      ) : (
        <FarmIQNavbar
          theme={theme}
          language={language}
          onThemeToggle={toggleTheme}
          onLanguageChange={setLanguage}
        />
      )}

      <div className="container mx-auto px-4 py-8 pt-24">
        <div className="flex items-center gap-4 mb-8 relative group">
          <div className="absolute top-0 right-0 z-10">
            <SectionSpeaker
              getText={getText}
              sectionId="profile-page"
              ariaLabel="Read profile page information"
              alwaysVisible
            />
          </div>
        </div>

        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl lg:text-4xl font-bold mb-4">
              {user?.role === 'farmer' ? 'Farmer' :
                user?.role === 'vendor' ? 'Vendor' :
                  user?.role === 'admin' ? 'Admin' : 'User'} Profile
            </h1>
            <p className="text-xl text-muted-foreground">
              Manage your personal information
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="h-5 w-5" />
                Personal Information
              </CardTitle>
              <CardDescription>
                Your basic profile information and contact details
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Profile Photo Section */}
              <div className="flex flex-col items-center space-y-4">
                <div className="relative">
                  <Avatar className="h-24 w-24">
                    <AvatarImage src={profileImage || undefined} alt="Profile" />
                    <AvatarFallback className="text-2xl bg-primary text-primary-foreground">
                      {getInitials(profileData.full_name)}
                    </AvatarFallback>
                  </Avatar>
                  <label htmlFor="profile-photo" className="absolute -bottom-2 -right-2 cursor-pointer">
                    <div className="bg-primary text-primary-foreground rounded-full p-2 hover:bg-primary/90 transition-colors">
                      <Camera className="h-4 w-4" />
                    </div>
                  </label>
                  <input
                    id="profile-photo"
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </div>
                <p className="text-sm text-muted-foreground text-center">
                  Click the camera icon to upload a profile photo
                </p>
              </div>

              {/* Form Fields (NO aadhar, username, village, district, state) */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="full_name" className="flex items-center gap-2">
                    <User className="h-4 w-4" />
                    Full Name
                  </Label>
                  {isEditing ? (
                    <Input
                      id="full_name"
                      value={editData.full_name || ''}
                      onChange={(e) => setEditData({ ...editData, full_name: e.target.value })}
                      placeholder="Enter your full name"
                    />
                  ) : (
                    <div className="p-3 bg-muted rounded-md">
                      <p className="font-medium">{profileData.full_name || 'Not provided'}</p>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email" className="flex items-center gap-2">
                    <Mail className="h-4 w-4" />
                    Email
                  </Label>
                  <div className="p-3 bg-muted rounded-md">
                    <p className="font-medium">{profileData.email || 'Not provided'}</p>
                    <p className="text-xs text-muted-foreground mt-1">Email cannot be changed</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="phone" className="flex items-center gap-2">
                    <Phone className="h-4 w-4" />
                    Phone Number
                  </Label>
                  {isEditing ? (
                    <Input
                      id="phone"
                      value={editData.phone_number || ''}
                      onChange={(e) => setEditData({ ...editData, phone_number: e.target.value })}
                      placeholder="Enter your phone number"
                    />
                  ) : (
                    <div className="p-3 bg-muted rounded-md">
                      <p className="font-medium">{profileData.phone_number || 'Not provided'}</p>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="language_pref">Language Preference</Label>
                  {isEditing ? (
                    <select
                      id="language_pref"
                      value={editData.language_pref || 'en'}
                      onChange={(e) => setEditData({ ...editData, language_pref: e.target.value })}
                      className="w-full p-3 border rounded-md"
                    >
                      <option value="en">English</option>
                      <option value="hi">Hindi</option>
                      <option value="pa">Punjabi</option>
                    </select>
                  ) : (
                    <div className="p-3 bg-muted rounded-md">
                      <p className="font-medium capitalize">
                        {profileData.language_pref === 'hi' ? 'Hindi' :
                          profileData.language_pref === 'pa' ? 'Punjabi' : 'English'}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4">
                {isEditing ? (
                  <>
                    <Button onClick={handleSave} className="flex-1" disabled={isSaving}>
                      {isSaving ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
                      Save Changes
                    </Button>
                    <Button variant="outline" onClick={handleCancel} className="flex-1" disabled={isSaving}>
                      <X className="h-4 w-4 mr-2" />
                      Cancel
                    </Button>
                  </>
                ) : (
                  <Button onClick={handleEdit} className="w-full">
                    <Edit className="h-4 w-4 mr-2" />
                    Edit Profile
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Additional Information Card */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Account Information</CardTitle>
              <CardDescription>
                Additional details about your account
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-muted-foreground">User ID</Label>
                  <p className="text-sm font-mono">{profileData.id || 'N/A'}</p>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-muted-foreground">Role</Label>
                  <p className="text-sm capitalize">{user?.role || 'N/A'}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Profile;
