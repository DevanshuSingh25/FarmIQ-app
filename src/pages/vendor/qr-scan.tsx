import { useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import {
    Menu,
    Globe,
    Moon,
    User,
    Leaf,
    Camera,
    Image as ImageIcon,
    LogOut,
    Info,
    UserCircle,
    LayoutDashboard,
    Search,
    BarChart3,
    ChevronRight,
    QrCode
} from 'lucide-react';
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
import { useToast } from "@/hooks/use-toast";
import jsQR from 'jsqr';

export default function VendorQRScan() {
    const navigate = useNavigate();
    const { logout } = useAuth();
    const { toast } = useToast();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [theme, setTheme] = useState<'light' | 'dark'>('light');
    const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');
    const languages = ['English', 'Hindi', 'Punjabi'] as const;
    const [qrResult, setQrResult] = useState<string>('');
    const [scanning, setScanning] = useState(false);

    const toggleTheme = () => {
        setTheme(prev => prev === 'light' ? 'dark' : 'light');
        document.documentElement.classList.toggle('dark');
    };

    const handleQRTextSubmit = async (text: string) => {
        try {
            setScanning(true);
            const response = await fetch('http://localhost:3001/api/qr/parse', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ qr_text: text })
            });

            if (!response.ok) {
                throw new Error('Failed to parse QR code');
            }

            const data = await response.json();
            setQrResult(data.qr_text);

            toast({
                title: "QR Code Scanned",
                description: "Product information displayed",
            });
        } catch (error) {
            console.error('QR parse error:', error);
            toast({
                title: "Scan Failed",
                description: "Could not process QR code",
                variant: "destructive",
            });
        } finally {
            setScanning(false);
        }
    };

    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setScanning(true);
        const reader = new FileReader();

        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Create canvas to process image
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                if (!ctx) {
                    toast({
                        title: "Error",
                        description: "Could not process image",
                        variant: "destructive",
                    });
                    setScanning(false);
                    return;
                }

                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                // Get image data
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                // Decode QR code
                const code = jsQR(imageData.data, imageData.width, imageData.height);

                if (code) {
                    // QR code found
                    handleQRTextSubmit(code.data);
                } else {
                    // No QR code found
                    setQrResult('QR not present');
                    toast({
                        title: "No QR Code",
                        description: "QR not present in the uploaded image",
                        variant: "destructive",
                    });
                    setScanning(false);
                }
            };

            img.onerror = () => {
                toast({
                    title: "Error",
                    description: "Could not load image",
                    variant: "destructive",
                });
                setScanning(false);
            };

            img.src = e.target?.result as string;
        };

        reader.onerror = () => {
            toast({
                title: "Error",
                description: "Could not read file",
                variant: "destructive",
            });
            setScanning(false);
        };

        reader.readAsDataURL(file);

        // Reset file input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleManualInput = () => {
        const text = prompt('Enter QR code text:');
        if (text) {
            handleQRTextSubmit(text);
        }
    };

    const formatQRResult = (text: string) => {
        try {
            const parsed = JSON.parse(text);
            return JSON.stringify(parsed, null, 2);
        } catch {
            return text;
        }
    };

    return (
        <div className="min-h-screen bg-[#F8F9FA] font-sans text-gray-900">
            {/* Top Navigation Bar */}
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

                                        <Link to="/vendor/farmer-search" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <Search className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                                                <span className="font-medium text-sm">Farmer Search</span>
                                            </div>
                                            <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-green-700 transition-colors" />
                                        </Link>

                                        <Link to="/vendor/market-prices" className="flex items-center justify-between px-4 py-3 bg-white text-gray-700 hover:bg-green-50 hover:text-green-900 rounded-full transition-all group shadow-sm border border-transparent hover:border-green-100">
                                            <div className="flex items-center gap-3">
                                                <BarChart3 className="h-5 w-5 text-gray-500 group-hover:text-green-700 transition-colors" />
                                                <span className="font-medium text-sm">Market Price</span>
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
                            <Link to="/vendor/qr-scan" className="text-gray-900 font-medium border-b-2 border-green-500 px-1 py-5">
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
                                            onClick={() => setLanguage(lang)}
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
                                onClick={toggleTheme}
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

            {/* Main Content */}
            <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900">QR Code Scanner</h1>
                </div>

                {/* Two Main Sections Side-by-Side */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* LEFT CARD: Scan QR Code */}
                    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8 h-[500px] flex flex-col">
                        <h2 className="text-lg font-bold text-gray-900 mb-6">Scan QR Code</h2>

                        <div className="flex-1 border-2 border-dashed border-gray-200 rounded-xl flex flex-col items-center justify-center p-8 bg-gray-50/50">
                            <div className="mb-6 opacity-20">
                                <QrCode className="h-24 w-24 text-gray-900" />
                            </div>

                            <p className="text-gray-500 text-center mb-8">
                                Scan QR code to verify product authenticity
                            </p>

                            <div className="flex gap-4 w-full max-w-md justify-center">
                                <button
                                    onClick={handleManualInput}
                                    disabled={scanning}
                                    className="flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors min-w-[160px] disabled:opacity-50"
                                >
                                    <Camera className="h-5 w-5" />
                                    <span>Scan QR</span>
                                </button>

                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    onChange={handleImageUpload}
                                    className="hidden"
                                />

                                <button
                                    onClick={() => fileInputRef.current?.click()}
                                    disabled={scanning}
                                    className="flex items-center justify-center gap-2 px-6 py-3 bg-white border border-green-600 text-green-600 hover:bg-green-50 font-medium rounded-lg transition-colors min-w-[160px] disabled:opacity-50"
                                >
                                    <ImageIcon className="h-5 w-5" />
                                    <span>Upload Image</span>
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* RIGHT CARD: Product Information */}
                    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8 h-[500px] flex flex-col">
                        <h2 className="text-lg font-bold text-gray-900 mb-6">Product Information</h2>

                        <div className="flex-1 flex flex-col p-4 bg-gray-50 rounded-lg overflow-auto">
                            {qrResult ? (
                                <pre className="text-sm text-gray-900 whitespace-pre-wrap font-mono">
                                    {formatQRResult(qrResult)}
                                </pre>
                            ) : (
                                <div className="flex-1 flex flex-col items-center justify-center">
                                    <div className="mb-6 opacity-20">
                                        <QrCode className="h-24 w-24 text-gray-900" />
                                    </div>
                                    <p className="text-gray-400 text-center">
                                        Scan a QR code to view product details
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
