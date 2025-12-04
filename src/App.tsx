import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ChatbotWidget } from "@/components/ChatbotWidget";
import { setupMockTTSAPI } from "@/utils/mockTTSApi";
import { AuthProvider } from "@/contexts/AuthContext";
import { LanguageProvider } from "@/i18n/LanguageContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import ErrorBoundary from "@/components/ErrorBoundary";
import Index from "./pages/Index";
import Login from "./pages/Login";
import Register from "./pages/Register";
import SoilAnalysis from "./pages/SoilAnalysis";
import CropDiseaseDetection from "./pages/CropDiseaseDetection";
import MarketPrices from "./pages/MarketPrices";
import Weather from "./pages/Weather";
import NGOSchemes from "./pages/NGOSchemes";
import Teaching from "./pages/Teaching";
import YieldPrediction from "./pages/YieldPrediction";
import FarmIQ from "./pages/FarmIQ";
import QRGeneration from "./pages/QRGeneration";
import IoTSensor from "./pages/IoTSensor";
import Profile from "./pages/Profile";
import VendorDashboard from "./pages/VendorDashboard";
import VendorQRScan from "./pages/vendor/qr-scan";
import VendorFarmerSearch from "./pages/vendor/farmer-search";
import VendorMarketPrices from "./pages/vendor/market-prices";
import AdminDashboard from "./pages/AdminDashboard";
import Consultancy from "./pages/farmer/Consultancy";
import FarmerForum from "./pages/FarmerForum";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Setup mock TTS API for development
setupMockTTSAPI();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <ErrorBoundary>
          <LanguageProvider>
            <AuthProvider>
              <Routes>
                {/* Authentication Routes */}
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />

                {/* Protected Dashboard Routes */}
                <Route
                  path="/farmer/dashboard"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <FarmIQ />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/vendor/dashboard"
                  element={
                    <ProtectedRoute requiredRole="vendor">
                      <VendorDashboard />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/vendor/qr-scan"
                  element={
                    <ProtectedRoute requiredRole="vendor">
                      <VendorQRScan />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/vendor/farmer-search"
                  element={
                    <ProtectedRoute requiredRole="vendor">
                      <VendorFarmerSearch />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/vendor/market-prices"
                  element={
                    <ProtectedRoute requiredRole="vendor">
                      <VendorMarketPrices />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/admin/dashboard"
                  element={
                    <ProtectedRoute requiredRole="admin">
                      <AdminDashboard />
                    </ProtectedRoute>
                  }
                />

                {/* Protected Farmer Routes */}
                <Route
                  path="/soil-analysis"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <SoilAnalysis />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/crop-disease"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <CropDiseaseDetection />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/weather"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <Weather />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/ngo-schemes"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <NGOSchemes />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/teaching"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <Teaching />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/qr/generate"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <QRGeneration />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/iot"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <IoTSensor />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/consultancy"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <Consultancy />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/farmer/forum"
                  element={
                    <ProtectedRoute requiredRole="farmer">
                      <FarmerForum />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/profile"
                  element={
                    <ProtectedRoute>
                      <Profile />
                    </ProtectedRoute>
                  }
                />

                {/* Public Routes */}
                <Route path="/" element={<Login />} />
                <Route path="/old-homepage" element={<Index />} />
                <Route path="/iot" element={<div className="container mx-auto p-6">IoT page coming soon</div>} />
                <Route path="/market-prices" element={<MarketPrices />} />
                <Route path="/yield-prediction" element={<YieldPrediction />} />

                {/* Catch-all route */}
                <Route path="*" element={<NotFound />} />
              </Routes>
              <ChatbotWidget />
            </AuthProvider>
          </LanguageProvider>
        </ErrorBoundary>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
