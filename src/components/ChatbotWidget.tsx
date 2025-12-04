import { useState } from "react";
import { MessageCircle, X, ChevronLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useAuth } from "@/contexts/AuthContext";

interface Question {
  id: string;
  text: string;
  answer: string;
}

interface Feature {
  id: string;
  name: string;
  questions: Question[];
}

const farmerData: Feature[] = [
  {
    id: 'soil',
    name: 'Soil Analysis',
    questions: [
      {
        id: 'how-to-use',
        text: 'How do I use Soil Analysis?',
        answer: 'To use Soil Analysis: 1) Navigate to the Soil Analysis page from the dashboard. 2) Upload a clear image of your soil sample. 3) Our AI will analyze the soil type and provide recommendations. 4) Review the detailed report with soil characteristics and fertilizer suggestions.'
      },
      {
        id: 'what-info',
        text: 'What information will I get?',
        answer: 'You will receive: Soil type classification (Clay, Sandy, Loamy, etc.), Soil health indicators, Nutrient deficiency analysis, Personalized fertilizer recommendations, and Crop suitability suggestions based on your soil type.'
      },
      {
        id: 'image-tips',
        text: 'Tips for best results?',
        answer: 'For accurate results: Take photos in good natural lighting, Ensure the soil surface is clearly visible, Remove any debris or stones, Take close-up shots from multiple angles, and Upload high-quality images (not blurry).'
      }
    ]
  },
  {
    id: 'crop-disease',
    name: 'Crop Disease Detection',
    questions: [
      {
        id: 'how-detect',
        text: 'How does disease detection work?',
        answer: 'Disease Detection steps: 1) Go to Crop Disease Detection page. 2) Take or upload a photo of the affected plant/leaf. 3) Our AI analyzes the image instantly. 4) Get disease identification, severity level, treatment recommendations, and preventive measures.'
      },
      {
        id: 'photo-quality',
        text: 'What makes a good photo?',
        answer: 'Good photo guidelines: Focus on the affected area (spots, discoloration, etc.), Use natural daylight for clear visibility, Keep the camera steady to avoid blur, Fill the frame with the leaf/plant part, and Take multiple angles if symptoms vary.'
      },
      {
        id: 'supported-crops',
        text: 'Which crops are supported?',
        answer: 'Our system supports major crops including: Rice, Wheat, Maize, Cotton, Tomato, Potato, and many more common vegetables and grains. The AI is continuously learning to support more crop varieties.'
      }
    ]
  },
  {
    id: 'weather',
    name: 'Weather Forecast',
    questions: [
      {
        id: 'forecast-days',
        text: 'How many days forecast?',
        answer: 'Weather Forecast provides: 7-day detailed forecast, Hourly predictions for the next 24 hours, Temperature highs and lows, Rainfall predictions, Humidity and wind speed data, and Special agricultural weather alerts.'
      },
      {
        id: 'location-based',
        text: 'Is it location-specific?',
        answer: 'Yes! The weather forecast is location-specific. We use your current location to provide accurate local weather data. You can also search for weather in different areas to plan field activities across multiple farm locations.'
      },
      {
        id: 'farming-tips',
        text: 'Does it give farming advice?',
        answer: 'Absolutely! Based on weather conditions, we provide: Best days for sowing/harvesting, Irrigation scheduling recommendations, Pest risk alerts during humid conditions, Frost warnings for sensitive crops, and Advice on protecting crops during extreme weather.'
      }
    ]
  },
  {
    id: 'market',
    name: 'Market Prices',
    questions: [
      {
        id: 'price-updates',
        text: 'How often are prices updated?',
        answer: 'Market prices are updated daily from government data sources. You can view current prices for various crops across different markets/mandis. Prices include minimum, maximum, and modal (average) rates.'
      },
      {
        id: 'search-crops',
        text: 'How to search for my crop?',
        answer: 'To find prices: 1) Go to Market Prices page. 2) Use the search bar to find your crop. 3) Filter by state and district. 4) View price trends and market rates. 5) Compare prices across different mandis to get the best deal.'
      },
      {
        id: 'export-data',
        text: 'Can I export price data?',
        answer: 'Yes! You can export market price data as CSV files for record-keeping or offline analysis. This helps you track price trends over time and make informed selling decisions.'
      }
    ]
  },
  {
    id: 'schemes',
    name: 'Government Schemes',
    questions: [
      {
        id: 'find-schemes',
        text: 'How to find schemes for me?',
        answer: 'Finding eligible schemes: 1) Visit Government Schemes page. 2) Enter your details (state, land size, crop, category, age). 3) Click "Find My Schemes". 4) View personalized list of schemes you qualify for with eligibility criteria, benefits, and application links.'
      },
      {
        id: 'scheme-types',
        text: 'What types of schemes are listed?',
        answer: 'We list: Central Government schemes (PM-KISAN, etc.), State-specific agricultural schemes, Subsidies for equipment/seeds, Crop insurance programs, Financial assistance schemes, and Special programs for small/marginal farmers.'
      },
      {
        id: 'apply-scheme',
        text: 'How do I apply?',
        answer: 'For each scheme, we provide: Complete eligibility criteria, Required documents list, Official application link, Contact information for help, and Application deadline. Click on "Apply Now" to visit the official government portal.'
      }
    ]
  },
  {
    id: 'iot',
    name: 'IoT Sensor Booking',
    questions: [
      {
        id: 'what-is-iot',
        text: 'What are IoT sensors?',
        answer: 'IoT sensors are smart devices that monitor: Soil moisture levels, Temperature and humidity, Light intensity for crops, Real-time field conditions. They send data to your dashboard for informed decision-making on irrigation and crop care.'
      },
      {
        id: 'how-book',
        text: 'How to book IoT sensors?',
        answer: 'Booking process: 1) Go to IoT Sensor page. 2) Fill in the booking form with your name, phone, location, and preferred visit date. 3) Submit request. 4) Our team will contact you to install sensors. 5) View real-time data once installed.'
      },
      {
        id: 'view-readings',
        text: 'How to view sensor data?',
        answer: 'Once your IoT device is active: Visit the IoT Sensor page, View real-time readings with graphs, Check temperature, humidity, soil moisture, and light levels, Monitor trends over 24 hours, and Get alerts for critical conditions.'
      }
    ]
  },
  {
    id: 'qr',
    name: 'QR Code Generation',
    questions: [
      {
        id: 'why-qr',
        text: 'Why generate a QR code?',
        answer: 'QR codes help you: Share crop details with vendors instantly, Enable quick scanning at mandis/markets, Provide product traceability, Show crop history and quality information, and Facilitate faster transactions without paperwork.'
      },
      {
        id: 'create-qr',
        text: 'How to create a QR code?',
        answer: 'Creating QR code: 1) Go to QR Generation page. 2) Fill in crop details (name, quantity, price, quality grade, harvest date). 3) Add any special notes. 4) Click "Generate QR Code". 5) Download the QR image to share with buyers.'
      },
      {
        id: 'vendor-scan',
        text: 'How do buyers scan it?',
        answer: 'Vendors/buyers can: Scan the QR using their phone camera or QR scanner app, View complete crop information instantly, See your contact details, Check product authenticity, and Make faster purchase decisions with verified data.'
      }
    ]
  },
  {
    id: 'yield',
    name: 'Yield Prediction',
    questions: [
      {
        id: 'predict-yield',
        text: 'How does yield prediction work?',
        answer: 'Yield Prediction uses: Your crop history data, Current weather conditions, Soil health information, Farming practices input, to provide estimated yield for the season. This helps in planning sales and resources.'
      },
      {
        id: 'accuracy',
        text: 'How accurate are predictions?',
        answer: 'Prediction accuracy improves with: More historical data you provide, Regular updates of crop conditions, Accurate input of farming practices, and Local weather data. The AI learns from your farm patterns to give better estimates over time.'
      },
      {
        id: 'use-predictions',
        text: 'How to use yield data?',
        answer: 'Use yield predictions to: Plan harvest timing, Estimate revenue for the season, Arrange transport and storage in advance, Negotiate with buyers beforehand, and Make informed decisions about crop insurance.'
      }
    ]
  },
  {
    id: 'consultancy',
    name: 'Expert Consultancy',
    questions: [
      {
        id: 'find-experts',
        text: 'How to find experts?',
        answer: 'Finding experts: 1) Go to Consultancy page. 2) Browse available agricultural experts with ratings and specializations. 3) Filter by expertise (Soil Health, Pest Management, etc.). 4) Search by name or specialty. 5) View expert profiles with experience and consultation counts.'
      },
      {
        id: 'contact-expert',
        text: 'How to contact an expert?',
        answer: 'Contacting experts: Click "Chat" button to start messaging with an expert about your farming queries, or Click "Call" to initiate a voice call for urgent consultations. Most consultations are free for farmers.'
      },
      {
        id: 'expert-specialties',
        text: 'What can I ask experts?',
        answer: 'Consult experts about: Crop disease diagnosis, Soil health improvement, Pest management strategies, Irrigation planning, Fertilizer recommendations, Crop rotation advice, Organic farming techniques, and Market advisory.'
      }
    ]
  }
];

const vendorData: Feature[] = [
  {
    id: 'qr-scan',
    name: 'QR Code Scanning',
    questions: [
      {
        id: 'how-scan',
        text: 'How to scan farmer QR codes?',
        answer: 'To scan QR codes: 1) Go to QR Scan page from your dashboard. 2) Allow camera access when prompted. 3) Point your camera at the farmer\'s QR code. 4) View complete crop details instantly including farmer info, crop type, quantity, quality grade, and pricing.'
      },
      {
        id: 'upload-qr',
        text: 'Can I upload QR image?',
        answer: 'Yes! If you have a QR code image: 1) Click "Upload Image" button on QR Scan page. 2) Select the image file from your device. 3) The system will automatically decode and display the crop information. This is useful for saved QR codes or images sent via messaging apps.'
      },
      {
        id: 'verify-info',
        text: 'How to verify crop information?',
        answer: 'After scanning: Review farmer contact details, Check crop specifications (type, quantity, grade), Verify harvest date and freshness, Compare with market prices, and Contact the farmer directly through provided phone number for negotiation.'
      }
    ]
  },
  {
    id: 'farmer-search',
    name: 'Farmer Search',
    questions: [
      {
        id: 'search-farmers',
        text: 'How to search for farmers?',
        answer: 'Searching for farmers: 1) Navigate to Farmer Search page. 2) Use the search bar to find farmers by name or location. 3) Browse through the list of registered farmers. 4) View their available crops, quantities, and expected prices. 5) Click on a farmer profile for detailed information.'
      },
      {
        id: 'filter-crops',
        text: 'Can I filter by crop type?',
        answer: 'Yes! You can search and filter farmers by: Crop type they grow, Location/district, Available quantity, Expected price range, and Quality grade. This helps you find farmers with the specific produce you need.'
      },
      {
        id: 'contact-farmers',
        text: 'How to contact farmers?',
        answer: 'Contacting farmers: Each farmer profile shows their phone number and email. You can: Call them directly for immediate discussion, Send inquiry via platform messaging, View their crop availability and prices, and Schedule farm visits for bulk purchases.'
      }
    ]
  },
  {
    id: 'market-prices',
    name: 'Market Prices',
    questions: [
      {
        id: 'check-prices',
        text: 'How to check market prices?',
        answer: 'Checking market prices: 1) Go to Market Prices page. 2) Search for specific crops or commodities. 3) Filter by state and district. 4) View daily updated prices including minimum, maximum, and average rates. 5) Compare across different mandis.'
      },
      {
        id: 'price-trends',
        text: 'Can I see price trends?',
        answer: 'Market prices are updated daily from official sources. You can: View current day prices, Compare across different markets, Check state-wise variations, and Export data for analysis. Use this information to set competitive buying prices.'
      },
      {
        id: 'best-time',
        text: 'When to buy for best prices?',
        answer: 'Use market price data to: Identify seasonal price drops, Compare rates across markets, Plan bulk purchases during surplus, Avoid peak price periods, and Negotiate better deals with farmers based on current market rates.'
      }
    ]
  },
  {
    id: 'dashboard',
    name: 'Dashboard Overview',
    questions: [
      {
        id: 'dashboard-stats',
        text: 'What information is on dashboard?',
        answer: 'Your vendor dashboard shows: Total purchases made, Active orders count, Available crops from farmers, Growth rate metrics, Recent QR scans, and Quick action buttons for scanning, searching, and checking market prices.'
      },
      {
        id: 'track-orders',
        text: 'How to track my orders?',
        answer: 'Track orders: View "Active Orders" count on dashboard, Check pending verifications, Monitor recent QR scans for quality checks, and Keep track of your purchase history and growth metrics.'
      },
      {
        id: 'quick-actions',
        text: 'What are quick actions?',
        answer: 'Quick Actions provide one-click access to: Scan QR codes for instant crop verification, Search for farmers and available crops, Check current market prices, and Access chat support for any queries or assistance.'
      }
    ]
  },
  {
    id: 'profile',
    name: 'Profile Management',
    questions: [
      {
        id: 'update-profile',
        text: 'How to update my profile?',
        answer: 'Updating profile: 1) Click on your profile icon. 2) Go to Profile page. 3) Edit your business name, contact details, and location. 4) Update your language preference. 5) Save changes. Keep your profile updated for farmers to contact you easily.'
      },
      {
        id: 'business-info',
        text: 'What business info should I add?',
        answer: 'Add complete business details: Business/shop name, Contact phone number, Business location/address, Operating area (districts you buy from), and Preferred crop types. This helps farmers find and trust you as a buyer.'
      },
      {
        id: 'language-settings',
        text: 'Can I change language?',
        answer: 'Yes! FarmIQ supports multiple languages: English, Hindi, and Punjabi. Change language from your profile settings or using the globe icon in the navbar. The entire interface will switch to your preferred language.'
      }
    ]
  }
];

type ViewState = 'main' | 'feature' | 'answer';

export const ChatbotWidget = () => {
  const { user } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const [currentView, setCurrentView] = useState<ViewState>('main');
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(null);

  // Determine which data to use based on user role
  const chatbotData = user?.role === 'vendor' ? vendorData : farmerData;
  const roleLabel = user?.role === 'vendor' ? 'Vendor' : 'Farmer';

  const handleFeatureClick = (feature: Feature) => {
    setSelectedFeature(feature);
    setCurrentView('feature');
  };

  const handleQuestionClick = (question: Question) => {
    setSelectedQuestion(question);
    setCurrentView('answer');
  };

  const handleBackToFeatures = () => {
    setSelectedFeature(null);
    setSelectedQuestion(null);
    setCurrentView('main');
  };

  const handleBackToQuestions = () => {
    setSelectedQuestion(null);
    setCurrentView('feature');
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <Button
          onClick={() => setIsOpen(true)}
          size="lg"
          className="h-14 w-14 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 bg-primary hover:bg-primary/90"
          title="Open FarmIQ Help"
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-80 max-w-[calc(100vw-2rem)] sm:w-96">
      <Card className="shadow-xl border-2">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <MessageCircle className="h-5 w-5 text-primary" />
              FarmIQ Help
            </CardTitle>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                setIsOpen(false);
                setCurrentView('main');
                setSelectedFeature(null);
                setSelectedQuestion(null);
              }}
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>

        <CardContent className="p-0">
          <ScrollArea className="h-96 px-4">
            <div className="space-y-4 pb-4">
              {/* Bot Avatar and Greeting */}
              <div className="flex gap-3 justify-start">
                <Avatar className="h-8 w-8 mt-1">
                  <AvatarFallback className="bg-primary text-primary-foreground text-xs">
                    AI
                  </AvatarFallback>
                </Avatar>
                <div className="max-w-[80%] rounded-lg px-3 py-2 text-sm bg-muted">
                  {currentView === 'main' && `Hi! 👋 Welcome to FarmIQ ${roleLabel} Help. What would you like to learn about?`}
                  {currentView === 'feature' && selectedFeature && `Great! What would you like to know about ${selectedFeature.name}?`}
                  {currentView === 'answer' && selectedQuestion && selectedQuestion.answer}
                </div>
              </div>

              {/* Main Menu - Feature Buttons */}
              {currentView === 'main' && (
                <div className="space-y-2 mt-4">
                  {chatbotData.map((feature) => (
                    <Button
                      key={feature.id}
                      variant="outline"
                      className="w-full justify-start text-left h-auto py-3 px-4"
                      onClick={() => handleFeatureClick(feature)}
                    >
                      {feature.name}
                    </Button>
                  ))}
                </div>
              )}

              {/* Feature View - Question Buttons */}
              {currentView === 'feature' && selectedFeature && (
                <div className="space-y-2 mt-4">
                  {selectedFeature.questions.map((question) => (
                    <Button
                      key={question.id}
                      variant="outline"
                      className="w-full justify-start text-left h-auto py-3 px-4 whitespace-normal"
                      onClick={() => handleQuestionClick(question)}
                    >
                      {question.text}
                    </Button>
                  ))}
                  <Button
                    variant="ghost"
                    className="w-full mt-4"
                    onClick={handleBackToFeatures}
                  >
                    <ChevronLeft className="h-4 w-4 mr-2" />
                    Back to Main Menu
                  </Button>
                </div>
              )}

              {/* Answer View - Back Button */}
              {currentView === 'answer' && (
                <div className="space-y-2 mt-4">
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={handleBackToQuestions}
                  >
                    <ChevronLeft className="h-4 w-4 mr-2" />
                    Back to Questions
                  </Button>
                  <Button
                    variant="ghost"
                    className="w-full"
                    onClick={handleBackToFeatures}
                  >
                    <ChevronLeft className="h-4 w-4 mr-2" />
                    Back to Main Menu
                  </Button>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
};