import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Phone, MessageSquare, Star, Search, User, X, Send, PhoneOff, Award, Users } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";

interface Expert {
    id: number;
    name: string;
    experience_years: number;
    specializations: string[];
    rating: number;
    consultation_count: number;
    phone_number: string;
}

const Consultancy = () => {
    const [theme, setTheme] = useState<'light' | 'dark'>('light');
    const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');
    const [experts, setExperts] = useState<Expert[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [specializationFilter, setSpecializationFilter] = useState('All');
    const [availableSpecializations, setAvailableSpecializations] = useState<string[]>([]);

    // Modal states
    const [chatExpert, setChatExpert] = useState<Expert | null>(null);
    const [callExpert, setCallExpert] = useState<Expert | null>(null);
    const [chatMessage, setChatMessage] = useState('');
    const [chatHistory, setChatHistory] = useState<{ sender: 'user' | 'expert', text: string }[]>([]);

    const toggleTheme = () => {
        setTheme(prev => prev === 'light' ? 'dark' : 'light');
        document.documentElement.classList.toggle('dark');
    };

    const fetchExperts = async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams();
            if (searchTerm) params.append('q', searchTerm);
            if (specializationFilter && specializationFilter !== 'All') params.append('specialization', specializationFilter);

            const res = await fetch(`/api/experts?${params.toString()}`);
            if (!res.ok) throw new Error('Failed to load experts');
            const json = await res.json();
            setExperts(json.data);

            if (availableSpecializations.length === 0) {
                const specs = new Set<string>();
                json.data.forEach((e: Expert) => {
                    e.specializations.forEach(s => specs.add(s));
                });
                setAvailableSpecializations(Array.from(specs));
            }
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchExperts();
    }, [searchTerm, specializationFilter]);

    // Chat handlers
    const openChat = (expert: Expert) => {
        setChatExpert(expert);
        setChatHistory([{ sender: 'expert', text: `Namaste! I am ${expert.name}. How can I help you today?` }]);
        setChatMessage('');
    };

    const sendChatMessage = () => {
        if (!chatMessage.trim()) return;

        const newHistory = [...chatHistory, { sender: 'user' as const, text: chatMessage }];
        setChatHistory(newHistory);
        setChatMessage('');

        // Simulate expert response
        setTimeout(() => {
            setChatHistory(prev => [...prev, {
                sender: 'expert',
                text: 'Thank you for your query. Let me analyze that for you.'
            }]);
        }, 1000);
    };

    // Call handlers
    const openCall = (expert: Expert) => {
        setCallExpert(expert);
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <FarmIQNavbar
                theme={theme}
                language={language}
                onThemeToggle={toggleTheme}
                onLanguageChange={setLanguage}
            />

            <div className="container mx-auto p-6 space-y-8 pt-20">
                <div className="space-y-2">
                    <h1 className="text-2xl font-bold text-gray-800">Experts Consultancy</h1>
                    <p className="text-gray-500">Get personalized advice from top agricultural experts.</p>
                </div>

                <div className="flex flex-col md:flex-row gap-4 bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                        <Input
                            placeholder="Search experts by name or expertise..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="pl-10 bg-gray-50 border-gray-200"
                        />
                    </div>
                    <Select value={specializationFilter} onValueChange={setSpecializationFilter}>
                        <SelectTrigger className="w-full md:w-[200px] bg-gray-50 border-gray-200">
                            <SelectValue placeholder="All Expertise" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="All">All Expertise</SelectItem>
                            {availableSpecializations.map(spec => (
                                <SelectItem key={spec} value={spec}>{spec}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                <div className="space-y-4">
                    <h2 className="text-lg font-semibold text-gray-700">Available Experts</h2>
                    <p className="text-sm text-gray-500">Choose the right specialist for your farming needs.</p>

                    {loading ? (
                        <div className="flex justify-center p-12">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
                        </div>
                    ) : experts.length === 0 ? (
                        <div className="text-center p-12 text-gray-500 bg-white rounded-lg border border-dashed border-gray-300">
                            No experts found matching your criteria.
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {experts.map((expert) => (
                                <Card key={expert.id} className="border border-gray-100 shadow-sm hover:shadow-md transition-all duration-200 bg-white">
                                    <CardContent className="p-6">
                                        <div className="flex justify-between items-start mb-4">
                                            <div className="flex gap-4">
                                                <div className="h-12 w-12 rounded-full bg-green-100 flex items-center justify-center text-green-700 font-bold text-lg">
                                                    {expert.name.split(' ').map(n => n[0]).join('').substring(0, 2)}
                                                </div>
                                                <div>
                                                    <h3 className="font-bold text-gray-800">{expert.name}</h3>
                                                    <p className="text-xs text-gray-500">{expert.experience_years}+ years Experience</p>
                                                    <div className="flex items-center mt-1 text-xs text-green-600 font-medium">
                                                        <Award className="w-3 h-3 mr-1" />
                                                        Specialist
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="flex items-center bg-yellow-50 text-yellow-700 px-2 py-1 rounded text-xs font-bold">
                                                <Star className="w-3 h-3 mr-1 fill-current" />
                                                {expert.rating.toFixed(1)}
                                            </div>
                                        </div>

                                        <div className="flex flex-wrap gap-2 mb-4">
                                            {expert.specializations.map((spec, idx) => (
                                                <Badge key={idx} variant="secondary" className="bg-orange-50 text-orange-700 hover:bg-orange-100 border-none font-normal text-xs px-2 py-1">
                                                    {spec}
                                                </Badge>
                                            ))}
                                        </div>

                                        <div className="flex items-center text-xs text-gray-400 mb-6">
                                            <Users className="w-3 h-3 mr-1" />
                                            {expert.consultation_count}+ consultations
                                        </div>

                                        <div className="flex gap-3">
                                            <Button
                                                className="flex-1 bg-green-600 hover:bg-green-700 text-white shadow-none"
                                                onClick={() => openChat(expert)}
                                            >
                                                <MessageSquare className="w-4 h-4 mr-2" />
                                                Chat
                                            </Button>
                                            <Button
                                                className="flex-1 border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                                                variant="outline"
                                                onClick={() => openCall(expert)}
                                            >
                                                <Phone className="w-4 h-4 mr-2" />
                                                Call
                                            </Button>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    )}
                </div>

                {/* Chat Modal */}
                <Dialog open={!!chatExpert} onOpenChange={(open) => !open && setChatExpert(null)}>
                    <DialogContent className="sm:max-w-[425px] p-0 gap-0 overflow-hidden">
                        <div className="p-4 border-b flex justify-between items-center bg-gray-50">
                            <div className="flex items-center gap-3">
                                <div className="h-10 w-10 rounded-full bg-green-100 flex items-center justify-center text-green-700 font-bold text-sm">
                                    {chatExpert?.name.split(' ').map(n => n[0]).join('').substring(0, 2)}
                                </div>
                                <div>
                                    <h4 className="font-semibold text-sm">{chatExpert?.name}</h4>
                                    <div className="flex items-center text-xs text-green-600">
                                        <span className="h-1.5 w-1.5 rounded-full bg-green-500 mr-1"></span>
                                        Online
                                    </div>
                                </div>
                            </div>
                            {/* Close button handled by Dialog primitive usually, but we can add custom if needed or rely on default X */}
                        </div>

                        <div className="h-[300px] overflow-y-auto p-4 space-y-4 bg-white">
                            {chatHistory.map((msg, idx) => (
                                <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm ${msg.sender === 'user'
                                        ? 'bg-green-50 text-gray-800 rounded-tr-none'
                                        : 'bg-gray-100 text-gray-800 rounded-tl-none'
                                        }`}>
                                        {msg.text}
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="p-4 border-t bg-white">
                            <div className="flex gap-2">
                                <Input
                                    placeholder="Type a message..."
                                    value={chatMessage}
                                    onChange={(e) => setChatMessage(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                                    className="flex-1"
                                />
                                <Button size="icon" className="bg-green-600 hover:bg-green-700" onClick={sendChatMessage}>
                                    <Send className="h-4 w-4" />
                                </Button>
                            </div>
                        </div>
                    </DialogContent>
                </Dialog>

                {/* Call Modal */}
                <Dialog open={!!callExpert} onOpenChange={(open) => !open && setCallExpert(null)}>
                    <DialogContent className="sm:max-w-[350px] text-center p-6">
                        <DialogHeader>
                            <DialogTitle className="text-center text-lg font-semibold">Start Voice Call</DialogTitle>
                        </DialogHeader>
                        <div className="py-6 flex flex-col items-center justify-center space-y-6">
                            <p className="text-sm text-gray-500">Consulting with {callExpert?.name}</p>

                            <div className="relative">
                                <div className="h-24 w-24 rounded-full bg-green-50 flex items-center justify-center animate-pulse">
                                    <div className="h-20 w-20 rounded-full bg-green-100 flex items-center justify-center">
                                        <Phone className="h-10 w-10 text-green-600" />
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-1">
                                <h3 className="font-medium text-gray-900">Free Consultation</h3>
                                <p className="text-xs text-green-600">Connecting to secure line...</p>
                            </div>

                            <Button
                                className="w-full bg-red-500 hover:bg-red-600 text-white"
                                onClick={() => setCallExpert(null)}
                            >
                                End Call
                            </Button>
                        </div>
                    </DialogContent>
                </Dialog>
            </div>
        </div>
    );
};

export default Consultancy;
