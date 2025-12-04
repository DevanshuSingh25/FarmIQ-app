import React, { useState, useEffect } from "react";
import { FarmIQNavbar } from "@/components/farmiq/FarmIQNavbar";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
    MessageSquare,
    Send,
    User,
    Clock,
    ThumbsUp,
    Paperclip,
    Bookmark,
    ChevronDown,
    ChevronUp,
    Filter,
    Search,
    Image as ImageIcon,
    X,
    CheckCircle,
    AlertCircle
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { formatDistanceToNow } from "date-fns";

// --- Mock Data Interfaces (kept for type compatibility) ---
interface Reply {
    id: number;
    name: string;
    role: 'user' | 'expert' | 'verified';
    reply: string;
    createdAt: string; // ISO string or relative time
}

interface Post {
    id: number;
    name: string;
    question: string;
    category: string;
    createdAt: string;
    upvotes: number;
    isUpvoted: boolean;
    isBookmarked: boolean;
    replies: Reply[];
    image?: string;
}

export default function FarmerForum() {
    // --- State ---
    const [theme, setTheme] = useState<'light' | 'dark'>('light');
    const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>('English');

    // Data State (now from database instead of mock)
    const [posts, setPosts] = useState<Post[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    // Input State
    const [newQuestion, setNewQuestion] = useState("");
    const [selectedCategory, setSelectedCategory] = useState("");
    // Disabled for now: const [mockImageAttached, setMockImageAttached] = useState(false);
    // Disabled for now: const [showSuggestions, setShowSuggestions] = useState(false);

    // Filter & Sort State
    const [activeFilter, setActiveFilter] = useState("All");
    const [sortBy, setSortBy] = useState("latest"); // latest, upvoted, replied

    // Reply Input State
    const [replyText, setReplyText] = useState<{ [key: number]: string }>({});

    // Collapsed Replies State
    const [expandedPosts, setExpandedPosts] = useState<{ [key: number]: boolean }>({});

    const { user } = useAuth();
    const [currentUserName, setCurrentUserName] = useState("Farmer");

    // API Base URL helper
    const getAPIBaseURL = () => {
        return 'http://localhost:3001/api';
    };

    useEffect(() => {
        fetchUserProfile();
        fetchForumPosts();
    }, []);

    const fetchUserProfile = async () => {
        try {
            const API_BASE_URL = getAPIBaseURL();

            const response = await fetch(`${API_BASE_URL}/me/profile`, {
                credentials: 'include',
            });
            if (response.ok) {
                const data = await response.json();
                if (data.full_name) {
                    setCurrentUserName(data.full_name);
                }
            }
        } catch (error) {
            console.error("Error fetching profile:", error);
        }
    };


    // Fetch forum posts from database
    const fetchForumPosts = async () => {
        try {
            setIsLoading(true);
            const API_BASE_URL = getAPIBaseURL();

            console.log('Fetching from:', `${API_BASE_URL}/forum/posts`);
            const response = await fetch(`${API_BASE_URL}/forum/posts`, {
                credentials: 'include',
            });

            if (response.ok) {
                const dbPosts = await response.json();
                console.log('Received posts from database:', dbPosts.length);

                // Map database posts to UI format
                const transformedPosts: Post[] = dbPosts.map((dbPost: any) => ({
                    id: dbPost.id,
                    name: dbPost.user_name || 'Anonymous Farmer',
                    question: dbPost.question,
                    category: dbPost.category || dbPost.community,
                    createdAt: dbPost.created_at,
                    upvotes: dbPost.upvotes || 0,
                    isUpvoted: false,
                    isBookmarked: false,
                    replies: dbPost.replies ? dbPost.replies.map((reply: any) => ({
                        id: reply.id,
                        name: reply.replied_by || 'Expert',
                        role: (reply.replied_by && (reply.replied_by.includes('Expert') || reply.replied_by.includes('Advisor'))) ? 'expert' as const :
                            (reply.replied_by && (reply.replied_by.includes('Officer') || reply.replied_by.includes('Scientist'))) ? 'verified' as const : 'user' as const,
                        reply: reply.reply_text,
                        createdAt: reply.created_at
                    })) : []
                }));

                setPosts(transformedPosts);
                console.log(`✓ Loaded ${transformedPosts.length} posts from database`);
            } else {
                const errorText = await response.text();
                console.error(`Failed to fetch: ${response.status} - ${errorText}`);
                setPosts([]); // Set empty array on error, no mock data
            }
        } catch (error) {
            console.error("Error fetching forum posts:", error);
            setPosts([]); // Set empty array on error, no mock data
        } finally {
            setIsLoading(false);
        }
    };

    const categories = [
        "Crop",
        "Soil",
        "Weather",
        "Disease & Pests",
        "Market",
        "Fertilizers",
        "General Queries"
    ];

    const toggleTheme = () => {
        setTheme(prev => prev === 'light' ? 'dark' : 'light');
        document.documentElement.classList.toggle('dark');
    };

    // --- Handlers ---

    const handlePostQuery = async () => {
        if (!newQuestion.trim() || !selectedCategory) return;

        try {
            const API_BASE_URL = getAPIBaseURL();

            // Post the question to database
            const response = await fetch(`${API_BASE_URL}/forum`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    question: newQuestion,
                    category: selectedCategory
                })
            });

            if (response.ok) {
                const data = await response.json();
                console.log('✓ Question posted:', data);

                // If backend returned the complete post with answer, add it immediately to UI
                if (data.post) {
                    const newPost: Post = {
                        id: data.post.id,
                        name: data.post.user_name || 'You',
                        question: data.post.question,
                        category: data.post.community || data.post.category,
                        createdAt: data.post.created_at,
                        upvotes: data.post.upvotes || 0,
                        isUpvoted: false,
                        isBookmarked: false,
                        replies: data.post.replies ? data.post.replies.map((reply: any) => ({
                            id: reply.id,
                            name: reply.replied_by,
                            role: reply.replied_by.includes('Expert') || reply.replied_by.includes('Advisor') ? 'expert' as const :
                                reply.replied_by.includes('Officer') || reply.replied_by.includes('Scientist') ? 'verified' as const : 'user' as const,
                            reply: reply.reply_text,
                            createdAt: reply.created_at
                        })) : []
                    };

                    // Add new post at the beginning (most recent first)
                    setPosts(prev => [newPost, ...prev]);
                    console.log('✓ New post added to UI immediately');
                }

                // Clear inputs
                setNewQuestion("");
                setSelectedCategory("");

                // Refresh all posts from database to ensure consistency
                await fetchForumPosts();
            } else {
                console.error('Failed to post question');
            }
        } catch (error) {
            console.error('Error posting question:', error);
        }
    };

    const handleReply = async (postId: number) => {
        const text = replyText[postId];
        if (!text?.trim()) return;

        const newReply: Reply = {
            id: Date.now(),
            name: currentUserName || "You",
            role: "user",
            reply: text,
            createdAt: new Date().toISOString()
        };

        // Optimistic UI update
        setPosts(posts.map(p =>
            p.id === postId ? { ...p, replies: [...p.replies, newReply] } : p
        ));

        setReplyText(prev => ({ ...prev, [postId]: "" }));
        // Auto expand replies when replying
        setExpandedPosts(prev => ({ ...prev, [postId]: true }));

        // Persist to database (just increment count, actual reply not stored in simple version)
        try {
            const API_BASE_URL = getAPIBaseURL();
            const response = await fetch(`${API_BASE_URL}/forum/reply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ postId, replyText: text, repliedBy: currentUserName })
            });

            if (!response.ok) {
                console.error('Failed to persist reply to database');
            }
        } catch (error) {
            console.error('Error persisting reply:', error);
        }
    };

    const toggleUpvote = async (postId: number) => {
        const post = posts.find(p => p.id === postId);
        if (!post) return;

        const action = post.isUpvoted ? 'decrement' : 'increment';
        const newUpvoteCount = post.isUpvoted ? post.upvotes - 1 : post.upvotes + 1;
        const newIsUpvoted = !post.isUpvoted;

        // Optimistic UI update
        setPosts(posts.map(p => {
            if (p.id === postId) {
                return {
                    ...p,
                    upvotes: newUpvoteCount,
                    isUpvoted: newIsUpvoted
                };
            }
            return p;
        }));

        // Persist to database
        try {
            const API_BASE_URL = getAPIBaseURL();
            const response = await fetch(`${API_BASE_URL}/forum/upvote`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ postId, action })
            });

            if (response.ok) {
                const data = await response.json();
                // Update with actual count from server
                setPosts(posts.map(p => {
                    if (p.id === postId) {
                        return {
                            ...p,
                            upvotes: data.upvotes,
                            isUpvoted: newIsUpvoted
                        };
                    }
                    return p;
                }));
            } else {
                console.error('Failed to persist upvote to database');
                // Revert on error
                setPosts(posts.map(p => {
                    if (p.id === postId) {
                        return {
                            ...p,
                            upvotes: post.upvotes,
                            isUpvoted: post.isUpvoted
                        };
                    }
                    return p;
                }));
            }
        } catch (error) {
            console.error('Error persisting upvote:', error);
            // Revert on error
            setPosts(posts.map(p => {
                if (p.id === postId) {
                    return {
                        ...p,
                        upvotes: post.upvotes,
                        isUpvoted: post.isUpvoted
                    };
                }
                return p;
            }));
        }
    };

    const toggleBookmark = (postId: number) => {
        setPosts(posts.map(p =>
            p.id === postId ? { ...p, isBookmarked: !p.isBookmarked } : p
        ));
    };

    const toggleExpandReplies = (postId: number) => {
        setExpandedPosts(prev => ({ ...prev, [postId]: !prev[postId] }));
    };

    // --- Derived Data (Filtering & Sorting) ---

    const filteredPosts = posts.filter(post => {
        if (activeFilter === "All") return true;
        return post.category === activeFilter;
    });

    const sortedPosts = [...filteredPosts].sort((a, b) => {
        if (sortBy === "latest") {
            return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
        } else if (sortBy === "upvoted") {
            return b.upvotes - a.upvotes;
        } else if (sortBy === "replied") {
            return b.replies.length - a.replies.length;
        }
        return 0;
    });

    return (
        <div className="min-h-screen bg-background font-sans">
            <FarmIQNavbar
                theme={theme}
                language={language}
                onThemeToggle={toggleTheme}
                onLanguageChange={setLanguage}
            />

            <main className="pt-24 pb-8 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto">
                <div className="mb-8">
                    <h1 className="text-2xl font-bold text-foreground mb-2">Farmer Forum</h1>
                    <p className="text-muted-foreground">Connect with other farmers and experts.</p>
                </div>

                {/* Query Input Section */}
                <Card className="mb-8 shadow-sm border-border">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-lg font-medium">Ask a Question</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4 relative">
                            <Textarea
                                placeholder="Ask your question..."
                                className="min-h-[100px] resize-none text-sm"
                                value={newQuestion}
                                onChange={(e) => setNewQuestion(e.target.value)}
                            // Disabled for now: onFocus={() => setShowSuggestions(true)}
                            // Disabled for now: onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                            />

                            {/* Suggested Questions - Disabled for now
                            {showSuggestions && !newQuestion && (
                                <div className="absolute top-[110px] left-0 right-0 z-10 bg-popover border border-border rounded-md shadow-md p-2 mx-1">
                                    <p className="text-xs text-muted-foreground mb-2 px-2">Suggested Questions:</p>
                                    {SUGGESTED_QUESTIONS.map((q, i) => (
                                        <button
                                            key={i}
                                            className="block w-full text-left text-sm px-2 py-1.5 hover:bg-accent hover:text-accent-foreground rounded-sm transition-colors"
                                            onClick={() => setNewQuestion(q)}
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                            )}
                            */}

                            {/* Mock Image Preview - Disabled for now
                            {mockImageAttached && (
                                <div className="flex items-center gap-2 p-2 bg-muted/30 rounded-md border border-dashed border-muted-foreground/30 w-fit">
                                    <div className="h-8 w-8 bg-gray-200 rounded flex items-center justify-center">
                                        <ImageIcon className="h-4 w-4 text-gray-500" />
                                    </div>
                                    <span className="text-xs text-muted-foreground">image_preview.jpg</span>
                                    <button onClick={() => setMockImageAttached(false)} className="ml-2 text-muted-foreground hover:text-destructive">
                                        <X className="h-3 w-3" />
                                    </button>
                                </div>
                            )}
                            */}

                            <div className="flex flex-col sm:flex-row gap-3 justify-between items-center">
                                <div className="flex items-center gap-2 w-full sm:w-auto">
                                    <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                                        <SelectTrigger className="w-full sm:w-[200px]">
                                            <SelectValue placeholder="Select community" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {categories.map((cat) => (
                                                <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>

                                    {/* Attachment button - Disabled for now
                                    <Button
                                        variant="outline"
                                        size="icon"
                                        className="shrink-0"
                                        onClick={() => setMockImageAttached(true)}
                                        title="Attach Image"
                                    >
                                        <Paperclip className="h-4 w-4" />
                                    </Button>
                                    */}
                                </div>

                                <Button
                                    onClick={handlePostQuery}
                                    className="w-full sm:w-auto bg-green-600 hover:bg-green-700 text-white"
                                    disabled={!newQuestion.trim() || !selectedCategory}
                                >
                                    Post Query
                                </Button>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Filters & Sorting */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
                    {/* Category Filters */}
                    <div className="flex flex-wrap gap-2">
                        <Button
                            variant={activeFilter === "All" ? "default" : "outline"}
                            size="sm"
                            onClick={() => setActiveFilter("All")}
                            className="rounded-full text-xs h-7"
                        >
                            All
                        </Button>
                        {categories.slice(0, 5).map(cat => (
                            <Button
                                key={cat}
                                variant={activeFilter === cat ? "default" : "outline"}
                                size="sm"
                                onClick={() => setActiveFilter(cat)}
                                className="rounded-full text-xs h-7"
                            >
                                {cat}
                            </Button>
                        ))}
                    </div>

                    {/* Sorting */}
                    <Select value={sortBy} onValueChange={setSortBy}>
                        <SelectTrigger className="w-[160px] h-8 text-xs">
                            <div className="flex items-center gap-2">
                                <Filter className="h-3 w-3" />
                                <SelectValue placeholder="Sort by" />
                            </div>
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="latest">Latest</SelectItem>
                            <SelectItem value="upvoted">Most Upvoted</SelectItem>
                            <SelectItem value="replied">Most Replied</SelectItem>
                        </SelectContent>
                    </Select>
                </div>

                {/* Posts List */}
                <div className="space-y-6">
                    {isLoading ? (
                        <div className="text-center py-12 bg-muted/10 rounded-xl border border-dashed border-border">
                            <div className="h-10 w-10 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                            <p className="text-muted-foreground">Loading forum posts...</p>
                        </div>
                    ) : sortedPosts.length === 0 ? (
                        <div className="text-center py-12 bg-muted/10 rounded-xl border border-dashed border-border">
                            <MessageSquare className="h-10 w-10 text-muted-foreground mx-auto mb-3 opacity-50" />
                            <p className="text-muted-foreground">No discussions found in this category.</p>
                        </div>
                    ) : (
                        sortedPosts.map((post) => (
                            <Card key={post.id} className="shadow-sm border-border overflow-hidden transition-all hover:shadow-md">
                                <CardHeader className="bg-muted/30 pb-3 pt-4">
                                    <div className="flex justify-between items-start gap-4">
                                        <div className="flex items-center gap-3">
                                            <div className="h-9 w-9 rounded-full bg-green-100 flex items-center justify-center text-green-700 border border-green-200">
                                                <User className="h-5 w-5" />
                                            </div>
                                            <div>
                                                <h3 className="text-sm font-semibold text-foreground">{post.name}</h3>
                                                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                                    <Clock className="h-3 w-3" />
                                                    <span>
                                                        {(() => {
                                                            try {
                                                                return formatDistanceToNow(new Date(post.createdAt), { addSuffix: true });
                                                            } catch (e) {
                                                                return post.createdAt;
                                                            }
                                                        })()}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {post.replies.length > 0 ? (
                                                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200 font-normal text-[10px] flex items-center gap-1">
                                                    <CheckCircle className="h-3 w-3" /> Answered
                                                </Badge>
                                            ) : (
                                                <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200 font-normal text-[10px] flex items-center gap-1">
                                                    <AlertCircle className="h-3 w-3" /> Unanswered
                                                </Badge>
                                            )}
                                            <Badge variant="secondary" className="font-normal text-xs">
                                                {post.category}
                                            </Badge>
                                            <button
                                                onClick={() => toggleBookmark(post.id)}
                                                className={`p-1 rounded-full hover:bg-muted transition-colors ${post.isBookmarked ? "text-primary fill-primary" : "text-muted-foreground"}`}
                                            >
                                                <Bookmark className={`h-4 w-4 ${post.isBookmarked ? "fill-current" : ""}`} />
                                            </button>
                                        </div>
                                    </div>
                                </CardHeader>
                                <CardContent className="pt-4 pb-4">
                                    <p className="text-foreground text-sm leading-relaxed mb-4 font-medium">
                                        {post.question}
                                    </p>

                                    {/* Mock Image Attachment Display */}
                                    {post.image && (
                                        <div className="mb-4 rounded-lg bg-muted/20 border border-border p-8 flex items-center justify-center">
                                            <div className="text-center">
                                                <ImageIcon className="h-8 w-8 text-muted-foreground mx-auto mb-2 opacity-50" />
                                                <span className="text-xs text-muted-foreground">Attached Image</span>
                                            </div>
                                        </div>
                                    )}

                                    {/* Action Bar */}
                                    <div className="flex items-center gap-4 mb-4 border-b border-border pb-4">
                                        <button
                                            onClick={() => toggleUpvote(post.id)}
                                            className={`flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded-md transition-colors ${post.isUpvoted ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted"}`}
                                        >
                                            <ThumbsUp className={`h-3.5 w-3.5 ${post.isUpvoted ? "fill-current" : ""}`} />
                                            {post.upvotes} Upvotes
                                        </button>
                                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground px-2 py-1">
                                            <MessageSquare className="h-3.5 w-3.5" />
                                            {post.replies.length} Replies
                                        </div>
                                    </div>

                                    {/* Replies Section */}
                                    {post.replies.length > 0 && (
                                        <div className="space-y-4 pl-4 border-l-2 border-muted mb-4">
                                            {/* Show only first 2 replies unless expanded */}
                                            {(expandedPosts[post.id] ? post.replies : post.replies.slice(0, 2)).map((reply) => (
                                                <div key={reply.id} className="bg-muted/30 rounded-lg p-3">
                                                    <div className="flex justify-between items-start mb-1">
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-xs font-semibold text-foreground">
                                                                {reply.name}
                                                            </span>
                                                            {reply.role === 'expert' && (
                                                                <Badge variant="default" className="h-4 px-1 text-[9px] bg-blue-600 hover:bg-blue-700">Expert</Badge>
                                                            )}
                                                            {reply.role === 'verified' && (
                                                                <Badge variant="secondary" className="h-4 px-1 text-[9px] bg-green-100 text-green-800 hover:bg-green-200">Verified</Badge>
                                                            )}
                                                        </div>
                                                        <span className="text-[10px] text-muted-foreground">
                                                            {(() => {
                                                                try {
                                                                    return formatDistanceToNow(new Date(reply.createdAt), { addSuffix: true });
                                                                } catch (e) {
                                                                    return reply.createdAt;
                                                                }
                                                            })()}
                                                        </span>
                                                    </div>
                                                    <p className="text-sm text-foreground/90">{reply.reply}</p>
                                                </div>
                                            ))}

                                            {/* Expand/Collapse Button */}
                                            {post.replies.length > 2 && (
                                                <button
                                                    onClick={() => toggleExpandReplies(post.id)}
                                                    className="text-xs text-primary font-medium hover:underline flex items-center gap-1"
                                                >
                                                    {expandedPosts[post.id] ? (
                                                        <>Show less <ChevronUp className="h-3 w-3" /></>
                                                    ) : (
                                                        <>View all {post.replies.length} replies <ChevronDown className="h-3 w-3" /></>
                                                    )}
                                                </button>
                                            )}
                                        </div>
                                    )}

                                    {/* Reply Input */}
                                    <div className="flex gap-2">
                                        <Input
                                            placeholder="Write a reply..."
                                            className="h-9 text-sm"
                                            value={replyText[post.id] || ""}
                                            onChange={(e) => setReplyText(prev => ({ ...prev, [post.id]: e.target.value }))}
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter') handleReply(post.id);
                                            }}
                                        />
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            className="h-9 px-3"
                                            onClick={() => handleReply(post.id)}
                                            disabled={!replyText[post.id]?.trim()}
                                        >
                                            <Send className="h-4 w-4" />
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))
                    )}
                </div>
            </main>
        </div>
    );
}
