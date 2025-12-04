import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { MessageCircle, X, Send, Mic } from "lucide-react";
import { Input } from "@/components/ui/input";

export function ChatbotWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState("");

  const toggleChat = () => setIsOpen(!isOpen);

  return (
    <div className="fixed bottom-20 right-6 z-50">
      {/* Chat Window */}
      {isOpen && (
        <Card className="w-80 h-96 mb-4 shadow-strong bg-card border border-border animate-scale-in">
          <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border bg-primary text-primary-foreground">
              <h3 className="font-semibold">FarmIQ Assistant</h3>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={toggleChat}
                className="h-6 w-6 p-0 hover:bg-primary-foreground/20"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Chat Content */}
            <div className="flex-1 p-4 overflow-y-auto">
              <div className="space-y-4">
                <div className="flex justify-start">
                  <div className="bg-muted p-3 rounded-lg max-w-xs">
                    <p className="text-sm">
                      Hello! I'm your FarmIQ assistant. How can I help you with your farming needs today?
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Input */}
            <div className="p-4 border-t border-border">
              <div className="flex gap-2">
                <Input
                  placeholder="Type your message..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      // Handle send message
                      setMessage("");
                    }
                  }}
                  className="flex-1"
                />
                <Button size="sm" className="px-3">
                  <Send className="h-4 w-4" />
                </Button>
                <Button size="sm" variant="outline" className="px-3" title="Voice input (coming soon)">
                  <Mic className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Chat Toggle Button */}
      <Button
        onClick={toggleChat}
        className="h-14 w-14 rounded-full shadow-strong bg-primary hover:bg-primary-glow transition-all duration-300 hover:scale-110"
      >
        <MessageCircle className="h-6 w-6" />
      </Button>
    </div>
  );
}