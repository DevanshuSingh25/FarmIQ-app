import { useState } from "react";
import { Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface TextToSpeechProps {
  text: string;
  className?: string;
  size?: "sm" | "default" | "lg";
}

export const TextToSpeech = ({ text, className, size = "sm" }: TextToSpeechProps) => {
  const [isPlaying, setIsPlaying] = useState(false);

  const handleSpeak = async () => {
    if (isPlaying) {
      // Stop speaking
      speechSynthesis.cancel();
      setIsPlaying(false);
      return;
    }

    try {
      // Mock TTS API call
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang: 'en' })
      });

      if (!response.ok) {
        throw new Error('TTS service unavailable');
      }

      setIsPlaying(true);

      // Use Web Speech API as fallback
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.8;
      utterance.pitch = 1;
      utterance.volume = 0.8;
      
      utterance.onend = () => setIsPlaying(false);
      utterance.onerror = () => setIsPlaying(false);
      
      speechSynthesis.speak(utterance);
    } catch (error) {
      console.error('TTS Error:', error);
      setIsPlaying(false);
    }
  };

  const iconSize = size === "sm" ? "h-3 w-3" : size === "lg" ? "h-5 w-5" : "h-4 w-4";
  const buttonSize = size === "sm" ? "h-6 w-6" : size === "lg" ? "h-10 w-10" : "h-8 w-8";

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleSpeak}
      className={cn(
        buttonSize,
        "text-muted-foreground hover:text-foreground transition-colors",
        isPlaying && "text-primary animate-pulse",
        className
      )}
      title={isPlaying ? "Stop reading" : "Read aloud"}
      aria-label={isPlaying ? "Stop reading" : "Read aloud"}
    >
      {isPlaying ? (
        <VolumeX className={iconSize} />
      ) : (
        <Volume2 className={iconSize} />
      )}
    </Button>
  );
};