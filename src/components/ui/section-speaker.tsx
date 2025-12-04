import { useState, useEffect, useRef } from "react";
import { Volume2, VolumeX, Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { getCurrentLanguage } from "@/lib/translations";

// Singleton audio manager
class AudioManager {
  private static instance: AudioManager;
  private audio: HTMLAudioElement;
  private currentSectionId: string | null = null;
  private listeners: Map<string, (state: AudioState) => void> = new Map();
  private cache: Map<string, string> = new Map();

  private constructor() {
    this.audio = new Audio();
    this.audio.id = "global-tts-player";
    this.setupEventListeners();
  }

  static getInstance(): AudioManager {
    if (!AudioManager.instance) {
      AudioManager.instance = new AudioManager();
    }
    return AudioManager.instance;
  }

  private setupEventListeners() {
    this.audio.addEventListener('ended', () => {
      this.notifyListeners('ended');
      this.currentSectionId = null;
    });

    this.audio.addEventListener('error', () => {
      this.notifyListeners('error');
      this.currentSectionId = null;
    });

    this.audio.addEventListener('play', () => {
      this.notifyListeners('playing');
    });

    this.audio.addEventListener('pause', () => {
      this.notifyListeners('paused');
    });
  }

  private notifyListeners(state: AudioState) {
    if (this.currentSectionId) {
      const listener = this.listeners.get(this.currentSectionId);
      if (listener) {
        listener(state);
      }
    }
  }

  subscribe(sectionId: string, callback: (state: AudioState) => void) {
    this.listeners.set(sectionId, callback);
  }

  unsubscribe(sectionId: string) {
    this.listeners.delete(sectionId);
  }

  async play(sectionId: string, text: string, lang: string = 'en'): Promise<void> {
    // Stop current playback if different section
    if (this.currentSectionId && this.currentSectionId !== sectionId) {
      this.stop();
    }

    this.currentSectionId = sectionId;
    
    // Truncate long text
    const truncatedText = text.length > 1000 ? text.substring(0, 1000) + "..." : text;
    const cacheKey = `${sectionId}_${this.hashText(truncatedText)}_${lang}`;

    try {
      // Check cache first
      let audioUrl = this.cache.get(cacheKey);
      
      if (!audioUrl) {
        // Mock TTS API call
        const response = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: truncatedText, lang })
        });

        if (!response.ok) {
          throw new Error('TTS service unavailable');
        }

        const data = await response.json();
        audioUrl = data.audioUrl || '';
        this.cache.set(cacheKey, audioUrl);
      }

      // Use Web Speech API as fallback/primary method
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(truncatedText);
        utterance.rate = 0.8;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        utterance.lang = lang;
        
        utterance.onend = () => {
          this.notifyListeners('ended');
          this.currentSectionId = null;
        };
        
        utterance.onerror = () => {
          this.notifyListeners('error');
          this.currentSectionId = null;
        };
        
        speechSynthesis.speak(utterance);
      } else {
        throw new Error('Speech synthesis not supported');
      }
      
    } catch (error) {
      this.notifyListeners('error');
      this.currentSectionId = null;
      throw error;
    }
  }

  pause() {
    if ('speechSynthesis' in window) {
      speechSynthesis.pause();
    }
    this.audio.pause();
  }

  resume() {
    if ('speechSynthesis' in window) {
      speechSynthesis.resume();
    }
    this.audio.play();
  }

  stop() {
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel();
    }
    this.audio.pause();
    this.audio.currentTime = 0;
    this.currentSectionId = null;
  }

  isCurrentSection(sectionId: string): boolean {
    return this.currentSectionId === sectionId;
  }

  private hashText(text: string): string {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash.toString();
  }
}

type AudioState = 'idle' | 'playing' | 'paused' | 'ended' | 'error';

interface SectionSpeakerProps {
  getText: () => string;
  sectionId: string;
  lang?: string;
  ariaLabel?: string;
  className?: string;
  alwaysVisible?: boolean; // For mobile
}

export const SectionSpeaker = ({ 
  getText, 
  sectionId, 
  lang = 'en-IN', 
  ariaLabel,
  className,
  alwaysVisible = false 
}: SectionSpeakerProps) => {
  const [state, setState] = useState<AudioState>('idle');
  const [progress, setProgress] = useState(0);
  const audioManager = useRef(AudioManager.getInstance());
  const debounceTimer = useRef<NodeJS.Timeout>();
  const { toast } = useToast();

  useEffect(() => {
    const manager = audioManager.current;
    
    const handleStateChange = (newState: AudioState) => {
      setState(newState);
      if (newState === 'ended') {
        setProgress(0);
      }
    };

    manager.subscribe(sectionId, handleStateChange);

    return () => {
      manager.unsubscribe(sectionId);
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [sectionId]);

  const handleClick = async () => {
    // Debounce rapid clicks
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(async () => {
      const currentLang = getCurrentLanguage();
      if (currentLang === 'Hindi' || currentLang === 'Punjabi') {
        toast({
          title: "Not available",
          description: "",
          variant: "destructive"
        });
        return;
      }
      const manager = audioManager.current;
      const isCurrentSection = manager.isCurrentSection(sectionId);

      try {
        if (state === 'playing' && isCurrentSection) {
          manager.pause();
          setState('paused');
        } else if (state === 'paused' && isCurrentSection) {
          manager.resume();
          setState('playing');
        } else {
          // Start new playback
          const text = getText();
          if (!text.trim()) {
            toast({
              title: "No content",
              description: "No text available to read",
              variant: "destructive"
            });
            return;
          }

          setState('playing');
          await manager.play(sectionId, text, lang);
          
          // Announce to screen readers
          const announcement = document.createElement('div');
          announcement.setAttribute('aria-live', 'polite');
          announcement.className = 'sr-only';
          announcement.textContent = `Playing ${text.split('.')[0]}`;
          document.body.appendChild(announcement);
          setTimeout(() => document.body.removeChild(announcement), 1000);
        }
      } catch (error) {
        console.error('TTS Error:', error);
        setState('idle');
        
        if (!navigator.onLine) {
          toast({
            title: "Offline",
            description: "TTS not available offline",
            variant: "destructive"
          });
        } else {
          toast({
            title: "Audio Error",
            description: "Could not play audio",
            variant: "destructive"
          });
        }
      }
    }, 250);
  };

  const getIcon = () => {
    switch (state) {
      case 'playing':
        return <Pause className="h-4 w-4" />;
      case 'paused':
        return <Play className="h-4 w-4" />;
      default:
        return <Volume2 className="h-4 w-4" />;
    }
  };

  const getTooltip = () => {
    switch (state) {
      case 'playing':
        return 'Pause';
      case 'paused':
        return 'Resume';
      default:
        return ariaLabel || 'Speak section';
    }
  };

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        onClick={(e) => { e.stopPropagation(); handleClick(); }}
        onMouseDown={(e) => e.stopPropagation()}
        onTouchStart={(e) => e.stopPropagation()}
        className={cn(
          "h-8 w-8 p-0 text-muted-foreground hover:text-foreground transition-colors",
          state === 'playing' && "text-primary animate-pulse",
          !alwaysVisible && "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
          alwaysVisible && "opacity-100",
          className
        )}
        title={getTooltip()}
        aria-label={getTooltip()}
        aria-pressed={state === 'playing'}
      >
        {getIcon()}
      </Button>
      
      {/* Progress bar */}
      {state === 'playing' && (
        <div className="absolute -top-1 left-0 right-0">
          <div className="h-0.5 bg-primary/20 rounded-full">
            <div 
              className="h-0.5 bg-primary rounded-full transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};