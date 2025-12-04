// Mock TTS API utilities for development and testing

export interface TTSResponse {
  audioUrl: string;
  durationMs: number;
  message?: string;
}

export interface TTSRequest {
  text: string;
  lang: string;
}

/**
 * Mock TTS API handler - simulates a real TTS service
 * In production, this would be replaced with actual TTS service calls
 */
export const mockTTSHandler = async (request: TTSRequest): Promise<TTSResponse> => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 200));

  // Simulate occasional failures (5% chance)
  if (Math.random() < 0.05) {
    throw new Error('TTS service temporarily unavailable');
  }

  // Calculate mock duration based on text length (approximate 150 words per minute)
  const wordCount = request.text.split(/\s+/).length;
  const durationMs = Math.max(2000, (wordCount / 150) * 60 * 1000);

  return {
    audioUrl: `/mock/audio/tts_${Date.now()}.mp3`,
    durationMs: Math.round(durationMs),
    message: 'TTS audio generated successfully'
  };
};

/**
 * Intercept fetch requests to /api/tts and return mock responses
 * This allows the SectionSpeaker component to work without a real backend
 */
export const setupMockTTSAPI = () => {
  // Store original fetch
  const originalFetch = window.fetch;

  // Override fetch for TTS API calls only
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    // Convert input to string URL for checking
    let urlString: string;
    if (typeof input === 'string') {
      urlString = input;
    } else if (input instanceof URL) {
      urlString = input.href;
    } else if (input instanceof Request) {
      urlString = input.url;
    } else {
      urlString = String(input);
    }
    
    // Only intercept TTS API calls - explicitly exclude prediction API and other APIs
    const isTTSRequest = urlString.includes('/api/tts') && init?.method === 'POST';
    const isPredictionAPI = urlString.includes('localhost:8000') || urlString.includes('/predict');
    const isAuthAPI = urlString.includes('/api/auth');
    
    // Only handle TTS requests, pass through everything else (prediction, auth, etc.)
    if (isTTSRequest && !isPredictionAPI && !isAuthAPI) {
      try {
        const requestBody = JSON.parse(init.body as string) as TTSRequest;
        const response = await mockTTSHandler(requestBody);
        
        return new Response(JSON.stringify(response), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      } catch (error) {
        return new Response(JSON.stringify({ message: error instanceof Error ? error.message : 'Unknown error' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    // For all other requests (auth, prediction, etc.), use original fetch
    // This ensures they go through normally without interference
    return originalFetch(input, init);
  };
};

/**
 * Language code mapping for TTS
 */
export const TTS_LANGUAGES = {
  'en': 'en-US',
  'en-IN': 'en-IN',
  'hi': 'hi-IN',
  'te': 'te-IN',
  'ta': 'ta-IN',
  'bn': 'bn-IN',
  'mr': 'mr-IN',
  'gu': 'gu-IN',
  'kn': 'kn-IN',
  'ml': 'ml-IN',
  'pa': 'pa-IN',
  'or': 'or-IN'
} as const;

export type SupportedLanguage = keyof typeof TTS_LANGUAGES;