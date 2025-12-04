declare global {
  interface Window {
    googleTranslateElementInit?: () => void;
    google?: any;
  }
}

let scriptLoaded = false;

export function initGoogleTranslate(): void {
  if (scriptLoaded) return;
  scriptLoaded = true;

  // Create container (can stay hidden)
  if (!document.getElementById('google_translate_element')) {
    const container = document.createElement('div');
    container.id = 'google_translate_element';
    container.style.position = 'fixed';
    container.style.bottom = '0';
    container.style.right = '0';
    container.style.opacity = '0';
    container.style.pointerEvents = 'none';
    document.body.appendChild(container);
  }

  window.googleTranslateElementInit = function () {
    // Initialize with only the required languages
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const TranslateElement: any = window.google?.translate?.TranslateElement;
    if (TranslateElement) {
      // pageLanguage 'en' and included 'en,hi,pa'
      new TranslateElement({
        pageLanguage: 'en',
        includedLanguages: 'en,hi,pa',
        autoDisplay: false,
        layout: window.google.translate.TranslateElement.InlineLayout.SIMPLE,
      }, 'google_translate_element');
    }
  };

  const s = document.createElement('script');
  s.type = 'text/javascript';
  s.src = '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
  document.head.appendChild(s);
}

const LABEL_TO_CODE: Record<'English' | 'Hindi' | 'Punjabi', 'en' | 'hi' | 'pa'> = {
  English: 'en',
  Hindi: 'hi',
  Punjabi: 'pa',
};

export function setLanguage(label: 'English' | 'Hindi' | 'Punjabi'): void {
  const lang = LABEL_TO_CODE[label];
  const base = '/en/' + lang;
  // Set cookies used by Google Translate
  const host = window.location.hostname;
  document.cookie = `googtrans=${base}; path=/`;
  document.cookie = `googtrans=${base}; domain=${host}; path=/`;
  // Soft refresh to let translator re-run
  // Using location.reload() is reliable; avoid full navigation changes
  window.location.reload();
}

/**
 * Clears the Google Translate language preference cookies.
 * This ensures the website loads in English (default) on next visit.
 * Should be called during logout to prevent language persistence.
 */
export function clearLanguagePreference(): void {
  const host = window.location.hostname;
  const pastDate = 'Thu, 01 Jan 1970 00:00:00 GMT';

  // Clear cookies from all possible domains and paths
  document.cookie = `googtrans=; expires=${pastDate}; path=/`;
  document.cookie = `googtrans=; expires=${pastDate}; domain=${host}; path=/`;
  document.cookie = `googtrans=; expires=${pastDate}; domain=.${host}; path=/`;

  // Also clear the alternate cookie name that Google Translate sometimes uses
  document.cookie = `googtrans=/en/en; expires=${pastDate}; path=/`;
  document.cookie = `googtrans=/en/en; expires=${pastDate}; domain=${host}; path=/`;
}

/**
 * Alternative method to reset language to English.
 * Sets the cookie to /en/en instead of deleting it.
 * Can be used as a fallback if clearLanguagePreference doesn't work.
 */
export function resetToEnglish(): void {
  const host = window.location.hostname;
  document.cookie = `googtrans=/en/en; path=/`;
  document.cookie = `googtrans=/en/en; domain=${host}; path=/`;
}
