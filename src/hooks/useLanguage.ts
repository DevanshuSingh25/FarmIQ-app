import { useState, useEffect } from 'react';
import { getCurrentLanguage } from '@/lib/translations';

export function useLanguage() {
  const [language, setLanguage] = useState<'English' | 'Hindi' | 'Punjabi'>(getCurrentLanguage());

  useEffect(() => {
    const handleLanguageChange = (event: CustomEvent) => {
      const { language: newLanguage } = event.detail;
      setLanguage(newLanguage);
    };

    window.addEventListener('languageChanged', handleLanguageChange as EventListener);
    
    return () => {
      window.removeEventListener('languageChanged', handleLanguageChange as EventListener);
    };
  }, []);

  return language;
}
