import React, {createContext, useState, useContext, useEffect} from 'react';
import i18n, {
  languages,
  getStoredLanguage,
  setStoredLanguage,
  initializeLanguage,
} from '../locales';

const LanguageContext = createContext();

export const LanguageProvider = ({children}) => {
  console.log('=== LanguageProvider rendering ===');
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    console.log('=== LanguageProvider useEffect triggered ===');
    loadLanguage();
  }, []);

  const loadLanguage = async () => {
    console.log('=== loadLanguage started ===');
    try {
      const language = await initializeLanguage();
      console.log('=== Language loaded:', language, '===');
      setCurrentLanguage(language);
    } catch (error) {
      console.error('Error loading language:', error);
    } finally {
      console.log('=== setIsLoading(false) ===');
      setIsLoading(false);
    }
  };

  const changeLanguage = async (languageCode) => {
    try {
      const success = await setStoredLanguage(languageCode);
      if (success) {
        setCurrentLanguage(languageCode);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error changing language:', error);
      return false;
    }
  };

  const t = (key, options) => {
    return i18n.t(key, options);
  };

  const value = {
    currentLanguage,
    languages,
    changeLanguage,
    t,
    isLoading,
  };

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

export default LanguageContext;
