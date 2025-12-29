import {I18n} from 'i18n-js';
import AsyncStorage from '@react-native-async-storage/async-storage';

import en from './en.json';
import si from './si.json';
import ta from './ta.json';

const i18n = new I18n({
  en,
  si,
  ta,
});

// Set default locale
i18n.defaultLocale = 'en';
i18n.locale = 'en';
i18n.enableFallback = true;

// Storage key for language preference
const LANGUAGE_KEY = '@app_language';

// Available languages
export const languages = [
  {code: 'en', name: 'English', nativeName: 'English'},
  {code: 'si', name: 'Sinhala', nativeName: 'සිංහල'},
  {code: 'ta', name: 'Tamil', nativeName: 'தமிழ்'},
];

// Get stored language preference
export const getStoredLanguage = async () => {
  try {
    const language = await AsyncStorage.getItem(LANGUAGE_KEY);
    return language || 'en';
  } catch (error) {
    console.error('Error getting stored language:', error);
    return 'en';
  }
};

// Save language preference
export const setStoredLanguage = async (languageCode) => {
  try {
    await AsyncStorage.setItem(LANGUAGE_KEY, languageCode);
    i18n.locale = languageCode;
    return true;
  } catch (error) {
    console.error('Error saving language:', error);
    return false;
  }
};

// Initialize language from storage
export const initializeLanguage = async () => {
  const storedLanguage = await getStoredLanguage();
  i18n.locale = storedLanguage;
  return storedLanguage;
};

// Translation function shorthand
export const t = (key, options) => i18n.t(key, options);

export default i18n;
