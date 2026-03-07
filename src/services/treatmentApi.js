

import AsyncStorage from '@react-native-async-storage/async-storage';

// API Configuration - Groq (using llama-3.3-70b - fast and free)
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const API_KEY_STORAGE = '@groq_api_key';

// Store API key
let GROQ_API_KEY = null;

/**
 * Set the Groq API key
 */
export const setApiKey = async (apiKey) => {
  GROQ_API_KEY = apiKey;
  await AsyncStorage.setItem(API_KEY_STORAGE, apiKey);
};

/**
 * Get the stored API key
 */
export const getApiKey = async () => {
  if (GROQ_API_KEY) return GROQ_API_KEY;
  const storedKey = await AsyncStorage.getItem(API_KEY_STORAGE);
  if (storedKey) {
    GROQ_API_KEY = storedKey;
  }
  return GROQ_API_KEY;
};

/**
 * Check if API key is configured
 */
export const isApiKeyConfigured = async () => {
  const key = await getApiKey();
  return !!key;
};

/**
 * Generate treatment recommendations using Gemini AI
 */
export const getTreatmentRecommendations = async ({
  pestType,
  severity,
  confidence,
  language = 'en',
}) => {
  const apiKey = await getApiKey();
  console.log('Treatment API - Key loaded:', apiKey ? apiKey.substring(0, 15) + '...' : 'NO KEY');

  if (!apiKey) {
    return {
      success: false,
      error: 'API key not configured',
      fallback: getFallbackTreatment(pestType, severity, language),
    };
  }

  const languageInstruction = getLanguageInstruction(language);
  const pestName = getPestDisplayName(pestType, language);

  const prompt = `You are an expert agricultural consultant specializing in coconut palm diseases and pest management in Sri Lanka.

A farmer has detected the following pest/disease on their coconut tree:
- Pest/Disease: ${pestName}
- Severity Level: ${severity} (${getSeverityDescription(severity)})
- Detection Confidence: ${(confidence * 100).toFixed(1)}%

${languageInstruction}

Provide comprehensive treatment recommendations in the following JSON format:
{
  "summary": "Brief 1-2 sentence summary of the situation",
  "urgency": "low|medium|high|critical",
  "treatments": [
    {
      "type": "chemical|organic|cultural",
      "name": "Treatment name",
      "description": "How to apply",
      "dosage": "Amount per liter/tree",
      "frequency": "How often to apply",
      "duration": "For how long",
      "cost_estimate": "Approximate cost in LKR"
    }
  ],
  "preventive_measures": ["List of prevention tips"],
  "safety_precautions": ["Safety tips when applying treatments"],
  "expected_recovery": "Expected recovery timeline",
  "when_to_seek_expert": "When to contact agricultural officer"
}

Important:
- Include both chemical and organic treatment options
- Use locally available products in Sri Lanka
- Consider the severity level when recommending treatments
- Be specific about dosages and frequencies
- Include safety precautions for chemical treatments

Respond ONLY with valid JSON, no additional text or markdown.`;

  try {
    const response = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        messages: [{
          role: 'user',
          content: prompt
        }],
        temperature: 0.7,
        max_tokens: 2000,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Groq API Error:', JSON.stringify(errorData, null, 2));
      console.error('API Key used:', apiKey ? apiKey.substring(0, 10) + '...' : 'none');
      console.error('Response status:', response.status);
      return {
        success: false,
        error: errorData.error?.message || 'API request failed',
        fallback: getFallbackTreatment(pestType, severity, language),
      };
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    if (!content) {
      return {
        success: false,
        error: 'Empty response from API',
        fallback: getFallbackTreatment(pestType, severity, language),
      };
    }

    // Clean the response - remove markdown code blocks if present
    let cleanedContent = content.trim();
    if (cleanedContent.startsWith('```json')) {
      cleanedContent = cleanedContent.slice(7);
    } else if (cleanedContent.startsWith('```')) {
      cleanedContent = cleanedContent.slice(3);
    }
    if (cleanedContent.endsWith('```')) {
      cleanedContent = cleanedContent.slice(0, -3);
    }
    cleanedContent = cleanedContent.trim();

    // Parse the JSON response
    try {
      const treatmentData = JSON.parse(cleanedContent);
      return {
        success: true,
        data: treatmentData,
        source: 'ai',
      };
    } catch (parseError) {
      console.error('JSON Parse Error:', parseError);
      console.error('Raw content:', content);
      return {
        success: false,
        error: 'Failed to parse treatment data',
        fallback: getFallbackTreatment(pestType, severity, language),
      };
    }
  } catch (error) {
    console.error('Treatment API Error:', error);
    return {
      success: false,
      error: error.message || 'Network error',
      fallback: getFallbackTreatment(pestType, severity, language),
    };
  }
};

/**
 * Get language-specific instruction for Gemini
 */
const getLanguageInstruction = (language) => {
  switch (language) {
    case 'si':
      return 'IMPORTANT: Respond in Sinhala (සිංහල) language. Use Sinhala script for all text content in the JSON values.';
    case 'ta':
      return 'IMPORTANT: Respond in Tamil (தமிழ்) language. Use Tamil script for all text content in the JSON values.';
    default:
      return 'Respond in English.';
  }
};

/**
 * Get pest display name
 */
const getPestDisplayName = (pestType, language) => {
  const names = {
    coconut_mite: {
      en: 'Coconut Mite (Aceria guerreronis)',
      si: 'පොල් මයිටාව (Aceria guerreronis)',
      ta: 'தென்னை பேன் (Aceria guerreronis)',
    },
    caterpillar: {
      en: 'Coconut Black-Headed Caterpillar (Opisina arenosella)',
      si: 'පොල් කළු හිස දළඹුවා (Opisina arenosella)',
      ta: 'தென்னை கருந்தலை புழு (Opisina arenosella)',
    },
    white_fly: {
      en: 'Coconut Whitefly (Aleurodicus destructor)',
      si: 'පොල් සුදු මැස්සා (Aleurodicus destructor)',
      ta: 'தென்னை வெள்ளை ஈ (Aleurodicus destructor)',
    },
  };

  return names[pestType]?.[language] || names[pestType]?.en || pestType;
};

/**
 * Get severity description
 */
const getSeverityDescription = (severity) => {
  const descriptions = {
    mild: '0-30% infestation, early stage',
    moderate: '30-60% infestation, spreading',
    severe: '60-100% infestation, heavy damage',
  };
  return descriptions[severity] || severity;
};

/**
 * Fallback treatment data when API is unavailable
 */
const getFallbackTreatment = (pestType, severity, language) => {
  const fallbackData = {
    coconut_mite: {
      en: {
        summary: 'Coconut mite infestation detected. Treatment recommended based on severity.',
        urgency: severity === 'severe' ? 'high' : severity === 'moderate' ? 'medium' : 'low',
        treatments: [
          {
            type: 'chemical',
            name: 'Wettable Sulphur',
            description: 'Mix with water and spray on affected nuts',
            dosage: '5g per liter of water',
            frequency: 'Every 21 days',
            duration: '3 applications',
            cost_estimate: 'LKR 500-800 per application',
          },
          {
            type: 'organic',
            name: 'Neem Oil Spray',
            description: 'Mix neem oil with water and mild soap, spray on affected areas',
            dosage: '30ml per liter of water',
            frequency: 'Weekly',
            duration: '4-6 weeks',
            cost_estimate: 'LKR 300-500 per application',
          },
        ],
        preventive_measures: [
          'Regular inspection of coconut bunches',
          'Remove and destroy heavily infested nuts',
          'Maintain proper spacing between trees',
          'Apply preventive spray during dry season',
        ],
        safety_precautions: [
          'Wear protective gloves and mask when spraying',
          'Avoid spraying during windy conditions',
          'Keep children and pets away during application',
          'Wash hands thoroughly after handling chemicals',
        ],
        expected_recovery: '4-8 weeks with proper treatment',
        when_to_seek_expert: 'If infestation spreads to more than 50% of trees or no improvement after 4 weeks',
      },
      si: {
        summary: 'පොල් මයිටා ආසාදනය හඳුනාගෙන ඇත. බරපතලකම අනුව ප්‍රතිකාර නිර්දේශ කෙරේ.',
        urgency: severity === 'severe' ? 'high' : severity === 'moderate' ? 'medium' : 'low',
        treatments: [
          {
            type: 'chemical',
            name: 'තෙත් කළ හැකි සල්ෆර්',
            description: 'ජලය සමඟ මිශ්‍ර කර ආසාදිත ගෙඩි මත ඉසින්න',
            dosage: 'ජලය ලීටරයකට 5g',
            frequency: 'සෑම දින 21කට වරක්',
            duration: 'අයදුම් 3ක්',
            cost_estimate: 'අයදුම් එකකට රු. 500-800',
          },
          {
            type: 'organic',
            name: 'කොහොඹ තෙල් ඉසීම',
            description: 'කොහොඹ තෙල් ජලය සහ මෘදු සබන් සමඟ මිශ්‍ර කර ආසාදිත ප්‍රදේශවල ඉසින්න',
            dosage: 'ජලය ලීටරයකට 30ml',
            frequency: 'සතිපතා',
            duration: 'සති 4-6',
            cost_estimate: 'අයදුම් එකකට රු. 300-500',
          },
        ],
        preventive_measures: [
          'පොල් කැකුළු නිරන්තර පරීක්ෂා කිරීම',
          'දැඩි ලෙස ආසාදිත ගෙඩි ඉවත් කර විනාශ කරන්න',
          'ගස් අතර නිසි පරතරය පවත්වන්න',
          'වියළි කාලයේ වැළැක්වීමේ ඉසීම යොදන්න',
        ],
        safety_precautions: [
          'ඉසින විට ආරක්ෂිත අත්වැසුම් සහ මාස්ක් පළඳින්න',
          'සුළං තත්ත්වයන්හිදී ඉසීමෙන් වළකින්න',
          'යෙදීමේදී ළමුන් සහ සුරතල් සතුන් දුරස් කරන්න',
          'රසායනික ද්‍රව්‍ය හැසිරවීමෙන් පසු අත් හොඳින් සෝදන්න',
        ],
        expected_recovery: 'නිසි ප්‍රතිකාර සමඟ සති 4-8',
        when_to_seek_expert: 'ආසාදනය ගස්වලින් 50%කට වඩා පැතිරුණහොත් හෝ සති 4කට පසුවත් වැඩිදියුණුවක් නොමැතිනම්',
      },
      ta: {
        summary: 'தென்னை பேன் தொற்று கண்டறியப்பட்டது. தீவிரத்தின் அடிப்படையில் சிகிச்சை பரிந்துரைக்கப்படுகிறது.',
        urgency: severity === 'severe' ? 'high' : severity === 'moderate' ? 'medium' : 'low',
        treatments: [
          {
            type: 'chemical',
            name: 'ஈரமான கந்தகம்',
            description: 'தண்ணீரில் கலந்து பாதிக்கப்பட்ட காய்களில் தெளிக்கவும்',
            dosage: 'ஒரு லிட்டர் தண்ணீருக்கு 5 கிராம்',
            frequency: 'ஒவ்வொரு 21 நாட்களுக்கும்',
            duration: '3 முறை',
            cost_estimate: 'ஒரு முறைக்கு ரூ. 500-800',
          },
          {
            type: 'organic',
            name: 'வேப்ப எண்ணெய் தெளிப்பு',
            description: 'வேப்ப எண்ணெயை தண்ணீர் மற்றும் சோப்புடன் கலந்து பாதிக்கப்பட்ட பகுதிகளில் தெளிக்கவும்',
            dosage: 'ஒரு லிட்டர் தண்ணீருக்கு 30 மில்லி',
            frequency: 'வாரந்தோறும்',
            duration: '4-6 வாரங்கள்',
            cost_estimate: 'ஒரு முறைக்கு ரூ. 300-500',
          },
        ],
        preventive_measures: [
          'தென்னை குலைகளை தொடர்ந்து ஆய்வு செய்யுங்கள்',
          'கடுமையாக பாதிக்கப்பட்ட காய்களை அகற்றி அழிக்கவும்',
          'மரங்களுக்கு இடையே சரியான இடைவெளியை பராமரிக்கவும்',
          'வறண்ட காலத்தில் தடுப்பு தெளிப்பு செய்யுங்கள்',
        ],
        safety_precautions: [
          'தெளிக்கும்போது பாதுகாப்பு கையுறைகள் மற்றும் முகமூடி அணியுங்கள்',
          'காற்று வீசும்போது தெளிப்பதை தவிர்க்கவும்',
          'பயன்படுத்தும்போது குழந்தைகள் மற்றும் செல்லப்பிராணிகளை தள்ளி வையுங்கள்',
          'இரசாயனங்களை கையாண்ட பின் கைகளை நன்றாக கழுவுங்கள்',
        ],
        expected_recovery: 'சரியான சிகிச்சையுடன் 4-8 வாரங்கள்',
        when_to_seek_expert: 'தொற்று 50% மரங்களுக்கு மேல் பரவினால் அல்லது 4 வாரங்களுக்குப் பிறகும் முன்னேற்றம் இல்லாவிட்டால்',
      },
    },
    caterpillar: {
      en: {
        summary: 'Caterpillar damage detected on coconut leaves. Immediate action recommended.',
        urgency: severity === 'severe' ? 'critical' : severity === 'moderate' ? 'high' : 'medium',
        treatments: [
          {
            type: 'chemical',
            name: 'Chlorpyrifos 20 EC',
            description: 'Spray on affected leaves, focusing on leaf undersides',
            dosage: '2ml per liter of water',
            frequency: 'Every 14 days',
            duration: '2-3 applications',
            cost_estimate: 'LKR 600-1000 per application',
          },
          {
            type: 'organic',
            name: 'Bacillus thuringiensis (Bt)',
            description: 'Biological insecticide, spray when caterpillars are small',
            dosage: '2g per liter of water',
            frequency: 'Every 7 days',
            duration: '3-4 applications',
            cost_estimate: 'LKR 400-700 per application',
          },
        ],
        preventive_measures: [
          'Regular monitoring of leaf condition',
          'Release natural predators like parasitic wasps',
          'Remove and burn heavily infested leaves',
          'Maintain tree health with proper nutrition',
        ],
        safety_precautions: [
          'Use full protective equipment for chemical sprays',
          'Do not spray during flowering to protect pollinators',
          'Wait 14 days after spraying before harvesting',
          'Store chemicals safely away from food and water',
        ],
        expected_recovery: '3-6 weeks depending on severity',
        when_to_seek_expert: 'If more than 30% of leaves are damaged or caterpillars reappear within 2 weeks',
      },
      si: {
        summary: 'පොල් කොළවල දළඹු හානි හඳුනාගෙන ඇත. ක්ෂණික ක්‍රියාමාර්ග නිර්දේශ කෙරේ.',
        urgency: severity === 'severe' ? 'critical' : severity === 'moderate' ? 'high' : 'medium',
        treatments: [
          {
            type: 'chemical',
            name: 'ක්ලෝර්පයිරිෆොස් 20 EC',
            description: 'ආසාදිත කොළ මත ඉසින්න, කොළ යටි පැත්තේ අවධානය යොමු කරන්න',
            dosage: 'ජලය ලීටරයකට 2ml',
            frequency: 'සෑම දින 14කට වරක්',
            duration: 'අයදුම් 2-3ක්',
            cost_estimate: 'අයදුම් එකකට රු. 600-1000',
          },
          {
            type: 'organic',
            name: 'බැසිලස් තුරින්ජියෙන්සිස් (Bt)',
            description: 'ජීව විද්‍යාත්මක කෘමිනාශකය, දළඹුවන් කුඩා විටදී ඉසින්න',
            dosage: 'ජලය ලීටරයකට 2g',
            frequency: 'සෑම දින 7කට වරක්',
            duration: 'අයදුම් 3-4ක්',
            cost_estimate: 'අයදුම් එකකට රු. 400-700',
          },
        ],
        preventive_measures: [
          'කොළ තත්ත්වය නිරන්තර අධීක්ෂණය',
          'පරපෝෂී බඹරුන් වැනි ස්වාභාවික විනාශකයින් මුදා හරින්න',
          'දැඩි ලෙස ආසාදිත කොළ ඉවත් කර පුළුස්සන්න',
          'නිසි පෝෂණය සමඟ ගස් සෞඛ්‍යය පවත්වන්න',
        ],
        safety_precautions: [
          'රසායනික ඉසීම් සඳහා සම්පූර්ණ ආරක්ෂිත උපකරණ භාවිතා කරන්න',
          'පරාග වාහකයන් ආරක්ෂා කිරීමට මල් පිපෙන කාලයේ ඉසීමෙන් වළකින්න',
          'අස්වනු නෙළීමට පෙර ඉසීමෙන් පසු දින 14ක් රැඳී සිටින්න',
          'ආහාර සහ ජලයෙන් ඈතින් රසායනික ද්‍රව්‍ය ආරක්ෂිතව ගබඩා කරන්න',
        ],
        expected_recovery: 'බරපතලකම අනුව සති 3-6',
        when_to_seek_expert: 'කොළවලින් 30%කට වඩා හානි වුවහොත් හෝ සති 2ක් ඇතුළත දළඹුවන් නැවත පෙනී ගියහොත්',
      },
      ta: {
        summary: 'தென்னை இலைகளில் புழு சேதம் கண்டறியப்பட்டது. உடனடி நடவடிக்கை பரிந்துரைக்கப்படுகிறது.',
        urgency: severity === 'severe' ? 'critical' : severity === 'moderate' ? 'high' : 'medium',
        treatments: [
          {
            type: 'chemical',
            name: 'குளோர்பைரிபாஸ் 20 EC',
            description: 'பாதிக்கப்பட்ட இலைகளில் தெளிக்கவும், இலையின் அடிப்பகுதியில் கவனம் செலுத்தவும்',
            dosage: 'ஒரு லிட்டர் தண்ணீருக்கு 2 மில்லி',
            frequency: 'ஒவ்வொரு 14 நாட்களுக்கும்',
            duration: '2-3 முறை',
            cost_estimate: 'ஒரு முறைக்கு ரூ. 600-1000',
          },
          {
            type: 'organic',
            name: 'பேசில்லஸ் துரிஞ்சியென்சிஸ் (Bt)',
            description: 'உயிரியல் பூச்சிக்கொல்லி, புழுக்கள் சிறியதாக இருக்கும்போது தெளிக்கவும்',
            dosage: 'ஒரு லிட்டர் தண்ணீருக்கு 2 கிராம்',
            frequency: 'ஒவ்வொரு 7 நாட்களுக்கும்',
            duration: '3-4 முறை',
            cost_estimate: 'ஒரு முறைக்கு ரூ. 400-700',
          },
        ],
        preventive_measures: [
          'இலை நிலையை தொடர்ந்து கண்காணிக்கவும்',
          'ஒட்டுண்ணி குளவிகள் போன்ற இயற்கை வேட்டையாடிகளை விடுவிக்கவும்',
          'கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றி எரிக்கவும்',
          'சரியான ஊட்டச்சத்துடன் மர ஆரோக்கியத்தை பராமரிக்கவும்',
        ],
        safety_precautions: [
          'இரசாயன தெளிப்புகளுக்கு முழு பாதுகாப்பு உபகரணங்களைப் பயன்படுத்தவும்',
          'மகரந்தச் சேர்க்கையாளர்களைப் பாதுகாக்க பூக்கும் காலத்தில் தெளிக்க வேண்டாம்',
          'அறுவடைக்கு முன் தெளித்த பின் 14 நாட்கள் காத்திருக்கவும்',
          'உணவு மற்றும் தண்ணீரிலிருந்து தொலைவில் இரசாயனங்களை பாதுகாப்பாக சேமிக்கவும்',
        ],
        expected_recovery: 'தீவிரத்தைப் பொறுத்து 3-6 வாரங்கள்',
        when_to_seek_expert: '30% க்கும் அதிகமான இலைகள் சேதமடைந்தால் அல்லது 2 வாரங்களுக்குள் புழுக்கள் மீண்டும் தோன்றினால்',
      },
    },
  };

  const pestData = fallbackData[pestType];
  if (!pestData) {
    return {
      summary: 'Treatment information not available for this pest type.',
      urgency: 'medium',
      treatments: [],
      preventive_measures: ['Consult local agricultural officer'],
      safety_precautions: [],
      expected_recovery: 'Varies',
      when_to_seek_expert: 'Immediately if unsure about treatment',
    };
  }

  return pestData[language] || pestData.en;
};

export default {
  getTreatmentRecommendations,
  setApiKey,
  getApiKey,
  isApiKeyConfigured,
};
