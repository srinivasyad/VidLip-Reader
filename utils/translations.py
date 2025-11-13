# utils/translations.py
"""
Combined translations module.

- Uses an offline dictionary (OFFLINE_DICT) for known phrases.
- If phrase not found offline, tries googletrans (if installed) as fallback.
- If googletrans is unavailable or fails, returns "Translation not available".
"""

import os

# ---- Available languages (map display names -> codes) ----
AVAILABLE_LANGS = {
    "Kannada": "kn",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Odia": "or",
    "Urdu": "ur",
    "English": "en"
}

# ---- Offline dictionary (keep/extend the phrases you provided) ----
OFFLINE_DICT = {
    "Hello": {"Kannada":"ನಮಸ್ಕಾರ","Hindi":"नमस्ते","Tamil":"வணக்கம்","Telugu":"హలో","Malayalam":"ഹലോ","Marathi":"नमस्कार","Gujarati":"હેલો","Bengali":"নমস্কার","Punjabi":"ਸਤ ਸ੍ਰੀ ਅਕਾਲ","Odia":"ନମସ୍କାର","Urdu":"ہیلو"},
    "Stop navigation.": {"Kannada":"ನ್ಯಾವಿಗೇಶನ್ ನಿಲ್ಲಿಸಿ.","Hindi":"नेविगेशन रोकें।","Tamil":"வழிசெலுத்துதலை நிறுத்துங்கள்.","Telugu":"నావిగేషన్ ఆపు.","Malayalam":"നാവിഗേഷൻ നിർത്തുക.","Marathi":"नॅव्हिगेशन थांबवा.","Gujarati":"નેવિગેશન રોકો.","Bengali":"নেভিগেশন বন্ধ করুন।","Punjabi":"ਨੈਵੀਗੇਸ਼ਨ ਰੋਕੋ।","Odia":"ନେଭିଗେସନ୍ ବନ୍ଦ କରନ୍ତୁ।","Urdu":"نیویگیشن بند کریں۔"},
    "I am sorry.": {"Kannada":"ಕ್ಷಮಿಸಿ.","Hindi":"मुझे खेद है।","Tamil":"மன்னிக்கவும்.","Telugu":"క్షమించండి.","Malayalam":"ക്ഷമിക്കണം.","Marathi":"मला खेद आहे.","Gujarati":"મને ખેદ છે.","Bengali":"আমি দুঃখিত।","Punjabi":"ਮੈਂ ਮਾਫੀ ਚਾਹੁੰਦਾ/ਚਾਹੁੰਦੀ ਹਾਂ।","Odia":"ମୁଁ ଦୁଃଖିତ।","Urdu":"مجھے افسوس ہے۔"},
    "Thank you.": {"Kannada":"ಧನ್ಯವಾದಗಳು.","Hindi":"धन्यवाद।","Tamil":"நன்றி.","Telugu":"ధన్యవాదాలు.","Malayalam":"നന്ദി.","Marathi":"धन्यवाद.","Gujarati":"આભાર.","Bengali":"ধন্যবাদ।","Punjabi":"ਸ਼ੁਕਰੀਆ।","Odia":"ଧନ୍ୟବାଦ।","Urdu":"شکریہ۔"},
    "Good bye.": {"Kannada":"ವಿದಾಯ.","Hindi":"अलविदा।","Tamil":"பிரியாவிடை.","Telugu":"వీడ్కోలు.","Malayalam":"വിട.","Marathi":"निरोप.","Gujarati":"અલવિદા.","Bengali":"বিদায়।","Punjabi":"ਅਲਵਿਦਾ।","Odia":"ବିଦାୟ।","Urdu":"الوداع۔"},
    "Begin": {"Kannada":"ಆರಂಭಿಸಿ","Hindi":"आरंभ करें","Tamil":"தொடங்கு","Telugu":"ప్రారంభించు","Malayalam":"ആരഭിക്കുക","Marathi":"सुरू करा","Gujarati":"શરૂ કરો","Bengali":"শুরু করুন","Punjabi":"ਸ਼ੁਰੂ ਕਰੋ","Odia":"ଆରମ୍ଭ କରନ୍ତୁ","Urdu":"شروع کریں"},
    "Choose": {"Kannada":"ಆಯ್ಕೆಮಾಡಿ","Hindi":"चुनें","Tamil":"தேர்வு செய்","Telugu":"ఎంచుకోండి","Malayalam":"തിരഞ്ഞെടുക്കുക","Marathi":" निवडा","Gujarati":"પસંદ કરો","Bengali":"পছন্দ করুন","Punjabi":"ਚੁਣੋ","Odia":"ଚୟନ କରନ୍ତୁ","Urdu":"منتخب کریں"},
    "Navigation": {"Kannada":"ನ್ಯಾವಿಗೇಶನ್","Hindi":"नेविगेशन","Tamil":"நாக்கிசேன","Telugu":"నావిగేషన్","Malayalam":"നാവിഗേഷൻ","Marathi":"नॅव्हिगेशन","Gujarati":"નેવિગેશન","Bengali":"নেভিগেশন","Punjabi":"ਨੈਵੀਗੇਸ਼ਨ","Odia":"ନେଭିଗେସନ୍","Urdu":"نیویگیشن"},
    "Next": {"Kannada":"ಮುಂದೆ","Hindi":"अगला","Tamil":"அடுத்தது","Telugu":"తరువాత","Malayalam":"അടുത്തത്","Marathi":"पुढचे","Gujarati":"આગળ","Bengali":"পরবর্তী","Punjabi":"ਅਗਲਾ","Odia":"ପରବର୍ତ୍ତୀ","Urdu":"اگلا"},
    "Previous": {"Kannada":"ಹಿಂದೆ","Hindi":"पिछला","Tamil":"முந்தையது","Telugu":"మునుపటి","Malayalam":"മുമ്പത്തെ","Marathi":"मागील","Gujarati":"પાછળ","Bengali":"পূর্ববর্তী","Punjabi":"ਪਿਛਲਾਂ","Odia":"ପୂର୍ବତନ","Urdu":"پچھلا"},
    "Start": {"Kannada":"ಪ್ರಾರಂಭಿಸಿ","Hindi":"शुरू","Tamil":"தொடங்கு","Telugu":"ప్రారంభించండి","Malayalam":"ആരഭിക്കുക","Marathi":"प्रारंभ","Gujarati":"પ્રારંભ","Bengali":"শুরু","Punjabi":"ਸ਼ੁਰੂ","Odia":"ଆରମ୍ଭ"},
    "Stop": {"Kannada":"ನಿಲ್ಲಿಸಿ","Hindi":"रुको","Tamil":"நிறுத்து","Telugu":"నిలుపు","Malayalam":"നിർത്തു","Marathi":"थांबा","Gujarati":"બંધ કરો","Bengali":"বন্দ করো","Punjabi":"ਰੋਕੋ","Odia":"ନିରୋଧ","Urdu":"روکیں"},
    "How are you?": {"Kannada":"ನೀವು ಹೇಗಿದ್ದೀರಾ?","Hindi":"आप कैसे हैं?","Tamil":"நீங்கள் எப்படி இருக்கிறீர்கள்?","Telugu":"మీరు ఎలా ఉన్నారు?","Malayalam":"സുഖമാണോ?","Marathi":"तू कसा आहेस?","Gujarati":"તમે કેમ છો?","Bengali":"কেমন আছো?","Punjabi":"ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?","Odia":"କେମିତି ଅଛନ୍ତି?","Urdu":"آپ کیسے ہیں؟"},
    "I love this game.": {"Kannada":"ನಾನು ಈ ಆಟವನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ.","Hindi":"मुझे यह खेल बहुत पसंद है।","Tamil":"எனக்கு இந்த விளையாட்டு பிடிக்கும்.","Telugu":"నాకు ఈ ఆట ఇష్టం.","Malayalam":"എനിക്ക് ഈ ഗെയിം ഇഷ്ടമാണ്.","Marathi":"मला हा खेळ आवडतो.","Gujarati":"મને આ રમત ગમે છે.","Bengali":"আমি এই খেলাটি পছন্দ করি।","Punjabi":"ਮੈਨੂੰ ਇਹ ਖੇਡ ਪਸੰਦ ਹੈ।","Odia":"ମୋତେ ଏହି ଖେଳ ପସନ୍ଦ।","Urdu":"مجھے یہ کھیل پسند ہے۔"},
}

# ---- Try to import googletrans as fallback ----
try:
    from googletrans import Translator
    _TRANSLATOR = Translator()
except Exception:
    _TRANSLATOR = None

# ---- Main translation function ----
def get_translation(english_text: str, target_language_name: str) -> str:
    """
    Return translation for english_text into the requested language name.
    Priority:
      1) OFFLINE_DICT exact match
      2) googletrans if available
      3) fallback message "Translation not available"
    """
    if not english_text:
        return ""

    # 1) offline exact-match lookup
    if english_text in OFFLINE_DICT:
        mapping = OFFLINE_DICT[english_text]
        if target_language_name in mapping:
            return mapping[target_language_name]

    # 2) googletrans fallback
    if _TRANSLATOR:
        lang_code = AVAILABLE_LANGS.get(target_language_name, None)
        if lang_code:
            try:
                res = _TRANSLATOR.translate(english_text, dest=lang_code)
                # sometimes googletrans returns same text if no translation available;
                # still return it because it's better than nothing.
                return res.text
            except Exception:
                pass

    # 3) final fallback
    return "Translation not available"
