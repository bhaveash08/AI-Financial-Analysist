"""
localization.py - Vernacular Investment Intelligence
=====================================================
Generates translated investment decision summaries and TTS-ready audio scripts
in Indic languages: Hindi, Tamil, Telugu, Marathi, English.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Translation Dictionaries
# ---------------------------------------------------------------------------

# Core financial term translations
TERM_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "BUY": {
        "en": "BUY", "hi": "खरीदें", "ta": "வாங்க", "te": "కొనండి", "mr": "विकत घ्या",
    },
    "SELL": {
        "en": "SELL", "hi": "बेचें", "ta": "விற்க", "te": "అమ్మండి", "mr": "विका",
    },
    "HOLD": {
        "en": "HOLD", "hi": "रखें", "ta": "வைத்திரு", "te": "ఉంచండి", "mr": "ठेवा",
    },
    "Bullish": {
        "en": "Bullish", "hi": "तेजी", "ta": "ஏற்றம்", "te": "బుల్లిష్", "mr": "तेजी",
    },
    "Bearish": {
        "en": "Bearish", "hi": "मंदी", "ta": "சரிவு", "te": "బియరిష్", "mr": "मंदी",
    },
    "Neutral": {
        "en": "Neutral", "hi": "तटस्थ", "ta": "நடுநிலை", "te": "తటస్థ", "mr": "तटस्थ",
    },
    "Risk Level": {
        "en": "Risk Level", "hi": "जोखिम स्तर", "ta": "ஆபத்து நிலை", "te": "ప్రమాద స్థాయి", "mr": "जोखिम पातळी",
    },
    "Low": {
        "en": "Low", "hi": "कम", "ta": "குறைவு", "te": "తక్కువ", "mr": "कमी",
    },
    "Moderate": {
        "en": "Moderate", "hi": "मध्यम", "ta": "சராசரி", "te": "మధ్యస్థం", "mr": "मध्यम",
    },
    "High": {
        "en": "High", "hi": "उच्च", "ta": "அதிகம்", "te": "ఎక్కువ", "mr": "जास्त",
    },
    "Expected Return": {
        "en": "Expected Return", "hi": "अपेक्षित रिटर्न", "ta": "எதிர்பார்க்கப்படும் வருமானம்",
        "te": "అంచనా రిటర్న్", "mr": "अपेक्षित परतावा",
    },
    "Investment Amount": {
        "en": "Investment Amount", "hi": "निवेश राशि", "ta": "முதலீட்டு தொகை",
        "te": "పెట్టుబడి మొత్తం", "mr": "गुंतवणूक रक्कम",
    },
    "Recommendation": {
        "en": "Recommendation", "hi": "सिफारिश", "ta": "பரிந்துரை",
        "te": "సిఫార్సు", "mr": "शिफारस",
    },
    "Confidence": {
        "en": "Confidence", "hi": "विश्वास", "ta": "நம்பிக்கை",
        "te": "నమ్మకం", "mr": "विश्वास",
    },
    "Market Overview": {
        "en": "Market Overview", "hi": "बाज़ार अवलोकन", "ta": "சந்தை மேலோட்டம்",
        "te": "మార్కెట్ అవలోకనం", "mr": "बाजार आढावा",
    },
    "Portfolio Summary": {
        "en": "Portfolio Summary", "hi": "पोर्टफोलियो सारांश", "ta": "போர்ட்ஃபோலியோ சுருக்கம்",
        "te": "పోర్ట్‌ఫోలియో సారాంశం", "mr": "पोर्टफोलिओ सारांश",
    },
    "Disclaimer": {
        "en": "Disclaimer", "hi": "अस्वीकरण", "ta": "மறுப்பு",
        "te": "నిరాకరణ", "mr": "अस्वीकरण",
    },
    "Volume Anomaly Detected": {
        "en": "Volume Anomaly Detected", "hi": "वॉल्यूम विसंगति का पता चला",
        "ta": "ஒலிஅளவு வித்தியாசம் கண்டறியப்பட்டது",
        "te": "వాల్యూమ్ అనోమలీ గుర్తించబడింది", "mr": "व्हॉल्यूम विसंगती आढळली",
    },
    "Data Degraded": {
        "en": "Data Degraded", "hi": "डेटा कम गुणवत्ता", "ta": "தரவு சரிவு",
        "te": "డేటా డిగ్రేడెడ్", "mr": "माहिती कम गुणवत्ता",
    },
}


# ---------------------------------------------------------------------------
# Sentence Templates per Language
# ---------------------------------------------------------------------------

SENTENCE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "recommendation_summary": {
        "en": "Based on our AI analysis, the recommendation for {ticker} ({company}) is **{recommendation}** with a confidence of {confidence}%.",
        "hi": "हमारे AI विश्लेषण के आधार पर, {ticker} ({company}) के लिए सिफारिश **{recommendation}** है, जिसका विश्वास स्तर {confidence}% है।",
        "ta": "எங்கள் AI பகுப்பாய்வின் அடிப்படையில், {ticker} ({company}) க்கான பரிந்துரை **{recommendation}**, நம்பிக்கை நிலை {confidence}%.",
        "te": "మా AI విశ్లేషణ ఆధారంగా, {ticker} ({company}) కోసం సిఫార్సు **{recommendation}**, నమ్మకం {confidence}%.",
        "mr": "आमच्या AI विश्लेषणाच्या आधारे, {ticker} ({company}) साठी शिफारस **{recommendation}** आहे, विश्वास स्तर {confidence}%.",
    },
    "portfolio_value": {
        "en": "Your portfolio of Rs {capital:,.0f} is projected to grow to Rs {projected:,.0f} over {months} months.",
        "hi": "आपका {capital:,.0f} रुपये का पोर्टफोलियो {months} महीनों में {projected:,.0f} रुपये तक बढ़ने का अनुमान है।",
        "ta": "உங்கள் Rs {capital:,.0f} போர்ட்ஃபோலியோ {months} மாதங்களில் Rs {projected:,.0f} ஆக வளரும் என எதிர்பார்க்கப்படுகிறது.",
        "te": "మీ Rs {capital:,.0f} పోర్ట్‌ఫోలియో {months} నెలల్లో Rs {projected:,.0f} కి పెరుగుతుందని అంచనా.",
        "mr": "तुमचे {capital:,.0f} रुपयांचे पोर्टफोलिओ {months} महिन्यांत {projected:,.0f} रुपयांपर्यंत वाढण्याचा अंदाज आहे.",
    },
    "crash_warning": {
        "en": "WARNING: Market crash risk is {level}. Consider defensive positioning.",
        "hi": "चेतावनी: बाज़ार में गिरावट का जोखिम {level} है। रक्षात्मक स्थिति अपनाने पर विचार करें।",
        "ta": "எச்சரிக்கை: சந்தை சரிவு ஆபத்து {level}. பாதுகாப்பு நிலையை கருத்தில் கொள்ளுங்கள்.",
        "te": "హెచ్చరిక: మార్కెట్ క్రాష్ ప్రమాదం {level}. రక్షణాత్మక వైఖరిని పరిగణించండి.",
        "mr": "सूचना: बाजारात मोठ्या घसरणीचा धोका {level} आहे. संरक्षणात्मक स्थान घ्यायला विचार करा.",
    },
    "fraud_alert": {
        "en": "FRAUD ALERT: This stock tip shows {level} risk indicators of potential pump-and-dump scheme.",
        "hi": "धोखाधड़ी अलर्ट: इस स्टॉक टिप में पंप-एंड-डंप योजना के {level} जोखिम संकेतक हैं।",
        "ta": "மோசடி எச்சரிக்கை: இந்த பங்கு டிப் பம்ப்-அண்ட்-டம்ப் திட்டத்தின் {level} ஆபத்து குறிகாட்டிகளைக் காட்டுகிறது.",
        "te": "మోసం హెచ్చరిక: ఈ స్టాక్ టిప్ పంప్-అండ్-డంప్ స్కీమ్ యొక్క {level} ప్రమాద సూచికలను చూపిస్తుంది.",
        "mr": "फसवणूक सूचना: या स्टॉक टिपमध्ये पंप-अँड-डम्प योजनेचे {level} जोखिम सूचक आहेत.",
    },
}


# ---------------------------------------------------------------------------
# Translation Engine
# ---------------------------------------------------------------------------

@dataclass
class LocalizedSummary:
    """Localized investment summary in a specific language."""
    language_code: str
    language_name: str
    recommendation_text: str = ""
    portfolio_text: str = ""
    crash_warning_text: str = ""
    fraud_alert_text: str = ""
    tts_script: str = ""
    full_summary: str = ""


def translate_recommendation(
    ticker: str,
    company: str,
    recommendation: str,
    confidence: float,
    language: str = "en",
) -> str:
    """Translate a recommendation summary into the target language."""
    template = SENTENCE_TEMPLATES.get("recommendation_summary", {}).get(language)
    if not template:
        template = SENTENCE_TEMPLATES["recommendation_summary"]["en"]

    rec_translation = TERM_TRANSLATIONS.get(recommendation, {}).get(language, recommendation)
    return template.format(
        ticker=ticker,
        company=company,
        recommendation=rec_translation,
        confidence=round(confidence, 1),
    )


def translate_portfolio_summary(
    capital: float,
    projected: float,
    months: int,
    language: str = "en",
) -> str:
    """Translate portfolio value projection."""
    template = SENTENCE_TEMPLATES.get("portfolio_value", {}).get(language)
    if not template:
        template = SENTENCE_TEMPLATES["portfolio_value"]["en"]
    return template.format(capital=capital, projected=projected, months=months)


def translate_crash_warning(level: str, language: str = "en") -> str:
    """Translate crash warning."""
    template = SENTENCE_TEMPLATES.get("crash_warning", {}).get(language)
    if not template:
        template = SENTENCE_TEMPLATES["crash_warning"]["en"]
    level_translation = TERM_TRANSLATIONS.get(level, {}).get(language, level)
    return template.format(level=level_translation)


def translate_fraud_alert(level: str, language: str = "en") -> str:
    """Translate fraud alert."""
    template = SENTENCE_TEMPLATES.get("fraud_alert", {}).get(language)
    if not template:
        template = SENTENCE_TEMPLATES["fraud_alert"]["en"]
    return template.format(level=level)


def generate_full_localized_summary(
    ticker: str = "",
    company: str = "",
    recommendation: str = "HOLD",
    confidence: float = 50.0,
    capital: float = 0.0,
    projected: float = 0.0,
    months: int = 12,
    crash_level: str = "LOW",
    fraud_level: str = "LOW",
    language: str = "en",
) -> LocalizedSummary:
    """
    Generate a complete localized summary including all components
    and a TTS-ready audio script.
    """
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")

    summary = LocalizedSummary(language_code=language, language_name=lang_name)

    # Recommendation
    if ticker and company:
        summary.recommendation_text = translate_recommendation(
            ticker, company, recommendation, confidence, language
        )

    # Portfolio
    if capital > 0:
        summary.portfolio_text = translate_portfolio_summary(
            capital, projected, months, language
        )

    # Crash warning
    if crash_level and crash_level != "LOW":
        summary.crash_warning_text = translate_crash_warning(crash_level, language)

    # Fraud alert
    if fraud_level and fraud_level not in ("LOW", "MEDIUM"):
        summary.fraud_alert_text = translate_fraud_alert(fraud_level, language)

    # Build full summary
    parts = []
    if summary.recommendation_text:
        parts.append(summary.recommendation_text)
    if summary.portfolio_text:
        parts.append(summary.portfolio_text)
    if summary.crash_warning_text:
        parts.append(summary.crash_warning_text)
    if summary.fraud_alert_text:
        parts.append(summary.fraud_alert_text)
    summary.full_summary = "\n\n".join(parts) if parts else f"Summary available in {lang_name}."

    # TTS script (simplified, natural-sounding)
    summary.tts_script = _build_tts_script(
        ticker, company, recommendation, confidence, crash_level, language
    )

    return summary


def _build_tts_script(
    ticker: str, company: str, recommendation: str,
    confidence: float, crash_level: str, language: str,
) -> str:
    """
    Build a text-to-speech ready audio script.
    The script is designed to sound natural when read aloud.
    """
    rec = TERM_TRANSLATIONS.get(recommendation, {}).get(language, recommendation)

    if language == "hi":
        script = (
            f"नमस्ते। {company} जिसका टिकर {ticker} है, उसके लिए हमारी AI-संचालित विश्लेषण "
            f"सिफारिश {rec} है। हमारा विश्वास स्तर {confidence:.0f} प्रतिशत है। "
        )
        if crash_level not in ("LOW", "MODERATE"):
            script += f"बाज़ार में गिरावट का जोखिम {crash_level} है, कृपया सावधान रहें। "
        script += "कृपया ध्यान दें, यह केवल शैक्षिक जानकारी है, निवेश सलाह नहीं।"
    elif language == "ta":
        script = (
            f"வணக்கம். {company} ({ticker}) க்கான எங்கள் AI பகுப்பாய்வு "
            f"பரிந்துரை {rec}. நம்பிக்கை நிலை {confidence:.0f} சதவீதம். "
        )
        if crash_level not in ("LOW", "MODERATE"):
            script += f"சந்தை சரிவு ஆபத்து {crash_level}. கவனமாக இருங்கள். "
        script += "இது கல்வி நோக்கத்திற்கு மட்டுமே. முதலீட்டு ஆலோசனை அல்ல."
    elif language == "te":
        script = (
            f"నమస్కారం. {company} ({ticker}) కోసం మా AI విశ్లేషణ "
            f"సిఫార్సు {rec}. నమ్మకం {confidence:.0f} శాతం. "
        )
        if crash_level not in ("LOW", "MODERATE"):
            script += f"మార్కెట్ క్రాష్ ప్రమాదం {crash_level}. జాగ్రత్తగా ఉండండి. "
        script += "ఇది విద్యా సమాచారం మాత్రమే. పెట్టుబడి సలహా కాదు."
    elif language == "mr":
        script = (
            f"नमस्कार. {company} ({ticker}) साठी आमचे AI-चालित विश्लेषण "
            f"शिफारस {rec} आहे. विश्वासाची पातळी {confidence:.0f} टक्के. "
        )
        if crash_level not in ("LOW", "MODERATE"):
            script += f"बाजारात घसरणीचा धोका {crash_level} आहे. कृपया सावधान रहा. "
        script += "हे फक्त शैक्षणिक माहिती आहे, गुंतवणूक सल्ला नाही."
    else:
        script = (
            f"Hello. Based on our AI analysis of {company} ({ticker}), "
            f"the recommendation is {rec} with a confidence of {confidence:.0f} percent. "
        )
        if crash_level not in ("LOW", "MODERATE"):
            script += f"Market crash risk is {crash_level}. Please exercise caution. "
        script += (
            "Please note, this is educational information only, "
            "not personalized investment advice."
        )

    return script


# ---------------------------------------------------------------------------
# Convenience: Generate All Languages
# ---------------------------------------------------------------------------

def generate_all_language_summaries(
    ticker: str = "",
    company: str = "",
    recommendation: str = "HOLD",
    confidence: float = 50.0,
    capital: float = 0.0,
    projected: float = 0.0,
    months: int = 12,
    crash_level: str = "LOW",
    fraud_level: str = "LOW",
) -> Dict[str, LocalizedSummary]:
    """Generate localized summaries for all supported languages."""
    results = {}
    for code in SUPPORTED_LANGUAGES:
        results[code] = generate_full_localized_summary(
            ticker=ticker, company=company,
            recommendation=recommendation, confidence=confidence,
            capital=capital, projected=projected, months=months,
            crash_level=crash_level, fraud_level=fraud_level,
            language=code,
        )
    return results
