"""
Text preprocessing pipeline for the FSM-based AI Interviewer.

This module handles:
- Text normalization (lowercase, trim, emoji removal)
- Hinglish to English translation
- Intent signal extraction

Efficiency considerations:
- Compiled regex patterns (one-time compilation cost)
- Early exit conditions (fail fast)
- Efficient set-based keyword matching
"""

import re
from typing import Optional, Set, Dict, Tuple

from .state import IntentSignals


# ============================================================================
# COMPILED REGEX PATTERNS (singleton - compiled once at module load)
# ============================================================================

# Emoji pattern - simplified to avoid overlapping ranges
# Uses a more focused approach covering common emoji ranges
_EMOJI_PATTERN = re.compile(
    r'['
    r'\U0001F600-\U0001F64F'  # emoticons
    r'\U0001F300-\U0001F5FF'  # symbols & pictographs
    r'\U0001F680-\U0001F6FF'  # transport & map symbols
    r'\U0001F1E0-\U0001F1FF'  # flags
    r'\U0001F900-\U0001F9FF'  # supplemental symbols
    r'\U00002700-\U000027BF'  # dingbats
    r'\U0000FE00-\U0000FE0F'  # variation selectors
    r'\U0000200D'              # zero width joiner
    r'\U00003030'              # wavy dash
    r'\U0001F926-\U0001F937'  # gestures
    r']+',
    flags=re.UNICODE
)

# Budget extraction pattern
_BUDGET_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:l|lakh|lac|lakhs|lacs|k|thousand)?',
    re.IGNORECASE
)

# Whitespace normalization
_WHITESPACE_PATTERN = re.compile(r'\s+')


# ============================================================================
# KEYWORD SETS (using frozenset for O(1) lookup and immutability)
# ============================================================================

# Hinglish to English mapping
_HINGLISH_MAP: Dict[str, str] = {
    'haan': 'yes',
    'ha': 'yes',
    'haa': 'yes',
    'ji': 'yes',
    'ji haan': 'yes',
    'bilkul': 'yes',
    'theek': 'ok',
    'thik': 'ok',
    'nahi': 'no',
    'nhi': 'no',
    'nahin': 'no',
    'samjhao': 'explain',
    'samjha do': 'explain',
    'batao': 'tell me',
    'bata do': 'tell me',
    'paise nahi': 'no money',
    'paisa nahi': 'no money',
    'thoda baad': 'later',
    'baad mein': 'later',
    'abhi nahi': 'not now',
    'kya': 'what',
    'kaise': 'how',
    'kyun': 'why',
    'kab': 'when',
    'kaun': 'who',
    'kitna': 'how much',
    'acha': 'ok',
    'accha': 'ok',
    'sahi': 'correct',
    'galat': 'wrong',
    'ruko': 'wait',
    'rukho': 'wait',
    'chalo': 'ok lets go',
    'chalega': 'will work',
    'nahi chalega': 'wont work',
}

# Yes indicators
_YES_KEYWORDS: frozenset = frozenset({
    'yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay', 'fine',
    'absolutely', 'definitely', 'correct', 'right', 'true',
    'agree', 'agreed', 'interested', 'go ahead', 'proceed',
    'lets do it', 'im in', 'count me in', 'sounds good',
    'will work', 'chalega', 'chalo'
})

# No indicators
_NO_KEYWORDS: frozenset = frozenset({
    'no', 'nope', 'nah', 'not', 'never', 'dont', "don't",
    'wont', "won't", 'cant', "can't", 'negative', 'disagree',
    'not interested', 'pass', 'skip', 'wrong', 'false',
    'wont work'
})

# Hard exit indicators (terminate conversation)
_HARD_EXIT_KEYWORDS: frozenset = frozenset({
    'stop', 'unsubscribe', 'remove', 'delete', 'block',
    'dont message', "don't message", 'stop messaging',
    'leave me alone', 'go away', 'quit', 'end', 'terminate'
})

# Soft exit indicators (pause conversation)
_SOFT_EXIT_KEYWORDS: frozenset = frozenset({
    'later', 'busy', 'maybe', 'not now', 'some other time',
    'thinking', 'let me think', 'will get back', 'call later',
    'contact later', 'wait', 'hold on'
})

# Education request indicators
_EDUCATION_KEYWORDS: frozenset = frozenset({
    'what is', 'explain', 'tell me', 'how does', 'how do',
    'i dont know', "i don't know", 'dont understand',
    "don't understand", 'confused', 'what does', 'meaning',
    'clarify', 'elaborate', 'details', 'more info',
    'what is amazon', 'explain business', 'how it works'
})

# Objection indicators
_OBJECTION_KEYWORDS: frozenset = frozenset({
    'expensive', 'costly', 'too much', 'cant afford',
    "can't afford", 'no money', 'broke', 'scam', 'fraud',
    'not sure', 'risky', 'risk', 'guarantee', 'refund',
    'loan', 'emi', 'installment', 'discount', 'offer'
})

# Newbie indicators
_NEWBIE_KEYWORDS: frozenset = frozenset({
    'new', 'newbie', 'beginner', 'fresher', 'fresh',
    'never sold', 'no experience', 'just starting',
    'want to learn', 'learning', 'first time', 'starter'
})

# Seller indicators
_SELLER_KEYWORDS: frozenset = frozenset({
    'selling', 'seller', 'revenue', 'sales', 'already sell',
    'have experience', 'experienced', 'been selling',
    'current business', 'my store', 'my shop', 'fba'
})

# Intent keywords
_INTENT_KEYWORDS: frozenset = frozenset({
    'side income', 'extra income', 'passive income',
    'long term', 'full time', 'career', 'business',
    'exploring', 'curious', 'checking out', 'learning'
})


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize user input text.
    
    Steps:
    1. Lowercase
    2. Strip whitespace
    3. Remove emojis
    4. Normalize whitespace
    5. Translate Hinglish
    
    This is O(n) where n is text length, with constant-time lookups.
    """
    if not text:
        return ""
    
    # Lowercase and strip
    normalized = text.lower().strip()
    
    # Remove emojis (regex is pre-compiled)
    normalized = _EMOJI_PATTERN.sub('', normalized)
    
    # Normalize whitespace
    normalized = _WHITESPACE_PATTERN.sub(' ', normalized).strip()
    
    # Hinglish translation (check each word and phrase)
    words = normalized.split()
    translated_words = []
    
    i = 0
    while i < len(words):
        # Check for two-word phrases first
        if i < len(words) - 1:
            phrase = f"{words[i]} {words[i+1]}"
            if phrase in _HINGLISH_MAP:
                translated_words.append(_HINGLISH_MAP[phrase])
                i += 2
                continue
        
        # Check single word
        word = words[i]
        if word in _HINGLISH_MAP:
            translated_words.append(_HINGLISH_MAP[word])
        else:
            translated_words.append(word)
        i += 1
    
    return ' '.join(translated_words)


def _contains_any(text: str, keywords: frozenset) -> bool:
    """
    Check if text contains any keyword from the set.
    
    Uses substring matching for multi-word keywords.
    Efficient: O(k) where k is number of keywords.
    """
    text_lower = text.lower()
    for keyword in keywords:
        if keyword in text_lower:
            return True
    return False


def _extract_budget(text: str) -> Optional[int]:
    """
    Extract budget value from text.
    
    Handles formats like:
    - "5 lakh", "5L", "5 lac"
    - "50000", "50k"
    - "5.5 lakh"
    
    Returns value in lakhs (normalized).
    """
    match = _BUDGET_PATTERN.search(text)
    if not match:
        return None
    
    value = float(match.group(1))
    suffix = match.group(0).lower()
    
    # Normalize to lakhs
    if 'k' in suffix or 'thousand' in suffix:
        value = value / 100  # Convert thousands to lakhs
    elif value > 100:  # Assume it's in raw rupees if large number
        value = value / 100000
    
    return int(value) if value >= 0 else None


def _detect_experience_keyword(text: str) -> Optional[str]:
    """Detect if user mentions experience level."""
    if _contains_any(text, _NEWBIE_KEYWORDS):
        return 'newbie'
    if _contains_any(text, _SELLER_KEYWORDS):
        return 'seller'
    return None


def extract_intent_signals(normalized_text: str) -> IntentSignals:
    """
    Extract all intent signals from normalized text.
    
    This is the ONLY function that interprets user text.
    FSM uses these signals, never raw text.
    
    Returns IntentSignals dataclass with all detected intents.
    """
    signals = IntentSignals(normalized_message=normalized_text)
    
    if not normalized_text:
        return signals
    
    # Check hard exit first (highest priority)
    if _contains_any(normalized_text, _HARD_EXIT_KEYWORDS):
        signals.is_hard_exit = True
        signals.is_neutral = False
        return signals  # Early exit - no further processing needed
    
    # Check soft exit
    if _contains_any(normalized_text, _SOFT_EXIT_KEYWORDS):
        signals.is_soft_exit = True
        signals.is_neutral = False
    
    # Check education request
    if _contains_any(normalized_text, _EDUCATION_KEYWORDS):
        signals.has_education_request = True
    
    # Check objection
    if _contains_any(normalized_text, _OBJECTION_KEYWORDS):
        signals.has_objection = True
    
    # Check yes/no (mutually exclusive - yes takes precedence if both present)
    if _contains_any(normalized_text, _YES_KEYWORDS):
        signals.is_yes = True
        signals.is_neutral = False
    elif _contains_any(normalized_text, _NO_KEYWORDS):
        signals.is_no = True
        signals.is_neutral = False
    
    # Extract budget
    signals.budget_value = _extract_budget(normalized_text)
    
    # Detect experience keyword
    signals.experience_keyword = _detect_experience_keyword(normalized_text)
    
    # Detect intent type
    if _contains_any(normalized_text, _INTENT_KEYWORDS):
        signals.intent_type = 'valid_intent'
    
    return signals


def preprocess_message(raw_text: str) -> Tuple[str, IntentSignals]:
    """
    Complete preprocessing pipeline.
    
    Returns:
        Tuple of (normalized_text, intent_signals)
    
    This is the single entry point for all text processing.
    """
    normalized = normalize_text(raw_text)
    signals = extract_intent_signals(normalized)
    return normalized, signals
