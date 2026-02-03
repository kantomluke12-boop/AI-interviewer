"""
Enumerations for the FSM-based AI Interviewer.

Using enums instead of strings improves:
- Type safety
- Performance (enum comparison is O(1) vs string comparison O(n))
- Memory efficiency (enums are cached singletons)
"""

from enum import Enum, auto


class Phase(Enum):
    """Conversation phases - transitions are FORWARD ONLY."""
    WELCOME = auto()
    INTENT = auto()
    EXPERIENCE = auto()
    BUDGET = auto()
    RECOMMENDATION = auto()
    CLOSING = auto()
    TERMINATED = auto()
    PAUSED = auto()


class QuestionType(Enum):
    """Last question type for resolving YES/NO responses."""
    NONE = auto()
    INTENT = auto()
    EXPERIENCE = auto()
    BUDGET = auto()
    CLOSING = auto()


class UserProfile(Enum):
    """User experience profile."""
    UNKNOWN = auto()
    NEWBIE = auto()
    SELLER = auto()


class Intent(Enum):
    """Detected user intent from message."""
    YES = auto()
    NO = auto()
    NEUTRAL = auto()
    BUDGET_VALUE = auto()
    EXPERIENCE_KEYWORD = auto()
    OBJECTION = auto()
    EDUCATION_REQUEST = auto()
    SOFT_EXIT = auto()
    HARD_EXIT = auto()


class Product(Enum):
    """Available products based on budget."""
    FREE_CONTENT = auto()
    MASTERCLASS = auto()
    PPI = auto()
    MENTORSHIP = auto()
