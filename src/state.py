"""
State management for the FSM-based AI Interviewer.

This module contains the authoritative state object that drives all decisions.
The state is immutable-friendly - all updates return new state instances.
"""

from dataclasses import dataclass, field
from typing import Optional

from .enums import Phase, QuestionType, UserProfile, Product


@dataclass
class ConversationState:
    """
    Master state object - authoritative source of truth.
    
    All phase transitions and decisions are based on this state.
    Uses dataclass for efficiency (slotted memory layout, fast attribute access).
    """
    # Core phase tracking
    phase: Phase = Phase.WELCOME
    last_question_type: QuestionType = QuestionType.NONE
    
    # Confirmation flags (once set, never reset - forward-only)
    welcome_done: bool = False
    intent_confirmed: bool = False
    experience_confirmed: bool = False
    budget_confirmed: bool = False
    product_pitched: bool = False
    
    # User profile data
    user_profile: UserProfile = UserProfile.UNKNOWN
    budget_amount: Optional[int] = None
    recommended_product: Optional[Product] = None
    
    # Exit states
    soft_exit: bool = False
    hard_exit: bool = False
    
    # Retry counters (to prevent infinite loops)
    intent_clarification_count: int = 0
    budget_retry_count: int = 0
    
    def __post_init__(self):
        """Validate state consistency."""
        # Validate: if hard_exit is True, phase should be TERMINATED
        if self.hard_exit and self.phase != Phase.TERMINATED:
            raise ValueError(
                "Invalid state: hard_exit=True requires phase=TERMINATED. "
                "Use copy_state_with_updates to create consistent states."
            )
        # Validate: if soft_exit is True, phase should be PAUSED or TERMINATED
        if self.soft_exit and self.phase not in (Phase.PAUSED, Phase.TERMINATED):
            raise ValueError(
                "Invalid state: soft_exit=True requires phase=PAUSED or TERMINATED. "
                "Use copy_state_with_updates to create consistent states."
            )


@dataclass
class IntentSignals:
    """
    Extracted intent signals from user message.
    
    These signals are the ONLY input to FSM transition logic.
    User text never directly influences phase - only these signals do.
    """
    is_yes: bool = False
    is_no: bool = False
    is_neutral: bool = True
    
    budget_value: Optional[int] = None
    experience_keyword: Optional[str] = None
    
    has_objection: bool = False
    has_education_request: bool = False
    
    is_soft_exit: bool = False
    is_hard_exit: bool = False
    
    # Original normalized message for context
    normalized_message: str = ""
    
    # Intent type (for explicit categorization)
    intent_type: Optional[str] = None


def create_initial_state() -> ConversationState:
    """Factory function to create a fresh conversation state."""
    return ConversationState()


def copy_state_with_updates(state: ConversationState, **updates) -> ConversationState:
    """
    Create a new state with specified updates.
    
    This pattern ensures immutability and makes state transitions explicit.
    More efficient than deep copy since dataclasses are lightweight.
    """
    # Get current state as dict
    current = {
        'phase': state.phase,
        'last_question_type': state.last_question_type,
        'welcome_done': state.welcome_done,
        'intent_confirmed': state.intent_confirmed,
        'experience_confirmed': state.experience_confirmed,
        'budget_confirmed': state.budget_confirmed,
        'product_pitched': state.product_pitched,
        'user_profile': state.user_profile,
        'budget_amount': state.budget_amount,
        'recommended_product': state.recommended_product,
        'soft_exit': state.soft_exit,
        'hard_exit': state.hard_exit,
        'intent_clarification_count': state.intent_clarification_count,
        'budget_retry_count': state.budget_retry_count,
    }
    
    # Apply updates
    current.update(updates)
    
    return ConversationState(**current)
