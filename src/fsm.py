"""
FSM (Finite State Machine) transition logic for the AI Interviewer.

This module contains the authoritative state transition rules.
Key design principles:
- Phases only move FORWARD
- Welcome happens ONLY ONCE
- Education/Objections do NOT change phase
- YES/NO resolved ONLY via last_question_type

Efficiency considerations:
- Single-pass state updates
- No redundant state copies
- Early exit conditions
- Dictionary-based dispatch (O(1) vs if-else chains)
"""

from typing import Tuple, Optional, Callable

from .enums import Phase, QuestionType, UserProfile, Product
from .state import ConversationState, IntentSignals, copy_state_with_updates


# ============================================================================
# BUDGET TO PRODUCT MAPPING
# ============================================================================

def _map_budget_to_product(budget_lakhs: int) -> Product:
    """
    Map budget amount to appropriate product.
    
    <1L        → Free Content
    1–4L       → Masterclass
    4–7L       → PPI
    7L+        → Mentorship
    """
    if budget_lakhs < 1:
        return Product.FREE_CONTENT
    elif budget_lakhs < 4:
        return Product.MASTERCLASS
    elif budget_lakhs < 7:
        return Product.PPI
    else:
        return Product.MENTORSHIP


# ============================================================================
# PHASE TRANSITION HANDLERS
# ============================================================================

def _handle_welcome_phase(
    state: ConversationState,
    signals: IntentSignals
) -> ConversationState:
    """
    Handle WELCOME phase - one-time, immutable.
    
    If welcome_done is false, mark it done and move to INTENT.
    This can NEVER be re-entered.
    """
    if state.welcome_done:
        # Should never happen, but guard against it
        return copy_state_with_updates(
            state,
            phase=Phase.INTENT,
            last_question_type=QuestionType.INTENT
        )
    
    return copy_state_with_updates(
        state,
        welcome_done=True,
        phase=Phase.INTENT,
        last_question_type=QuestionType.INTENT
    )


def _handle_intent_phase(
    state: ConversationState,
    signals: IntentSignals
) -> ConversationState:
    """
    Handle INTENT phase - asking why Amazon?
    
    Expecting: side income | long term | exploring | random yes
    
    If random YES with no context: ask clarifying question (ONCE)
    On valid answer: move to EXPERIENCE
    """
    # Check for valid intent keywords
    if signals.intent_type == 'valid_intent' or signals.normalized_message:
        # If just "yes" with no context, ask for clarification (once)
        if signals.is_yes and not signals.intent_type and state.intent_clarification_count < 1:
            return copy_state_with_updates(
                state,
                intent_clarification_count=state.intent_clarification_count + 1
            )
        
        # Accept any substantive response
        return copy_state_with_updates(
            state,
            intent_confirmed=True,
            phase=Phase.EXPERIENCE,
            last_question_type=QuestionType.EXPERIENCE
        )
    
    return state


def _handle_experience_phase(
    state: ConversationState,
    signals: IntentSignals
) -> ConversationState:
    """
    Handle EXPERIENCE phase - new or existing seller?
    
    Look for experience keywords to determine user profile.
    Move to BUDGET on confirmation.
    """
    updates = {}
    
    # Detect user profile from keywords
    if signals.experience_keyword == 'newbie':
        updates['user_profile'] = UserProfile.NEWBIE
    elif signals.experience_keyword == 'seller':
        updates['user_profile'] = UserProfile.SELLER
    elif signals.is_yes and state.user_profile == UserProfile.UNKNOWN:
        # If yes without context, assume newbie (safer default)
        updates['user_profile'] = UserProfile.NEWBIE
    
    # Check if we can confirm experience
    if updates.get('user_profile') or state.user_profile != UserProfile.UNKNOWN:
        # User has provided experience info, move to budget
        if 'user_profile' not in updates:
            updates['user_profile'] = state.user_profile
        
        updates['experience_confirmed'] = True
        updates['phase'] = Phase.BUDGET
        updates['last_question_type'] = QuestionType.BUDGET
    elif signals.is_yes or signals.is_no:
        # Resolve via last_question_type if it was EXPERIENCE
        if state.last_question_type == QuestionType.EXPERIENCE:
            updates['experience_confirmed'] = True
            updates['user_profile'] = UserProfile.NEWBIE if signals.is_yes else UserProfile.SELLER
            updates['phase'] = Phase.BUDGET
            updates['last_question_type'] = QuestionType.BUDGET
    
    if updates:
        return copy_state_with_updates(state, **updates)
    return state


def _handle_budget_phase(
    state: ConversationState,
    signals: IntentSignals
) -> ConversationState:
    """
    Handle BUDGET phase - investment capacity.
    
    - If budget already stored: SKIP (don't re-ask)
    - If user gives number: store and move to RECOMMENDATION
    - If YES/NO: ask ranges (max 1 retry)
    - If avoids twice: assume LOWEST SAFE TIER
    """
    # If budget already confirmed, don't re-process
    if state.budget_confirmed:
        return copy_state_with_updates(
            state,
            phase=Phase.RECOMMENDATION,
            last_question_type=QuestionType.NONE
        )
    
    updates = {}
    
    # User provided budget value
    if signals.budget_value is not None:
        updates['budget_amount'] = signals.budget_value
        updates['budget_confirmed'] = True
        updates['phase'] = Phase.RECOMMENDATION
        updates['last_question_type'] = QuestionType.NONE
    elif signals.is_yes or signals.is_no:
        # User avoiding direct answer
        if state.budget_retry_count >= 1:
            # Assume lowest safe tier after 2 attempts
            updates['budget_amount'] = 0
            updates['budget_confirmed'] = True
            updates['phase'] = Phase.RECOMMENDATION
            updates['last_question_type'] = QuestionType.NONE
        else:
            updates['budget_retry_count'] = state.budget_retry_count + 1
    
    if updates:
        return copy_state_with_updates(state, **updates)
    return state


def _handle_recommendation_phase(
    state: ConversationState,
    signals: IntentSignals
) -> ConversationState:
    """
    Handle RECOMMENDATION phase - product pitch.
    
    Map budget to product and move to CLOSING.
    Never re-pitch the same product.
    """
    if state.product_pitched:
        # Already pitched, move to closing
        return copy_state_with_updates(
            state,
            phase=Phase.CLOSING,
            last_question_type=QuestionType.CLOSING
        )
    
    # Map budget to product
    budget = state.budget_amount if state.budget_amount is not None else 0
    product = _map_budget_to_product(budget)
    
    return copy_state_with_updates(
        state,
        recommended_product=product,
        product_pitched=True,
        phase=Phase.CLOSING,
        last_question_type=QuestionType.CLOSING
    )


def _handle_closing_phase(
    state: ConversationState,
    signals: IntentSignals
) -> ConversationState:
    """
    Handle CLOSING phase - final conversion.
    
    - YES: proceed to next action (call/pdf/payment)
    - Objection: RAG answer, return to closing
    - Silence/maybe: soft exit
    - NO: downgrade option or free resource
    """
    if signals.is_yes:
        # User is ready to proceed - state stays in CLOSING
        # The response generator will handle the next action
        return state
    
    if signals.is_no:
        # Offer downgrade or free resource
        # If already at free content, soft exit
        if state.recommended_product == Product.FREE_CONTENT:
            return copy_state_with_updates(state, soft_exit=True, phase=Phase.PAUSED)
        
        # Downgrade product
        current_product = state.recommended_product
        downgrade_map = {
            Product.MENTORSHIP: Product.PPI,
            Product.PPI: Product.MASTERCLASS,
            Product.MASTERCLASS: Product.FREE_CONTENT,
        }
        new_product = downgrade_map.get(current_product, Product.FREE_CONTENT)
        return copy_state_with_updates(state, recommended_product=new_product)
    
    if signals.is_soft_exit:
        return copy_state_with_updates(state, soft_exit=True, phase=Phase.PAUSED)
    
    # Objections are handled by RAG, state doesn't change
    return state


# ============================================================================
# PHASE HANDLER DISPATCH TABLE (O(1) lookup)
# ============================================================================

_PHASE_HANDLERS: dict[Phase, Callable[[ConversationState, IntentSignals], ConversationState]] = {
    Phase.WELCOME: _handle_welcome_phase,
    Phase.INTENT: _handle_intent_phase,
    Phase.EXPERIENCE: _handle_experience_phase,
    Phase.BUDGET: _handle_budget_phase,
    Phase.RECOMMENDATION: _handle_recommendation_phase,
    Phase.CLOSING: _handle_closing_phase,
}


# ============================================================================
# MAIN FSM TRANSITION FUNCTION
# ============================================================================

def apply_fsm_transition(
    state: ConversationState,
    signals: IntentSignals
) -> Tuple[ConversationState, bool]:
    """
    Apply FSM transition rules based on current state and intent signals.
    
    This is the MASTER FUNCTION that implements the flow:
    1. Check hard exit → TERMINATE
    2. Check soft exit → PAUSE
    3. Check education/objection → RAG (no phase change)
    4. Resolve YES/NO via last_question_type
    5. Apply phase-specific transition
    
    Returns:
        Tuple of (new_state, needs_rag_answer)
    
    The needs_rag_answer flag indicates if RAG should be invoked
    for education/objection before generating the response.
    """
    needs_rag = False
    
    # Step 1: Check hard exit (highest priority)
    if signals.is_hard_exit:
        return copy_state_with_updates(
            state,
            hard_exit=True,
            phase=Phase.TERMINATED
        ), False
    
    # Step 2: Check soft exit
    if signals.is_soft_exit and not state.soft_exit:
        return copy_state_with_updates(
            state,
            soft_exit=True,
            phase=Phase.PAUSED
        ), False
    
    # Step 3: Check if already terminated or paused
    if state.phase in (Phase.TERMINATED, Phase.PAUSED):
        return state, False
    
    # Step 4: Check education/objection request (RAG side-channel)
    if signals.has_education_request or signals.has_objection:
        needs_rag = True
        # Note: We still process the FSM transition after RAG
        # but the phase doesn't change due to education/objection
    
    # Step 5: Get phase handler and apply transition
    handler = _PHASE_HANDLERS.get(state.phase)
    if handler:
        new_state = handler(state, signals)
    else:
        new_state = state
    
    return new_state, needs_rag


def get_phase_question_type(phase: Phase) -> QuestionType:
    """Get the question type for a given phase."""
    mapping = {
        Phase.INTENT: QuestionType.INTENT,
        Phase.EXPERIENCE: QuestionType.EXPERIENCE,
        Phase.BUDGET: QuestionType.BUDGET,
        Phase.CLOSING: QuestionType.CLOSING,
    }
    return mapping.get(phase, QuestionType.NONE)
