"""
Response generation for the FSM-based AI Interviewer.

This module generates bot responses based on FSM state.
Key principle: FSM decides the phase, this module fills the sentence.

The responses are templates that can be customized by an LLM
for more natural conversation while maintaining FSM authority.
"""

from typing import Optional, Dict

from .enums import Phase, QuestionType, UserProfile, Product
from .state import ConversationState, IntentSignals


# ============================================================================
# RESPONSE TEMPLATES (using dict for O(1) lookup)
# ============================================================================

_WELCOME_RESPONSE = """👋 Hi there! Welcome to Amazon Seller Support.

I'm here to help you explore opportunities in e-commerce and find the right path for your goals.

Let me ask you a quick question to understand you better..."""

_INTENT_QUESTIONS: Dict[int, str] = {
    0: "What brings you here today? Are you looking at Amazon selling for side income, building a long-term business, or just exploring your options?",
    1: "I'd love to understand better - are you thinking of this as extra income alongside your current work, or as a full-time business opportunity?",
}

_EXPERIENCE_QUESTIONS: Dict[str, str] = {
    'default': "Great! Now, have you sold on Amazon before, or would this be your first time?",
    'follow_up': "Are you completely new to e-commerce, or do you have some selling experience already?",
}

_BUDGET_QUESTIONS: Dict[int, str] = {
    0: "To recommend the best program for you, I need to understand your investment capacity. How much are you comfortable investing to start your Amazon business? (You can share a rough range)",
    1: "No worries if you're not sure about exact numbers. Would you say your budget is:\n• Under ₹1 lakh\n• ₹1-4 lakhs\n• ₹4-7 lakhs\n• Above ₹7 lakhs",
}

_PRODUCT_PITCHES: Dict[Product, str] = {
    Product.FREE_CONTENT: """Based on what you've shared, I'd recommend starting with our **Free Learning Resources**.

These include:
✅ Video tutorials on Amazon selling basics
✅ PDF guides on product research
✅ Access to our community group

This is perfect for learning the fundamentals before making any investment.""",

    Product.MASTERCLASS: """Based on your profile, our **Masterclass Program** would be perfect for you!

What you get:
✅ 20+ hours of structured video content
✅ Live Q&A sessions
✅ Product research templates
✅ 3-month community access

This program has helped 500+ sellers launch their first product successfully.""",

    Product.PPI: """I'd recommend our **PPI (Private Product Incubator)** program for you!

What's included:
✅ Personalized product selection guidance
✅ Supplier negotiation support
✅ Listing optimization
✅ 6-month mentorship
✅ Money-back guarantee on results

This is our most popular program with a 78% success rate.""",

    Product.MENTORSHIP: """Based on your experience and investment capacity, our **1-on-1 Mentorship** program is ideal!

You'll get:
✅ Personal mentor assigned to you
✅ Weekly strategy calls
✅ Complete done-with-you service
✅ Direct supplier introductions
✅ Priority support channel
✅ 12-month partnership

This is our premium program for serious entrepreneurs.""",
}

_CLOSING_PROMPTS: Dict[str, str] = {
    'initial': "Would you like to move forward with this? I can share more details or schedule a quick call to discuss.",
    'after_yes': "That's great! Let me share the next steps. Would you prefer:\n1. A quick 15-min call to discuss details\n2. A PDF with complete program information\n3. Direct registration link",
    'objection_handled': "I understand your concern. Does that address your question? Are you ready to take the next step?",
    'downgrade': "I understand. Let me share an alternative that might work better for your current situation...",
}

_EXIT_RESPONSES: Dict[str, str] = {
    'soft': "No problem! I'll be here whenever you're ready. Feel free to message back anytime. Have a great day! 👋",
    'hard': "I've noted your preference. You won't receive any more messages from us. Take care!",
}

_RAG_TRANSITION = "Coming back to where we were — "


# ============================================================================
# RESPONSE GENERATION FUNCTIONS
# ============================================================================

def generate_welcome_response() -> str:
    """Generate welcome message."""
    return _WELCOME_RESPONSE


def generate_intent_question(clarification_count: int) -> str:
    """Generate intent question based on retry count."""
    return _INTENT_QUESTIONS.get(clarification_count, _INTENT_QUESTIONS[0])


def generate_experience_question(is_follow_up: bool = False) -> str:
    """Generate experience question."""
    key = 'follow_up' if is_follow_up else 'default'
    return _EXPERIENCE_QUESTIONS[key]


def generate_budget_question(retry_count: int) -> str:
    """Generate budget question based on retry count."""
    return _BUDGET_QUESTIONS.get(retry_count, _BUDGET_QUESTIONS[1])


def generate_product_pitch(product: Product, user_profile: UserProfile) -> str:
    """Generate product pitch based on recommended product and user profile."""
    base_pitch = _PRODUCT_PITCHES.get(product, _PRODUCT_PITCHES[Product.FREE_CONTENT])
    
    # Customize based on user profile
    if user_profile == UserProfile.NEWBIE:
        base_pitch += "\n\nSince you're new to Amazon selling, this will give you a solid foundation."
    elif user_profile == UserProfile.SELLER:
        base_pitch += "\n\nWith your existing experience, this will help you scale to the next level."
    
    return base_pitch


def generate_closing_prompt(
    state: ConversationState,
    after_yes: bool = False,
    after_objection: bool = False
) -> str:
    """Generate closing prompt based on context."""
    if after_yes:
        return _CLOSING_PROMPTS['after_yes']
    if after_objection:
        return _CLOSING_PROMPTS['objection_handled']
    return _CLOSING_PROMPTS['initial']


def generate_exit_response(is_hard_exit: bool) -> str:
    """Generate exit response."""
    return _EXIT_RESPONSES['hard'] if is_hard_exit else _EXIT_RESPONSES['soft']


def generate_rag_transition(last_question_type: QuestionType) -> str:
    """
    Generate transition after RAG answer.
    
    This bridges the RAG answer back to the FSM conversation.
    """
    question_hints = {
        QuestionType.INTENT: "are you looking at this for side income or long-term?",
        QuestionType.EXPERIENCE: "are you new to Amazon selling or do you have experience?",
        QuestionType.BUDGET: "what's your comfortable investment range?",
        QuestionType.CLOSING: "are you ready to take the next step?",
    }
    
    hint = question_hints.get(last_question_type, "")
    if hint:
        return _RAG_TRANSITION + hint
    return ""


# ============================================================================
# MAIN RESPONSE GENERATION
# ============================================================================

def generate_response(
    state: ConversationState,
    signals: IntentSignals,
    rag_answer: Optional[str] = None
) -> str:
    """
    Generate bot response based on current state.
    
    This is the main entry point for response generation.
    The FSM has already determined the phase - this just fills the words.
    
    Args:
        state: Current conversation state (after FSM transition)
        signals: Extracted intent signals
        rag_answer: Optional RAG-generated answer for education/objection
    
    Returns:
        Complete bot response string
    """
    response_parts = []
    
    # Handle RAG answer if present
    if rag_answer:
        response_parts.append(rag_answer)
        response_parts.append("")  # Empty line for separation
        transition = generate_rag_transition(state.last_question_type)
        if transition:
            response_parts.append(transition)
            return "\n".join(response_parts)
    
    # Handle exit states
    if state.phase == Phase.TERMINATED:
        return generate_exit_response(is_hard_exit=True)
    
    if state.phase == Phase.PAUSED:
        return generate_exit_response(is_hard_exit=False)
    
    # Generate phase-specific response
    if state.phase == Phase.INTENT:
        if not state.welcome_done:
            # This shouldn't happen due to FSM, but handle gracefully
            response_parts.append(generate_welcome_response())
            response_parts.append("")
        response_parts.append(generate_intent_question(state.intent_clarification_count))
    
    elif state.phase == Phase.EXPERIENCE:
        response_parts.append(generate_experience_question())
    
    elif state.phase == Phase.BUDGET:
        response_parts.append(generate_budget_question(state.budget_retry_count))
    
    elif state.phase == Phase.RECOMMENDATION:
        product = state.recommended_product or Product.FREE_CONTENT
        response_parts.append(generate_product_pitch(product, state.user_profile))
        response_parts.append("")
        response_parts.append(generate_closing_prompt(state))
    
    elif state.phase == Phase.CLOSING:
        if signals.is_yes:
            response_parts.append(generate_closing_prompt(state, after_yes=True))
        elif signals.has_objection and rag_answer:
            response_parts.append(generate_closing_prompt(state, after_objection=True))
        else:
            response_parts.append(generate_closing_prompt(state))
    
    return "\n".join(response_parts)
