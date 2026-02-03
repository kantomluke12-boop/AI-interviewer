"""
Main Chatbot class for the FSM-based AI Interviewer.

This module provides the high-level API for processing messages
and generating responses. It orchestrates:
- Text preprocessing
- FSM state transitions
- RAG integration (stubbed for now)
- Response generation

The design ensures:
- FSM is authoritative
- LLM is subordinate (only fills sentences)
- User text never directly changes phase
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import asdict
import json

from .enums import Phase
from .state import ConversationState, IntentSignals, create_initial_state
from .preprocessor import preprocess_message
from .fsm import apply_fsm_transition
from .response_generator import generate_response, generate_welcome_response


class Chatbot:
    """
    FSM-based chatbot for sales qualification.
    
    This class maintains conversation state and processes messages
    through the FSM pipeline.
    
    Usage:
        bot = Chatbot()
        response = bot.process_message("Hello")
        response = bot.process_message("I want side income")
        ...
    
    The chatbot is stateful - each instance maintains one conversation.
    For multiple conversations, create multiple instances.
    """
    
    def __init__(self, state: Optional[ConversationState] = None):
        """
        Initialize chatbot with optional existing state.
        
        Args:
            state: Optional pre-existing state for resuming conversations
        """
        self._state = state or create_initial_state()
        self._message_count = 0
        self._history: List[Dict[str, Any]] = []
    
    @property
    def state(self) -> ConversationState:
        """Get current conversation state (read-only access)."""
        return self._state
    
    @property
    def phase(self) -> Phase:
        """Get current conversation phase."""
        return self._state.phase
    
    @property
    def is_active(self) -> bool:
        """Check if conversation is still active (not terminated or paused)."""
        return self._state.phase not in (Phase.TERMINATED, Phase.PAUSED)
    
    @property
    def message_count(self) -> int:
        """Get total number of messages processed."""
        return self._message_count
    
    def get_welcome_message(self) -> str:
        """
        Get the initial welcome message.
        
        Call this when starting a new conversation.
        """
        if self._state.welcome_done:
            # Already welcomed, don't re-welcome
            return ""
        
        # Process empty message to trigger welcome
        return self.process_message("")
    
    def process_message(self, raw_message: str) -> str:
        """
        Process incoming user message and generate response.
        
        This is the main entry point for the chatbot.
        
        Flow:
        1. Preprocess message (normalize, extract signals)
        2. Apply FSM transition
        3. Invoke RAG if needed
        4. Generate response
        
        Args:
            raw_message: Raw user input text
        
        Returns:
            Bot response string
        """
        self._message_count += 1
        
        # Step 1: Preprocess message
        normalized_text, signals = preprocess_message(raw_message)
        
        # Log for debugging
        self._log_message('user', raw_message, signals)
        
        # Special case: Initial welcome (empty or first message)
        if not self._state.welcome_done:
            from .fsm import _handle_welcome_phase
            self._state = _handle_welcome_phase(self._state, signals)
            response = generate_welcome_response()
            response += "\n\n"
            from .response_generator import generate_intent_question
            response += generate_intent_question(0)
            self._log_message('bot', response)
            return response
        
        # Step 2: Apply FSM transition
        new_state, needs_rag = apply_fsm_transition(self._state, signals)
        self._state = new_state
        
        # Step 3: Invoke RAG if needed (stubbed for now)
        rag_answer = None
        if needs_rag:
            rag_answer = self._get_rag_answer(signals)
        
        # Step 4: Generate response
        response = generate_response(self._state, signals, rag_answer)
        
        self._log_message('bot', response)
        return response
    
    def _get_rag_answer(self, signals: IntentSignals) -> Optional[str]:
        """
        Get RAG-generated answer for education/objection.
        
        This is a stub implementation. In production, this would:
        1. Query a vector database with the user's question
        2. Retrieve relevant knowledge chunks
        3. Generate a contextual answer
        
        For now, returns canned responses based on detected intent.
        """
        if signals.has_education_request:
            return self._get_education_answer(signals.normalized_message)
        if signals.has_objection:
            return self._get_objection_answer(signals.normalized_message)
        return None
    
    def _get_education_answer(self, message: str) -> str:
        """Get education answer based on message content."""
        # Simple keyword matching for common questions
        education_responses = {
            'what is amazon': (
                "Amazon is the world's largest e-commerce marketplace. "
                "As a seller, you can list products on Amazon and they handle "
                "customer discovery, payments, and optionally fulfillment through FBA."
            ),
            'explain business': (
                "The Amazon selling business model works like this:\n"
                "1. Find a product to sell\n"
                "2. Source it from suppliers (often from China)\n"
                "3. List it on Amazon\n"
                "4. Amazon brings customers to your listing\n"
                "5. You fulfill orders (or use FBA)\n"
                "6. Keep the profit margin"
            ),
            'how it works': (
                "Amazon FBA (Fulfillment by Amazon) works by:\n"
                "• You send your products to Amazon's warehouse\n"
                "• When someone orders, Amazon ships it\n"
                "• They handle returns and customer service\n"
                "• You pay a fee for this service"
            ),
        }
        
        for keyword, response in education_responses.items():
            if keyword in message:
                return response
        
        # Default response
        return (
            "Amazon selling allows anyone to start an e-commerce business "
            "by leveraging Amazon's marketplace, customer base, and logistics."
        )
    
    def _get_objection_answer(self, message: str) -> str:
        """Get objection handling answer based on message content."""
        objection_responses = {
            'expensive': (
                "I understand cost is a concern. Our programs are designed to deliver "
                "ROI within 3-6 months. Many of our students have recovered their "
                "investment in the first batch of products."
            ),
            'scam': (
                "That's a valid concern in today's world. We're a registered company "
                "with 1000+ verified student testimonials. We also offer a satisfaction "
                "guarantee on our premium programs."
            ),
            'risky': (
                "Every business has some risk, but we minimize it by:\n"
                "• Teaching proven product selection criteria\n"
                "• Providing market validation tools\n"
                "• Offering mentorship to avoid common mistakes\n"
                "• Starting with small test batches"
            ),
            'loan': (
                "We don't recommend taking loans to start. It's better to start "
                "with what you can comfortably invest. Our free resources are a "
                "great way to begin if budget is limited."
            ),
            'no money': (
                "I understand. We have free resources that can help you learn "
                "the basics and potentially start with very minimal investment. "
                "Would you like me to share those?"
            ),
        }
        
        for keyword, response in objection_responses.items():
            if keyword in message:
                return response
        
        # Default response
        return (
            "I appreciate you sharing your concern. Let me address that directly "
            "so you can make an informed decision."
        )
    
    def _log_message(
        self,
        role: str,
        content: str,
        signals: Optional[IntentSignals] = None
    ):
        """Log message to history for debugging."""
        entry = {
            'role': role,
            'content': content,
            'message_number': self._message_count,
            'phase': self._state.phase.name,
        }
        if signals:
            entry['signals'] = {
                'is_yes': signals.is_yes,
                'is_no': signals.is_no,
                'is_neutral': signals.is_neutral,
                'is_soft_exit': signals.is_soft_exit,
                'is_hard_exit': signals.is_hard_exit,
                'has_education_request': signals.has_education_request,
                'has_objection': signals.has_objection,
                'normalized_message': signals.normalized_message,
            }
        self._history.append(entry)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self._history.copy()
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state as dictionary (for persistence)."""
        state_dict = asdict(self._state)
        # Convert enums to their names for JSON serialization
        state_dict['phase'] = self._state.phase.name
        state_dict['last_question_type'] = self._state.last_question_type.name
        state_dict['user_profile'] = self._state.user_profile.name
        if self._state.recommended_product:
            state_dict['recommended_product'] = self._state.recommended_product.name
        return state_dict
    
    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> 'Chatbot':
        """Create chatbot from exported state dictionary."""
        # Convert enum values from strings
        from .enums import Phase, QuestionType, UserProfile, Product
        
        # Handle both string and enum values
        if isinstance(state_dict['phase'], str):
            state_dict['phase'] = Phase[state_dict['phase']]
        if isinstance(state_dict['last_question_type'], str):
            state_dict['last_question_type'] = QuestionType[state_dict['last_question_type']]
        if isinstance(state_dict['user_profile'], str):
            state_dict['user_profile'] = UserProfile[state_dict['user_profile']]
        
        if state_dict.get('recommended_product') and isinstance(state_dict['recommended_product'], str):
            state_dict['recommended_product'] = Product[state_dict['recommended_product']]
        
        state = ConversationState(**state_dict)
        return cls(state=state)
    
    def reset(self):
        """Reset chatbot to initial state."""
        self._state = create_initial_state()
        self._message_count = 0
        self._history = []


def create_chatbot() -> Chatbot:
    """Factory function to create a new chatbot instance."""
    return Chatbot()
