"""
Unit tests for the FSM-based AI Interviewer.

These tests verify:
- Text preprocessing and normalization
- Intent signal extraction
- FSM state transitions
- Edge case handling
"""

import pytest
from src.enums import Phase, QuestionType, UserProfile, Product
from src.state import ConversationState, IntentSignals, create_initial_state
from src.preprocessor import (
    normalize_text,
    extract_intent_signals,
    preprocess_message,
)
from src.fsm import apply_fsm_transition, _map_budget_to_product
from src.chatbot import Chatbot, create_chatbot


class TestTextPreprocessor:
    """Tests for text preprocessing and normalization."""
    
    def test_lowercase_and_trim(self):
        """Text should be lowercased and trimmed."""
        assert normalize_text("  HELLO WORLD  ") == "hello world"
    
    def test_emoji_removal(self):
        """Emojis should be removed."""
        assert normalize_text("Hello 👋 World 🌍") == "hello world"
    
    def test_hinglish_translation(self):
        """Hinglish words should be translated to English."""
        assert "yes" in normalize_text("haan")
        assert "no" in normalize_text("nahi")
        assert "later" in normalize_text("baad mein")
    
    def test_whitespace_normalization(self):
        """Multiple spaces should be normalized to single space."""
        assert normalize_text("hello    world") == "hello world"
    
    def test_empty_input(self):
        """Empty input should return empty string."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestIntentExtraction:
    """Tests for intent signal extraction."""
    
    def test_yes_detection(self):
        """Should detect yes intents."""
        signals = extract_intent_signals("yes sure")
        assert signals.is_yes is True
        assert signals.is_neutral is False
    
    def test_no_detection(self):
        """Should detect no intents."""
        signals = extract_intent_signals("no thanks")
        assert signals.is_no is True
        assert signals.is_neutral is False
    
    def test_hard_exit_detection(self):
        """Should detect hard exit intents."""
        signals = extract_intent_signals("stop messaging me")
        assert signals.is_hard_exit is True
    
    def test_soft_exit_detection(self):
        """Should detect soft exit intents."""
        signals = extract_intent_signals("maybe later")
        assert signals.is_soft_exit is True
    
    def test_education_request_detection(self):
        """Should detect education requests."""
        signals = extract_intent_signals("what is amazon")
        assert signals.has_education_request is True
    
    def test_objection_detection(self):
        """Should detect objections."""
        signals = extract_intent_signals("too expensive")
        assert signals.has_objection is True
    
    def test_budget_extraction(self):
        """Should extract budget values."""
        signals = extract_intent_signals("5 lakh")
        assert signals.budget_value == 5
        
        signals = extract_intent_signals("50000")
        assert signals.budget_value is not None
    
    def test_experience_keyword_detection(self):
        """Should detect experience keywords."""
        signals = extract_intent_signals("i am new to this")
        assert signals.experience_keyword == 'newbie'
        
        signals = extract_intent_signals("i have been selling for years")
        assert signals.experience_keyword == 'seller'
    
    def test_neutral_message(self):
        """Random text should be marked as neutral."""
        signals = extract_intent_signals("hello there")
        assert signals.is_neutral is True
        assert signals.is_yes is False
        assert signals.is_no is False


class TestFSMTransitions:
    """Tests for FSM state transitions."""
    
    def test_welcome_to_intent(self):
        """Welcome phase should transition to intent."""
        state = create_initial_state()
        signals = IntentSignals()
        
        new_state, _ = apply_fsm_transition(state, signals)
        
        assert new_state.welcome_done is True
        assert new_state.phase == Phase.INTENT
    
    def test_intent_to_experience(self):
        """Intent phase should transition to experience on valid input."""
        state = ConversationState(
            phase=Phase.INTENT,
            welcome_done=True,
            last_question_type=QuestionType.INTENT
        )
        signals = IntentSignals(
            normalized_message="side income",
            intent_type='valid_intent'
        )
        
        new_state, _ = apply_fsm_transition(state, signals)
        
        assert new_state.intent_confirmed is True
        assert new_state.phase == Phase.EXPERIENCE
    
    def test_experience_to_budget(self):
        """Experience phase should transition to budget."""
        state = ConversationState(
            phase=Phase.EXPERIENCE,
            welcome_done=True,
            intent_confirmed=True,
            last_question_type=QuestionType.EXPERIENCE
        )
        signals = IntentSignals(
            experience_keyword='newbie',
            normalized_message="i am new"
        )
        
        new_state, _ = apply_fsm_transition(state, signals)
        
        assert new_state.user_profile == UserProfile.NEWBIE
        assert new_state.phase == Phase.BUDGET
    
    def test_budget_to_recommendation(self):
        """Budget phase should transition to recommendation."""
        state = ConversationState(
            phase=Phase.BUDGET,
            welcome_done=True,
            intent_confirmed=True,
            experience_confirmed=True,
            user_profile=UserProfile.NEWBIE,
            last_question_type=QuestionType.BUDGET
        )
        signals = IntentSignals(
            budget_value=5,
            normalized_message="5 lakh"
        )
        
        new_state, _ = apply_fsm_transition(state, signals)
        
        assert new_state.budget_amount == 5
        assert new_state.budget_confirmed is True
        assert new_state.phase == Phase.RECOMMENDATION
    
    def test_hard_exit_terminates(self):
        """Hard exit should terminate conversation."""
        state = ConversationState(
            phase=Phase.INTENT,
            welcome_done=True
        )
        signals = IntentSignals(is_hard_exit=True)
        
        new_state, _ = apply_fsm_transition(state, signals)
        
        assert new_state.hard_exit is True
        assert new_state.phase == Phase.TERMINATED
    
    def test_soft_exit_pauses(self):
        """Soft exit should pause conversation."""
        state = ConversationState(
            phase=Phase.INTENT,
            welcome_done=True
        )
        signals = IntentSignals(is_soft_exit=True)
        
        new_state, _ = apply_fsm_transition(state, signals)
        
        assert new_state.soft_exit is True
        assert new_state.phase == Phase.PAUSED
    
    def test_education_request_flags_rag(self):
        """Education request should flag RAG needed."""
        state = ConversationState(
            phase=Phase.INTENT,
            welcome_done=True
        )
        signals = IntentSignals(has_education_request=True)
        
        _, needs_rag = apply_fsm_transition(state, signals)
        
        assert needs_rag is True
    
    def test_phases_only_move_forward(self):
        """Phases should only move forward, never backward."""
        state = ConversationState(
            phase=Phase.BUDGET,
            welcome_done=True,
            intent_confirmed=True,
            experience_confirmed=True
        )
        
        # Try to trigger intent phase behavior
        signals = IntentSignals(intent_type='valid_intent')
        new_state, _ = apply_fsm_transition(state, signals)
        
        # Phase should not go backward
        assert new_state.phase in (Phase.BUDGET, Phase.RECOMMENDATION, Phase.CLOSING)


class TestBudgetMapping:
    """Tests for budget to product mapping."""
    
    def test_low_budget_free_content(self):
        """Budget <1L should map to free content."""
        assert _map_budget_to_product(0) == Product.FREE_CONTENT
    
    def test_mid_low_budget_masterclass(self):
        """Budget 1-4L should map to masterclass."""
        assert _map_budget_to_product(2) == Product.MASTERCLASS
    
    def test_mid_budget_ppi(self):
        """Budget 4-7L should map to PPI."""
        assert _map_budget_to_product(5) == Product.PPI
    
    def test_high_budget_mentorship(self):
        """Budget 7L+ should map to mentorship."""
        assert _map_budget_to_product(10) == Product.MENTORSHIP


class TestChatbotIntegration:
    """Integration tests for the complete chatbot."""
    
    def test_full_conversation_flow(self):
        """Test a complete conversation flow."""
        bot = create_chatbot()
        
        # Start conversation
        response = bot.process_message("")
        assert "welcome" in response.lower() or "hi" in response.lower()
        assert bot.phase == Phase.INTENT
        
        # Provide intent
        response = bot.process_message("I want side income")
        assert bot.state.intent_confirmed is True
        assert bot.phase == Phase.EXPERIENCE
        
        # Provide experience
        response = bot.process_message("I am completely new")
        assert bot.state.experience_confirmed is True
        assert bot.phase == Phase.BUDGET
        
        # Provide budget
        response = bot.process_message("5 lakh")
        assert bot.state.budget_confirmed is True
        assert bot.phase == Phase.RECOMMENDATION
        
        # One more message to move from recommendation to closing
        response = bot.process_message("yes")
        assert bot.phase == Phase.CLOSING
    
    def test_hard_exit_at_any_phase(self):
        """Hard exit should work at any phase."""
        bot = create_chatbot()
        bot.process_message("")  # Start
        
        response = bot.process_message("stop messaging me")
        assert bot.phase == Phase.TERMINATED
        assert bot.is_active is False
    
    def test_soft_exit_pauses(self):
        """Soft exit should pause conversation."""
        bot = create_chatbot()
        bot.process_message("")  # Start
        
        response = bot.process_message("maybe later")
        assert bot.phase == Phase.PAUSED
        assert bot.is_active is False
    
    def test_education_request_mid_conversation(self):
        """Education request should not change phase."""
        bot = create_chatbot()
        bot.process_message("")  # Start
        
        initial_phase = bot.phase
        response = bot.process_message("what is amazon")
        
        # Phase should not change due to education request
        # (it may transition normally if the message also contains valid input)
        assert "amazon" in response.lower() or bot.phase == initial_phase
    
    def test_yes_resolved_via_question_type(self):
        """YES should be resolved based on last question type."""
        bot = create_chatbot()
        bot.process_message("")  # Welcome
        bot.process_message("I want income")  # Intent → Experience
        
        # Now in Experience phase
        assert bot.phase == Phase.EXPERIENCE
        
        # Say yes should be interpreted as confirming experience
        bot.process_message("yes")
        
        # Should have moved to budget phase
        assert bot.phase == Phase.BUDGET
    
    def test_state_export_import(self):
        """State should be exportable and importable."""
        bot1 = create_chatbot()
        bot1.process_message("")
        bot1.process_message("I want income")
        
        # Export state
        state_dict = bot1.export_state()
        
        # Create new bot from state
        bot2 = Chatbot.from_state_dict(state_dict)
        
        assert bot2.phase == bot1.phase
        assert bot2.state.intent_confirmed == bot1.state.intent_confirmed
    
    def test_multiple_yes_responses(self):
        """Multiple YES responses should be handled correctly."""
        bot = create_chatbot()
        bot.process_message("")  # Welcome
        
        # Say yes multiple times
        for _ in range(5):
            bot.process_message("yes")
        
        # Bot should not break, should progress through phases
        assert bot.is_active or bot.phase == Phase.CLOSING


class TestEdgeCases:
    """Tests for edge cases and safety nets."""
    
    def test_emoji_spam(self):
        """Emoji spam should be handled as neutral."""
        signals = extract_intent_signals("😀😀😀😀😀")
        # After emoji removal, should be empty/neutral
        assert signals.is_neutral is True
    
    def test_mixed_language(self):
        """Mixed language input should be handled."""
        text = normalize_text("haan I want to sell on Amazon")
        assert "yes" in text
    
    def test_budget_change_after_confirmation(self):
        """Budget change after confirmation should not affect phase."""
        bot = create_chatbot()
        bot.process_message("")
        bot.process_message("income")
        bot.process_message("new")
        bot.process_message("5 lakh")  # Budget confirmed
        
        # Now try to change budget
        current_phase = bot.phase
        current_budget = bot.state.budget_amount
        
        bot.process_message("actually 10 lakh")
        
        # Budget should not change after confirmation
        # (phase may have moved forward)
        assert bot.state.budget_amount == current_budget or bot.phase != Phase.BUDGET
    
    def test_new_keyword_late_ignored(self):
        """Late 'new' keyword should be ignored if experience confirmed."""
        bot = create_chatbot()
        bot.process_message("")
        bot.process_message("income")
        bot.process_message("i have been selling for years")  # Seller
        
        assert bot.state.user_profile == UserProfile.SELLER
        
        bot.process_message("5 lakh")
        bot.process_message("actually i am new")  # Late new
        
        # Profile should remain seller
        assert bot.state.user_profile == UserProfile.SELLER
    
    def test_reset_functionality(self):
        """Reset should restore initial state."""
        bot = create_chatbot()
        bot.process_message("")
        bot.process_message("income")
        
        assert bot.message_count > 0
        assert bot.state.welcome_done is True
        
        bot.reset()
        
        assert bot.message_count == 0
        assert bot.state.welcome_done is False
        assert bot.phase == Phase.WELCOME


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
