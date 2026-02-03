# AI Interviewer - FSM-based WhatsApp Chatbot

An unbreakable FSM (Finite State Machine) based chatbot for sales qualification, designed for WhatsApp conversations.

## Core Design Principle

> **FSM decides the phase. LLM only fills the sentence.**

User text NEVER directly changes phase. Only interpreted intent + last_question_type + stored flags do.

## Key Features

- **Forward-only phase transitions** - Phases only move forward, never backward
- **One-time welcome** - Welcome message sent exactly once
- **YES/NO resolution via context** - YES/NO answers resolved based on `last_question_type`
- **RAG side-channel** - Education/objection requests handled without changing phase
- **Exit handling** - Soft exits pause, hard exits terminate
- **Hinglish support** - Automatic translation of common Hindi phrases

## Architecture

```
USER MESSAGE
      │
      v
┌─────────────────────────────────────┐
│ PREPROCESSING PIPELINE              │
│ - Normalize text (lowercase, trim)  │
│ - Remove emojis                     │
│ - Translate Hinglish → English      │
│ - Extract intent signals            │
└─────────────────────────────────────┘
      │
      v
┌─────────────────────────────────────┐
│ FSM TRANSITION LOGIC                │
│ 1. Check hard exit → TERMINATE      │
│ 2. Check soft exit → PAUSE          │
│ 3. Check education/objection → RAG  │
│ 4. Resolve YES/NO via question type │
│ 5. Apply phase-specific transition  │
└─────────────────────────────────────┘
      │
      v
┌─────────────────────────────────────┐
│ RESPONSE GENERATION                 │
│ - Generate phase-appropriate text   │
│ - Include RAG answer if applicable  │
└─────────────────────────────────────┘
```

## Conversation Phases

1. **WELCOME** - One-time greeting (immutable, never re-entered)
2. **INTENT** - Why are you interested in Amazon selling?
3. **EXPERIENCE** - Are you new or an existing seller?
4. **BUDGET** - What's your investment capacity?
5. **RECOMMENDATION** - Product pitch based on budget mapping
6. **CLOSING** - Final conversion attempt

## Budget to Product Mapping

| Budget | Product |
|--------|---------|
| < ₹1L | Free Content |
| ₹1-4L | Masterclass |
| ₹4-7L | PPI (Private Product Incubator) |
| ₹7L+ | 1-on-1 Mentorship |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AI-interviewer

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Usage

```python
from src.chatbot import create_chatbot

# Create a new chatbot instance
bot = create_chatbot()

# Start conversation
response = bot.process_message("")
print(response)

# Continue conversation
response = bot.process_message("I want side income")
print(response)

# Check current phase
print(f"Current phase: {bot.phase.name}")

# Check if conversation is active
print(f"Is active: {bot.is_active}")
```

## State Persistence

The chatbot state can be exported and restored for long-running conversations:

```python
# Export state
state_dict = bot.export_state()

# Store state_dict in your database...

# Later, restore the conversation
from src.chatbot import Chatbot
bot = Chatbot.from_state_dict(state_dict)
```

## Edge Cases Handled

| Case | Result |
|------|--------|
| User says "Yes" 5 times | Resolved only via last_question_type |
| User changes budget | Update once, no backward phase |
| User asks education mid-pitch | RAG → return to current phase |
| User says "new" late | Ignored if already confirmed |
| User goes silent | Soft pause |
| Emoji spam | Treated as neutral |
| Hindi replies | Normalized to English |
| User says "no money" | Free content → exit |
| User says "loan?" | Reject loan → pause |

## Performance Optimizations

1. **Compiled regex patterns** - Regex patterns compiled once at module load
2. **Frozenset keyword matching** - O(1) lookup for keyword detection
3. **Dictionary-based dispatch** - Phase handlers use dict lookup instead of if-else chains
4. **Enum-based state** - Fast comparison and memory-efficient
5. **Immutable state updates** - No unnecessary deep copies

## Project Structure

```
AI-interviewer/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── enums.py              # Enum definitions (Phase, QuestionType, etc.)
│   ├── state.py              # State management (ConversationState, IntentSignals)
│   ├── preprocessor.py       # Text normalization and intent extraction
│   ├── fsm.py                # FSM transition logic
│   ├── response_generator.py # Response template generation
│   └── chatbot.py            # Main Chatbot class
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py       # Unit and integration tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## License

MIT
