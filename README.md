# With The Ancients

An interactive AI system for conversations with historical figures. Inspired by Helen de Cruz's "Thinking with the Ancients," this application enables meaningful dialogue with great thinkers of the past through advanced language models and Retrieval Augmented Generation (RAG).

## Overview

**With The Ancients** allows you to:
- **Converse one-on-one** with historical figures in their authentic voice
- **Compare perspectives** by asking multiple figures the same question
- **Watch dialogues unfold** between historical thinkers discussing important topics

Each character is carefully configured with their theories, vocabulary, and thinking patterns to maintain authenticity and educational value.

## Features

### Multi-Character Conversation System
- **Single Character Mode**: Traditional one-on-one dialogue
- **Multiple Perspectives**: Get responses from several figures simultaneously
- **Character Dialogue**: Watch historical figures discuss topics together

### Character Configuration
- Easy-to-configure YAML-based character definitions
- Character-specific knowledge bases from their actual writings
- Cognitive mode systems matching each figure's thinking style
- Creative marker tracking for voice consistency

### Analytics & Monitoring
- Admin dashboard for conversation analytics
- Response quality metrics
- User interaction analysis
- Temperature and mode tracking
- Character-specific vocabulary analysis

### Technical Features
- Support for multiple AI providers (Anthropic Claude, Ollama)
- TF-IDF-based document retrieval (RAG)
- Conversation history tracking
- Student-specific context management
- Dynamic temperature control

## Current Historical Figures

### Patrick Geddes (1854-1932)
Scottish biologist, sociologist, geographer, and town planner
- **Key Ideas**: Folk-Work-Place, Conservative Surgery, Synoptic Vision
- **Cognitive Modes**: Survey → Synthesis → Proposition
- **Style**: Eccentric, provocative, interdisciplinary

### Jane Jacobs (1916-2006)
American-Canadian urban activist and writer
- **Key Ideas**: Eyes on the Street, Mixed Use, Diversity, Density
- **Cognitive Modes**: Observation → Critique → Alternative
- **Style**: Direct, empirical, grassroots-focused

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/with-the-ancients.git
   cd with-the-ancients
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   Create a `.env` file in the project root:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ADMIN_PASSWORD=your_admin_password_here
   ```

### Running the Application

**Using the launcher scripts (recommended):**
```bash
# Linux/Mac
./run_ancients.sh

# Windows
run_ancients.bat
```

**Or run directly:**
```bash
# Main application
streamlit run ancients.py

# Admin dashboard (separate terminal)
streamlit run admin_dashboard.py
```

## Usage Examples

### Single Character Mode
1. Select "Single Character" in the sidebar
2. Choose a historical figure from the dropdown
3. Ask questions about their theories and ideas
4. Receive responses in their authentic voice

**Example Questions:**
- "What makes a neighborhood successful?"
- "How should we approach urban planning?"
- "What role does nature play in cities?"

### Multiple Perspectives Mode
1. Select "Multiple Perspectives" in the sidebar
2. Check boxes for 2+ characters
3. Ask a single question
4. Compare their different viewpoints side-by-side

**Example:**
- Question: "What is the role of density in urban vitality?"
- Get responses from both Geddes (ecological perspective) and Jacobs (street-level empiricism)

### Character Dialogue Mode
1. Select "Character Dialogue" in the sidebar
2. Check boxes for 2+ characters
3. Enter a discussion topic
4. Watch them engage in back-and-forth conversation

**Example Topics:**
- "Top-down vs bottom-up planning approaches"
- "The importance of mixed-use development"
- "How to preserve neighborhoods while enabling growth"

## Adding New Characters

See `MULTI_CHARACTER_SYSTEM.md` for comprehensive instructions. Quick overview:

1. **Create character directory:**
   ```bash
   mkdir -p characters/new_character/documents characters/new_character/images
   ```

2. **Create configuration** (`config.yaml`):
   - Character identity and biography
   - UI text and branding
   - Cognitive modes (how they think)
   - Creative markers (vocabulary tracking)

3. **Write system prompt** (`system_prompt.txt`):
   - First-person identity
   - Core theories and beliefs
   - Speaking style and personality

4. **Add knowledge base** (documents/):
   - PDFs, Markdown, or text files
   - Their writings and key concepts

5. **Add portrait** (images/):
   - JPEG format recommended
   - 400x500px portrait orientation

Character appears automatically in the selector!

## Project Structure

```
with-the-ancients/
├── ancients.py                    # Main application
├── admin_dashboard.py             # Analytics dashboard
├── character_config.py            # Character configuration system
├── multi_character.py             # Multi-character conversation engine
├── characters/                    # Character definitions
│   ├── patrick_geddes/
│   │   ├── config.yaml
│   │   ├── system_prompt.txt
│   │   ├── documents/
│   │   └── images/
│   └── jane_jacobs/
│       ├── config.yaml
│       ├── system_prompt.txt
│       ├── documents/
│       └── images/
├── logs/                          # Conversation logs
├── history/                       # Markdown conversation history
├── students/                      # Student-specific context
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── MULTI_CHARACTER_SYSTEM.md     # Comprehensive system documentation
```

## Configuration

### Model Provider

Edit the `MODEL_CONFIG` in `ancients.py` to switch between providers:

```python
MODEL_CONFIG = {
    "current_provider": "anthropic",  # or "ollama"
    "providers": {
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 4096,
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llama3.2",
            "temperature": 0.7,
        }
    }
}
```

### Character Configuration

Each character is defined in `characters/[character_id]/config.yaml`. See existing characters for examples or consult `MULTI_CHARACTER_SYSTEM.md` for the full specification.

## Documentation

- **README.md** (this file) - Quick start and overview
- **MULTI_CHARACTER_SYSTEM.md** - Comprehensive technical documentation
  - Architecture details
  - Character creation guide
  - Configuration reference
  - Troubleshooting
  - Future enhancements
- **PROJECT_DOCUMENTATION.md** - Original system documentation

## Philosophy & Inspiration

This project is inspired by **Helen de Cruz's work** on philosophical engagement with historical thinkers, particularly her concept of "thinking with the ancients" as a form of intellectual friendship.

The system facilitates what de Cruz describes as a "parasocial relationship" with deceased thinkers, allowing us to:
- Engage imaginatively with their ideas
- Test our thinking against different perspectives
- Participate in timeless conversations about enduring questions
- Learn through dialogue rather than passive reading

### Key Philosophical Concepts

**Friendship with the Ancients**
- Not ventriloquism or historical reenactment
- Active intellectual engagement through dialogue
- Respectful but critical examination of ideas
- Recognition of both wisdom and limitations

**Parasocial Learning**
- One-sided but meaningful relationship
- Learning through imagined conversation
- Building on traditions while thinking independently
- Connecting past insights to present challenges

## References

De Cruz, H. (2020). "Friendship with the ancients: The value of philosophical
    friendships in contemporary philosophy." *Journal of the American Philosophical
    Association*, 6(3), 305-319.

De Cruz, H. (2023). "Thinking with the Ancients: How Classical Philosophy Can
    Transform Contemporary Life." Oxford University Press.

## Educational Use

**With The Ancients** is designed as an educational tool for:
- Philosophy and urban planning courses
- Critical thinking development
- Exploring historical perspectives on contemporary issues
- Understanding theoretical frameworks through dialogue
- Comparative analysis of different thinkers

### Pedagogical Applications
- Pre-class reading preparation
- Debate simulation exercises
- Theory comparison assignments
- Critical analysis practice
- Writing prompt generation

## Contributing

Contributions are welcome! Particularly:
- New historical character configurations
- Additional knowledge base documents
- UI/UX improvements
- Documentation enhancements
- Bug fixes and optimizations

See `MULTI_CHARACTER_SYSTEM.md` for character contribution guidelines.

## Recommended Characters to Add

- **Lewis Mumford** - Regional planning, technology critique
- **Ebenezer Howard** - Garden Cities movement
- **Le Corbusier** - Modernist architecture and planning
- **Robert Moses** - Urban renewal (interesting contrast with Jacobs)
- **Kevin Lynch** - Image of the city, wayfinding
- **Jan Gehl** - Human-scale urbanism, public spaces

## Technical Requirements

- Python 3.8+
- Streamlit
- Anthropic API key (for Claude) OR Ollama (local)
- 4GB+ RAM recommended
- Internet connection (for Anthropic API)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Helen de Cruz** - Philosophical inspiration and conceptual framework
- **Rob Annable** - Original GeddesGhost project and urban planning focus
- **Birmingham School of Architecture** - Educational context and support
- The historical figures themselves - for their enduring ideas and continued relevance

## Support & Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Consult `MULTI_CHARACTER_SYSTEM.md` for technical details
- Review existing character configurations for examples

---

*"The best conversations are those that continue across centuries."*
