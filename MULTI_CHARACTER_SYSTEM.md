# With The Ancients: Multi-Character System Documentation

## Overview

**With The Ancients** is a flexible multi-character conversation system inspired by Helen de Cruz's work on philosophical dialogue with historical figures. The application that allows users to:

1. **Switch between different historical figures** easily
2. **Get multiple perspectives** on the same question from several figures simultaneously
3. **Watch historical figures engage in dialogue** about topics of interest

## Architecture

### Core Components

#### 1. Character Configuration System (`character_config.py`)

**`CharacterConfig` Class**
- Loads character-specific configuration from YAML files
- Provides accessor methods for all character properties
- Handles system prompts, UI text, cognitive modes, and creative markers

**`CognitiveModeSystem` Class**
- Character-agnostic thinking mode detection
- Analyzes queries to select appropriate response framing
- Configurable per character (e.g., Geddes' Survey→Synthesis→Proposition)

**`CreativeMarkerDetector` Class**
- Tracks character-specific vocabulary and thinking patterns
- Configurable markers for analytics (e.g., ecological references for Geddes)

**`CharacterManager` Class**
- Discovers and manages all available characters
- Provides character selection interfaces

#### 2. Multi-Character Conversation System (`multi_character.py`)

**`MultiCharacterConversation` Class**
- Manages conversations with multiple participants
- Generates prompts for parallel responses and dialogues
- Formats multi-voice outputs

**`MultiCharacterResponseGenerator` Class**
- Interfaces with LLM API to generate responses
- Handles both parallel (independent) and dialogue (interactive) modes

#### 3. Main Application (`geddesghost.py`)

Refactored to support:
- **Single Character Mode**: Traditional one-on-one conversation
- **Multiple Perspectives Mode**: Ask one question, get answers from multiple figures
- **Character Dialogue Mode**: Watch figures discuss a topic together

---

## Character Directory Structure

```
characters/
├── patrick_geddes/
│   ├── config.yaml                 # Character configuration
│   ├── system_prompt.txt           # Core identity/voice prompt
│   ├── documents/                  # Character-specific knowledge base
│   │   ├── Patrick Geddes says....pdf
│   │   ├── Site Analysis Topics - Folk, Work, Place.md
│   │   └── ethnographic-observation.md
│   └── images/
│       └── patrick_geddes.jpg      # Portrait
│
├── jane_jacobs/
│   ├── config.yaml
│   ├── system_prompt.txt
│   ├── documents/
│   │   └── key_concepts.md
│   └── images/
│       └── README.md               # Placeholder for portrait
│
└── [new_character]/                # Add more characters here
    ├── config.yaml
    ├── system_prompt.txt
    ├── documents/
    └── images/
```

---

## Character Configuration Format

Each character is defined by a `config.yaml` file with the following structure:

### Example: Patrick Geddes

```yaml
character:
  name: "Patrick Geddes"
  birth_year: 1854
  death_year: 1932
  nationality: "Scottish"
  professions:
    - "biologist"
    - "sociologist"
    - "geographer"
    - "town planner"
  description: "a man of many hats - biologist, sociologist..."
  famous_quote: "By leaves we live"
  personality_traits:
    - "eccentric"
    - "provocative"
    - "interdisciplinary"

ui:
  page_title: "The Ghost of Geddes..."
  loading_message: "Re-animating Geddes Ghost..."
  greeting: |
    Greetings, dear inquirer! I am Patrick Geddes...
  input_label: "Discuss your project with Patrick:"
  portrait: "images/patrick_geddes.jpg"
  theme_color: "#2E5C3E"

cognitive_modes:
  - name: "survey"
    keywords: ["what", "describe", "analyze", ...]
    prompt_prefix: "Let us first survey and observe..."
    temperature: 0.7
    guidance: "The question calls for careful observation..."

creative_markers:
  - name: "ecological_reference"
    display_name: "Ecological References"
    keywords: ["ecology", "nature", "environment", ...]

temperature_guidance:
  high: "In this moment, allow yourself to venture..."
  low: "Focus on diagnostic precision..."
  medium: "Respond with your natural voice..."

files:
  system_prompt: "system_prompt.txt"
  documents_dir: "documents"
  portrait: "images/patrick_geddes.jpg"

creator:
  name: "Rob Annable"
  affiliation: "Birmingham School of Architecture"

dialogue:
  interaction_style: "collegial_challenging"
  compatible_characters:
    - "jane_jacobs"
    - "lewis_mumford"
  debate_topics:
    - "urban planning philosophy"
    - "role of nature in cities"
```

---

## Adding a New Character

### Step-by-Step Guide

1. **Create Character Directory**
   ```bash
   mkdir -p characters/new_character/documents characters/new_character/images
   ```

2. **Create `config.yaml`**
   - Copy an existing config (e.g., `patrick_geddes/config.yaml`)
   - Customize all sections for your character
   - Define their cognitive modes (how they think)
   - Set creative markers (vocabulary tracking)

3. **Write System Prompt** (`system_prompt.txt`)
   - First-person identity statement
   - Core beliefs and theories
   - Speaking style and personality
   - Pedagogical approach
   - How they handle unknown topics

4. **Add Knowledge Base Documents**
   - PDFs, Markdown, or text files
   - Character's writings, theories, key concepts
   - Place in `documents/` directory

5. **Add Portrait Image**
   - JPEG format: `images/[character_name].jpg`
   - Recommended: 400x500px portrait orientation

6. **Test the Character**
   - Restart the application
   - Character should appear in the selector dropdown
   - Test single-character mode first
   - Then test in multi-character modes

---

## Using the Multi-Character System

### Single Character Mode

1. Select **"Single Character"** in the sidebar
2. Choose a historical figure from dropdown
3. Ask questions as before
4. Get responses in that character's voice

### Multiple Perspectives Mode

1. Select **"Multiple Perspectives"** in the sidebar
2. Check boxes for **2+ characters** to include
3. Ask a question
4. Receive separate responses from each character
5. Compare their different viewpoints

**Example Questions:**
- "What makes a neighborhood successful?"
- "How should we approach urban planning?"
- "What role does nature play in cities?"

### Character Dialogue Mode

1. Select **"Character Dialogue"** in the sidebar
2. Check boxes for **2+ characters**
3. Enter a **discussion topic** (not a question)
4. Set number of turns per character
5. Watch them engage in a back-and-forth conversation

**Example Topics:**
- "The role of density in creating vibrant urban neighborhoods"
- "Top-down vs bottom-up planning approaches"
- "The importance of mixed-use development"

---

## How It Works: Technical Details

### Single Character Flow

1. User submits query
2. System performs RAG search on character's documents
3. Cognitive mode system analyzes query keywords
4. Temperature set based on mode (or manual override)
5. System prompt assembled (character + temp guidance + mode guidance)
6. LLM generates response in character's voice
7. Response logged and displayed

### Multiple Perspectives Flow

1. User submits query
2. RAG search performed (general context)
3. For each selected character:
   - Load character's system prompt
   - Add "parallel mode" instruction
   - Generate independent response
4. Responses formatted and displayed side-by-side

### Character Dialogue Flow

1. User submits topic
2. RAG search for relevant context
3. Determine speaking order (based on relevance)
4. For each turn:
   - Current speaker gets topic + conversation history
   - Generates response referencing previous statements
   - Response added to history
5. Full dialogue formatted and displayed

---

## Current Characters

### Patrick Geddes (1854-1932)
- **Background**: Scottish biologist, sociologist, town planner
- **Key Ideas**: Folk-Work-Place, Conservative Surgery, Synoptic Vision
- **Cognitive Modes**: Survey → Synthesis → Proposition
- **Speaking Style**: Eccentric, provocative, interdisciplinary
- **Best Topics**: Holistic planning, ecology in cities, people planning

### Jane Jacobs (1916-2006)
- **Background**: American-Canadian urban activist, writer
- **Key Ideas**: Eyes on the Street, Mixed Use, Diversity, Density
- **Cognitive Modes**: Observation → Critique → Alternative
- **Speaking Style**: Direct, empirical, grassroots-focused, critical
- **Best Topics**: Community organizing, critique of urban renewal, street life

---

## Dialogue Dynamics

The system is designed to create engaging multi-character conversations:

### Interaction Styles

Characters have defined interaction styles (from `config.yaml`):
- **collegial_challenging** (Geddes): Respectful but intellectually pushing
- **respectfully_challenging** (Jacobs): Willing to disagree but constructive

### Compatible Pairs

Some character combinations are particularly interesting:

**Geddes + Jacobs**: Both valued community observation, but different eras/approaches
- Agreement: Importance of local knowledge, critique of top-down planning
- Tension: Geddes more theoretical/biological, Jacobs more activist/pragmatic

**Future Additions:**
- **Lewis Mumford**: Regional planning, cultural urbanism
- **Robert Moses** vs **Jane Jacobs**: Historic antagonists
- **Ebenezer Howard**: Garden Cities movement
- **Le Corbusier**: Modernist planning (interesting contrast with both)

---

## Extending the System

### Voice Consistency Mechanisms

1. **System Prompt**: Core identity and speaking rules
2. **Cognitive Modes**: Frame responses appropriately for query type
3. **Temperature Guidance**: Mode-specific instructions at different temps
4. **Creative Markers**: Track character-specific vocabulary usage
5. **RAG Context**: Ground responses in character's actual writings

### Analytics Integration

The `admin_dashboard.py` can be extended to track:
- Character-specific vocabulary usage (creative markers)
- Mode selection patterns per character
- Multi-character conversation dynamics
- User preferences for different characters

### Future Enhancements

**Potential Features:**
1. **Character Relationships**: Explicit modeling of who agrees/disagrees with whom
2. **Historical Context**: Characters aware of when others lived/worked
3. **Debate Moderator**: Structured debates with turn-taking rules
4. **Cross-Reference Awareness**: Characters cite each other's work
5. **Voice Fine-Tuning**: Per-character LLM temperature preferences
6. **Multimedia**: Video/audio of historical figures when available

**Technical Improvements:**
1. **Streaming Responses**: Show dialogue developing in real-time
2. **Character-Specific RAG**: Separate TF-IDF indices per character
3. **Caching**: Cache character responses to common questions
4. **Async Generation**: Parallel API calls for multiple perspectives
5. **Conversation Memory**: Multi-turn dialogues with persistence

---

## Configuration Best Practices

### Cognitive Modes

- **3-5 modes** per character (not too many)
- **Keywords** should reflect character's actual vocabulary
- **Temperature range**: 0.6-0.9 typically
- **Prompt prefixes** in character's voice
- **Guidance** helps maintain consistency

### Creative Markers

- **4-8 markers** per character
- Focus on **distinctive vocabulary**
- Include domain-specific terms
- Track interdisciplinary connections
- Use for analytics, not enforcement

### System Prompts

- **First person mandatory** ("I am...")
- **Historical dates** for self-awareness
- **Core theories** briefly stated
- **Speaking style** explicitly described
- **Response guidelines** (length, tone, pedagogy)
- **Limitations** acknowledged (outside experience/lifespan)
- **Attribution** to creator

### Temperature Guidance

- **High (≥0.85)**: Encourage speculation, creativity, bold connections
- **Low (≤0.5)**: Emphasize precision, observation, careful analysis
- **Medium**: Balance of character's natural style

---

## Troubleshooting

### Character not appearing in selector
- Check `config.yaml` syntax (valid YAML)
- Ensure `characters/[id]/config.yaml` path is correct
- Restart application to reload character manager

### Portrait not displaying
- Verify image path in `config.yaml` matches actual file
- Check file format (JPEG recommended)
- Ensure file permissions allow reading

### Inconsistent character voice
- Review system prompt - is it detailed enough?
- Check cognitive mode keywords - do they match character's vocabulary?
- Add more character-specific documents to RAG
- Adjust temperature guidance for more explicit instructions

### Multi-character mode errors
- Ensure at least 1 character selected (Multiple Perspectives)
- Ensure at least 2 characters selected (Dialogue)
- Check API rate limits if multiple simultaneous calls

### RAG not finding character documents
- Verify documents are in `characters/[id]/documents/`
- Check file formats (.pdf, .txt, .md supported)
- Click "Reload documents (RAG)" in sidebar
- Check debug logs for loading errors

---

## Credits

**Original Application**: GeddesGhost - Patrick Geddes AI conversation system
**Author**: Rob Annable (Birmingham School of Architecture)
**Multi-Character Refactor**: Enhanced to support multiple historical figures
**System Architecture**: Character-driven configuration with modular conversation modes

---

## License & Usage

This system is designed as an educational tool for exploring urban planning theories and historical perspectives. When adding new historical figures:

- Respect intellectual property of quoted materials
- Attribute sources properly in documents
- Use publicly available biographical information
- Consider privacy and ethical implications
- Acknowledge this is AI simulation, not the actual historical figure

---

## Next Steps

**Recommended Characters to Add:**
1. **Lewis Mumford** - Regional planning, technology critique
2. **Ebenezer Howard** - Garden Cities movement
3. **Le Corbusier** - Modernist architecture and planning
4. **Robert Moses** - Urban renewal (interesting contrast with Jacobs)
5. **Kevin Lynch** - Image of the city, wayfinding
6. **Jan Gehl** - Human-scale urbanism, public spaces

**For Each New Character:**
- Research their key theories and vocabulary
- Find representative writings (public domain if possible)
- Define cognitive modes that match their thinking style
- Create engaging system prompt capturing their personality
- Test in both single and multi-character modes

**Community Contribution:**
- Create character config templates
- Share successful character configurations
- Document interesting dialogue combinations
- Build library of debate topics
- Develop pedagogical exercises using the system
