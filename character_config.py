"""
Character Configuration System
Manages loading and accessing character-specific configurations for multiple historical figures.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CharacterConfig:
    """
    Loads and manages configuration for a historical character.
    Provides access to all character-specific elements: identity, voice, cognitive modes, etc.
    """

    def __init__(self, character_id: str, characters_dir: str = "characters"):
        """
        Initialize character configuration.

        Args:
            character_id: Directory name of the character (e.g., 'patrick_geddes')
            characters_dir: Base directory containing all character folders
        """
        self.character_id = character_id
        self.base_dir = Path(characters_dir) / character_id
        self.config_path = self.base_dir / "config.yaml"

        # Load configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Character config not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded character config for: {self.get_name()}")

    # === Character Identity ===

    def get_name(self) -> str:
        """Get character's full name."""
        return self.config['character']['name']

    def get_birth_year(self) -> int:
        """Get character's birth year."""
        return self.config['character']['birth_year']

    def get_death_year(self) -> int:
        """Get character's death year."""
        return self.config['character']['death_year']

    def get_lifespan(self) -> str:
        """Get formatted lifespan (e.g., '1854-1932')."""
        return f"{self.get_birth_year()}-{self.get_death_year()}"

    def get_nationality(self) -> str:
        """Get character's nationality."""
        return self.config['character']['nationality']

    def get_professions(self) -> List[str]:
        """Get list of character's professions."""
        return self.config['character']['professions']

    def get_description(self) -> str:
        """Get character description for UI."""
        return self.config['character']['description']

    def get_famous_quote(self) -> str:
        """Get character's famous quote or catchphrase."""
        return self.config['character'].get('famous_quote', '')

    def get_personality_traits(self) -> List[str]:
        """Get list of personality traits."""
        return self.config['character'].get('personality_traits', [])

    # === UI Configuration ===

    def get_page_title(self) -> str:
        """Get page title for UI."""
        return self.config['ui']['page_title']

    def get_loading_message(self) -> str:
        """Get loading/processing message."""
        return self.config['ui']['loading_message']

    def get_greeting(self) -> str:
        """Get greeting text for new users."""
        return self.config['ui']['greeting']

    def get_input_label(self) -> str:
        """Get input field label."""
        return self.config['ui']['input_label']

    def get_portrait_path(self) -> str:
        """Get absolute path to portrait image."""
        portrait_rel = self.config['ui']['portrait']
        return str(self.base_dir / portrait_rel)

    def get_theme_color(self) -> str:
        """Get theme color for UI."""
        return self.config['ui'].get('theme_color', '#2E5C3E')

    # === Cognitive Modes ===

    def get_cognitive_modes(self) -> List[Dict[str, Any]]:
        """Get all cognitive mode configurations."""
        return self.config['cognitive_modes']

    def get_mode_by_name(self, mode_name: str) -> Optional[Dict[str, Any]]:
        """Get specific cognitive mode configuration."""
        for mode in self.config['cognitive_modes']:
            if mode['name'] == mode_name:
                return mode
        return None

    # === Creative Markers ===

    def get_creative_markers(self) -> List[Dict[str, Any]]:
        """Get creative marker configurations for analytics."""
        return self.config['creative_markers']

    # === Temperature Guidance ===

    def get_temperature_guidance(self, temperature: float) -> str:
        """
        Get temperature-specific guidance for maintaining voice.

        Args:
            temperature: Current temperature setting

        Returns:
            Guidance instruction string
        """
        guidance = self.config.get('temperature_guidance', {})

        if temperature >= 0.85:
            return guidance.get('high', '')
        elif temperature <= 0.5:
            return guidance.get('low', '')
        else:
            return guidance.get('medium', '')

    # === File Paths ===

    def get_system_prompt_path(self) -> str:
        """Get absolute path to system prompt file."""
        prompt_rel = self.config['files']['system_prompt']
        return str(self.base_dir / prompt_rel)

    def get_documents_dir(self) -> str:
        """Get absolute path to documents directory."""
        docs_rel = self.config['files']['documents_dir']
        return str(self.base_dir / docs_rel)

    def get_about_path(self) -> Optional[str]:
        """Get absolute path to about file if it exists."""
        about_rel = self.config['files'].get('about')
        if about_rel:
            about_path = self.base_dir / about_rel
            return str(about_path) if about_path.exists() else None
        return None

    # === System Prompt ===

    def get_system_prompt(self) -> str:
        """
        Load and return the character's system prompt.

        Returns:
            Full system prompt text
        """
        prompt_path = self.get_system_prompt_path()
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"System prompt not found: {prompt_path}")
            # Return a basic fallback
            return self._generate_fallback_prompt()

    def _generate_fallback_prompt(self) -> str:
        """Generate a basic fallback prompt if file is missing."""
        name = self.get_name()
        professions = ", ".join(self.get_professions())
        nationality = self.get_nationality()

        return f"""You are {name}, a {nationality} {professions}.
When responding to users, maintain the character's voice and perspective.
Respond in the first person, drawing from the historical context of {self.get_lifespan()}."""

    # === Creator Attribution ===

    def get_creator_name(self) -> str:
        """Get creator's name."""
        return self.config.get('creator', {}).get('name', '')

    def get_creator_affiliation(self) -> str:
        """Get creator's institutional affiliation."""
        return self.config.get('creator', {}).get('affiliation', '')

    def get_attribution_text(self) -> str:
        """Get full attribution text."""
        return self.config.get('creator', {}).get('attribution_text', '')

    # === Core Concepts ===

    def get_core_concepts(self) -> List[Dict[str, Any]]:
        """Get list of character's core theoretical concepts."""
        return self.config.get('core_concepts', [])

    # === Voice Guidelines ===

    def get_voice_perspective(self) -> str:
        """Get voice perspective (e.g., 'first_person')."""
        return self.config.get('voice', {}).get('perspective', 'first_person')

    def get_response_style(self) -> List[str]:
        """Get response style guidelines."""
        return self.config.get('voice', {}).get('response_style', [])

    def get_pedagogical_approach(self) -> List[str]:
        """Get pedagogical approach guidelines."""
        return self.config.get('voice', {}).get('pedagogical_approach', [])

    # === Multi-Character Dialogue ===

    def get_interaction_style(self) -> str:
        """Get how this character interacts with others."""
        return self.config.get('dialogue', {}).get('interaction_style', 'neutral')

    def get_compatible_characters(self) -> List[str]:
        """Get list of compatible dialogue partners."""
        return self.config.get('dialogue', {}).get('compatible_characters', [])

    def get_debate_topics(self) -> List[str]:
        """Get topics this character would debate passionately."""
        return self.config.get('dialogue', {}).get('debate_topics', [])

    # === Utility Methods ===

    def __str__(self) -> str:
        """String representation of character."""
        return f"CharacterConfig({self.get_name()}, {self.get_lifespan()})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"CharacterConfig(id='{self.character_id}', name='{self.get_name()}')"


class CognitiveModeSystem:
    """
    Character-agnostic cognitive mode system.
    Analyzes queries and selects appropriate thinking modes based on character configuration.
    """

    def __init__(self, character_config: CharacterConfig):
        """
        Initialize cognitive mode system for a character.

        Args:
            character_config: Character configuration object
        """
        self.character_config = character_config
        self.modes = {}

        # Build modes dictionary from config
        for mode in character_config.get_cognitive_modes():
            self.modes[mode['name']] = mode

        logger.info(f"Initialized CognitiveModeSystem with {len(self.modes)} modes for {character_config.get_name()}")

    def get_mode_parameters(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt and return optimal mode parameters.

        Args:
            prompt: User's query text

        Returns:
            Dictionary with mode, prompt_prefix, temperature, and guidance
        """
        # Convert prompt to lowercase for matching
        prompt_lower = prompt.lower()

        # Count keyword matches for each mode with weighted scoring
        mode_scores = {}
        for mode_name, params in self.modes.items():
            # Count exact keyword matches
            exact_matches = sum(
                1 for keyword in params['keywords']
                if f" {keyword} " in f" {prompt_lower} "  # Ensure whole word matching
            )

            # Count partial matches (for compound words or variations)
            partial_matches = sum(
                0.5 for keyword in params['keywords']
                if keyword in prompt_lower and f" {keyword} " not in f" {prompt_lower} "
            )

            # Combine scores
            mode_scores[mode_name] = exact_matches + partial_matches

        # Select mode with highest score (default to first mode if tied or no matches)
        default_mode = list(self.modes.keys())[0]
        selected_mode = max(
            mode_scores.items(),
            key=lambda x: (x[1], x[0] == default_mode)
        )[0]

        # Get mode parameters
        mode_params = self.modes[selected_mode]

        # Log the selected mode and score
        logger.info(f"Selected mode: {selected_mode} (score: {mode_scores[selected_mode]})")

        return {
            'mode': selected_mode,
            'prompt_prefix': mode_params['prompt_prefix'],
            'temperature': mode_params['temperature'],
            'guidance': mode_params.get('guidance', ''),
            'description': mode_params.get('description', '')
        }


class CreativeMarkerDetector:
    """
    Detects character-specific creative markers in responses for analytics.
    """

    def __init__(self, character_config: CharacterConfig):
        """
        Initialize marker detector for a character.

        Args:
            character_config: Character configuration object
        """
        self.character_config = character_config
        self.markers = {}

        # Build markers dictionary from config
        for marker in character_config.get_creative_markers():
            self.markers[marker['name']] = {
                'display_name': marker.get('display_name', marker['name']),
                'keywords': marker['keywords'],
                'description': marker.get('description', ''),
                'count': 0
            }

        logger.info(f"Initialized CreativeMarkerDetector with {len(self.markers)} markers for {character_config.get_name()}")

    def detect_markers(self, response: str) -> Dict[str, int]:
        """
        Analyze response and detect creative markers.

        Args:
            response: AI response text

        Returns:
            Dictionary mapping marker names to counts
        """
        response_lower = response.lower()
        detected = {}

        for marker_name, marker_config in self.markers.items():
            count = sum(
                1 for keyword in marker_config['keywords']
                if keyword in response_lower
            )
            detected[marker_name] = count

        return detected

    def update_counts(self, response: str):
        """
        Update cumulative marker counts based on response.

        Args:
            response: AI response text
        """
        detected = self.detect_markers(response)
        for marker_name, count in detected.items():
            self.markers[marker_name]['count'] += count

    def get_counts(self) -> Dict[str, int]:
        """Get current cumulative counts for all markers."""
        return {name: config['count'] for name, config in self.markers.items()}

    def reset_counts(self):
        """Reset all marker counts to zero."""
        for marker in self.markers.values():
            marker['count'] = 0


class CharacterManager:
    """
    Manages multiple character configurations and provides access to available characters.
    """

    def __init__(self, characters_dir: str = "characters"):
        """
        Initialize character manager.

        Args:
            characters_dir: Base directory containing character folders
        """
        self.characters_dir = Path(characters_dir)
        self.characters = {}
        self._discover_characters()

    def _discover_characters(self):
        """Discover all available character configurations."""
        if not self.characters_dir.exists():
            logger.warning(f"Characters directory not found: {self.characters_dir}")
            return

        for char_dir in self.characters_dir.iterdir():
            if char_dir.is_dir():
                config_path = char_dir / "config.yaml"
                if config_path.exists():
                    try:
                        char_config = CharacterConfig(char_dir.name, str(self.characters_dir))
                        self.characters[char_dir.name] = char_config
                        logger.info(f"Discovered character: {char_config.get_name()}")
                    except Exception as e:
                        logger.error(f"Failed to load character {char_dir.name}: {e}")

    def get_character(self, character_id: str) -> Optional[CharacterConfig]:
        """Get character configuration by ID."""
        return self.characters.get(character_id)

    def get_all_characters(self) -> Dict[str, CharacterConfig]:
        """Get all available character configurations."""
        return self.characters

    def get_character_names(self) -> List[str]:
        """Get list of all character names."""
        return [char.get_name() for char in self.characters.values()]

    def get_character_ids(self) -> List[str]:
        """Get list of all character IDs."""
        return list(self.characters.keys())

    def get_character_display_list(self) -> List[tuple]:
        """
        Get list of (display_name, character_id) tuples for UI selection.

        Returns:
            List of tuples: (name with lifespan, character_id)
        """
        return [
            (f"{char.get_name()} ({char.get_lifespan()})", char_id)
            for char_id, char in self.characters.items()
        ]


# === Utility Functions ===

def load_character(character_id: str, characters_dir: str = "characters") -> CharacterConfig:
    """
    Convenience function to load a character configuration.

    Args:
        character_id: Character directory name
        characters_dir: Base characters directory

    Returns:
        CharacterConfig object
    """
    return CharacterConfig(character_id, characters_dir)


def get_available_characters(characters_dir: str = "characters") -> List[str]:
    """
    Get list of available character IDs.

    Args:
        characters_dir: Base characters directory

    Returns:
        List of character IDs
    """
    manager = CharacterManager(characters_dir)
    return manager.get_character_ids()
