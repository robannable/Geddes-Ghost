"""
Multi-Character Conversation System
Enables multiple historical figures to respond to queries and converse with each other.
"""

from typing import List, Dict, Tuple, Any
from character_config import CharacterConfig, CognitiveModeSystem
import logging

logger = logging.getLogger(__name__)


class MultiCharacterConversation:
    """
    Manages conversations involving multiple historical characters.
    Supports both parallel responses and simulated dialogues.
    """

    def __init__(self, characters: List[CharacterConfig]):
        """
        Initialize multi-character conversation.

        Args:
            characters: List of CharacterConfig objects to participate
        """
        self.characters = characters
        self.cognitive_systems = {
            char.character_id: CognitiveModeSystem(char)
            for char in characters
        }
        logger.info(f"Initialized MultiCharacterConversation with {len(characters)} characters")

    def generate_parallel_responses_prompt(
        self,
        user_query: str,
        character: CharacterConfig,
        context_text: str = ""
    ) -> Tuple[str, str]:
        """
        Generate prompt for a single character in parallel mode.

        Args:
            user_query: The user's question
            character: CharacterConfig for this character
            context_text: RAG context relevant to the query

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get character's system prompt
        system_prompt = character.get_system_prompt()

        # Add parallel mode instruction
        system_prompt += """

IMPORTANT: You are responding to this question alongside other historical figures.
Be yourself - express your own perspective authentically. You may agree or disagree
with others, but stay true to your own voice and theories. Keep your response focused
and distinct - this is your unique contribution to a multi-voice conversation."""

        # Get temperature guidance
        cognitive_mode = self.cognitive_systems[character.character_id]
        mode_params = cognitive_mode.get_mode_parameters(user_query)
        temperature = mode_params['temperature']

        temp_guidance = character.get_temperature_guidance(temperature)
        if temp_guidance:
            system_prompt += "\n\n" + temp_guidance

        # Construct user prompt with context
        user_prompt = f"""Context from knowledge base:
{context_text}

Question: {user_query}

Please respond in your own voice, drawing on your theories and experiences."""

        return system_prompt, user_prompt

    def generate_dialogue_prompt(
        self,
        topic: str,
        character: CharacterConfig,
        conversation_history: List[Dict[str, str]],
        context_text: str = ""
    ) -> Tuple[str, str]:
        """
        Generate prompt for a character participating in a dialogue.

        Args:
            topic: The discussion topic
            character: CharacterConfig for this character
            conversation_history: List of {"speaker": name, "message": text} dicts
            context_text: RAG context relevant to the topic

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get character's system prompt
        system_prompt = character.get_system_prompt()

        # Add dialogue mode instruction
        other_characters = [c.get_name() for c in self.characters if c.character_id != character.character_id]

        system_prompt += f"""

IMPORTANT: You are engaged in a conversation with {', '.join(other_characters)} about {topic}.
This is an imagined dialogue between historical figures. Respond naturally as yourself:
- Reference what others have said when relevant
- Agree or disagree based on your own views
- Build on or challenge others' ideas
- Stay true to your own theories and perspective
- Keep responses conversational but substantive (2-4 paragraphs typically)"""

        # Get temperature - use higher temperature for dialogue to encourage engagement
        temp_guidance = character.get_temperature_guidance(0.85)
        if temp_guidance:
            system_prompt += "\n\n" + temp_guidance

        # Construct user prompt with conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n\nConversation so far:\n"
            for turn in conversation_history:
                history_text += f"\n{turn['speaker']}: {turn['message']}\n"

        user_prompt = f"""Topic of discussion: {topic}

Context from knowledge base:
{context_text}
{history_text}

It's now your turn to contribute to this conversation. What are your thoughts?"""

        return system_prompt, user_prompt

    def determine_dialogue_order(self, topic: str) -> List[CharacterConfig]:
        """
        Determine optimal speaking order based on topic relevance.

        Args:
            topic: The discussion topic

        Returns:
            Ordered list of characters (most relevant first)
        """
        # For now, use a simple relevance heuristic based on cognitive mode matching
        # In a more advanced version, this could use semantic similarity

        scores = []
        for char in self.characters:
            cognitive_system = self.cognitive_systems[char.character_id]
            mode_params = cognitive_system.get_mode_parameters(topic)
            # Use mode score as rough relevance indicator
            score = mode_params.get('score', 0)
            scores.append((char, score))

        # Sort by score (descending)
        sorted_chars = [char for char, score in sorted(scores, key=lambda x: x[1], reverse=True)]

        logger.info(f"Dialogue order for '{topic}': {[c.get_name() for c in sorted_chars]}")
        return sorted_chars

    def format_parallel_responses(
        self,
        responses: List[Tuple[CharacterConfig, str]]
    ) -> str:
        """
        Format multiple parallel responses for display.

        Args:
            responses: List of (CharacterConfig, response_text) tuples

        Returns:
            Formatted markdown string with all responses
        """
        formatted = "# Multiple Perspectives\n\n"
        formatted += "*Responses from multiple historical figures on your question:*\n\n"
        formatted += "---\n\n"

        for char, response in responses:
            formatted += f"## {char.get_name()} ({char.get_lifespan()})\n\n"
            formatted += f"*{', '.join(char.get_professions())}*\n\n"
            formatted += response
            formatted += "\n\n---\n\n"

        return formatted

    def format_dialogue(
        self,
        conversation_history: List[Dict[str, str]],
        topic: str
    ) -> str:
        """
        Format a multi-character dialogue for display.

        Args:
            conversation_history: List of {"speaker": name, "message": text} dicts
            topic: The discussion topic

        Returns:
            Formatted markdown string with dialogue
        """
        # Get character info for metadata
        char_info = {char.get_name(): char for char in self.characters}

        formatted = f"# Dialogue: {topic}\n\n"
        formatted += "*An imagined conversation between historical figures:*\n\n"

        # Add character introductions
        formatted += "**Participants:**\n"
        for char in self.characters:
            formatted += f"- **{char.get_name()}** ({char.get_lifespan()}) - {char.get_description()}\n"
        formatted += "\n---\n\n"

        # Format dialogue
        for turn in conversation_history:
            speaker_name = turn['speaker']
            message = turn['message']

            formatted += f"### {speaker_name}\n\n"
            formatted += message
            formatted += "\n\n"

        return formatted


class MultiCharacterResponseGenerator:
    """
    Generates responses from multiple characters using an API handler.
    """

    def __init__(self, api_handler):
        """
        Initialize with an API handler that can make LLM requests.

        Args:
            api_handler: ModelAPIHandler instance
        """
        self.api_handler = api_handler

    def generate_parallel_responses(
        self,
        multi_char_conv: MultiCharacterConversation,
        user_query: str,
        context_text: str = "",
        temperature: float = None
    ) -> List[Tuple[CharacterConfig, str]]:
        """
        Generate responses from all characters in parallel mode.

        Args:
            multi_char_conv: MultiCharacterConversation instance
            user_query: User's question
            context_text: RAG context
            temperature: Optional temperature override (uses character defaults if None)

        Returns:
            List of (CharacterConfig, response_text) tuples
        """
        responses = []

        for character in multi_char_conv.characters:
            try:
                # Generate prompts for this character
                system_prompt, user_prompt = multi_char_conv.generate_parallel_responses_prompt(
                    user_query, character, context_text
                )

                # Determine temperature
                if temperature is None:
                    cognitive_system = multi_char_conv.cognitive_systems[character.character_id]
                    mode_params = cognitive_system.get_mode_parameters(user_query)
                    char_temperature = mode_params['temperature']
                else:
                    char_temperature = temperature

                # Make API request
                response_json = self.api_handler.make_request(
                    user_prompt,
                    system_prompt=system_prompt,
                    temperature=char_temperature
                )

                # Extract response text
                response_text = self._extract_response_text(response_json)

                responses.append((character, response_text))
                logger.info(f"Generated parallel response from {character.get_name()}")

            except Exception as e:
                logger.error(f"Error generating response from {character.get_name()}: {e}")
                responses.append((character, f"[Error generating response: {str(e)}]"))

        return responses

    def generate_dialogue(
        self,
        multi_char_conv: MultiCharacterConversation,
        topic: str,
        context_text: str = "",
        num_turns: int = 3,
        temperature: float = 0.85
    ) -> List[Dict[str, str]]:
        """
        Generate a multi-turn dialogue between characters.

        Args:
            multi_char_conv: MultiCharacterConversation instance
            topic: Discussion topic
            context_text: RAG context
            num_turns: Number of turns per character
            temperature: Temperature for dialogue generation

        Returns:
            List of {"speaker": name, "message": text} dicts representing the conversation
        """
        conversation_history = []

        # Determine speaking order
        character_order = multi_char_conv.determine_dialogue_order(topic)

        # Generate dialogue turns
        for turn_num in range(num_turns):
            for character in character_order:
                try:
                    # Generate prompts for this character
                    system_prompt, user_prompt = multi_char_conv.generate_dialogue_prompt(
                        topic, character, conversation_history, context_text
                    )

                    # Make API request
                    response_json = self.api_handler.make_request(
                        user_prompt,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )

                    # Extract response text
                    response_text = self._extract_response_text(response_json)

                    # Add to conversation history
                    conversation_history.append({
                        "speaker": character.get_name(),
                        "message": response_text
                    })

                    logger.info(f"Generated dialogue turn from {character.get_name()} (turn {turn_num + 1})")

                except Exception as e:
                    logger.error(f"Error generating dialogue from {character.get_name()}: {e}")
                    conversation_history.append({
                        "speaker": character.get_name(),
                        "message": f"[Error: {str(e)}]"
                    })

        return conversation_history

    def _extract_response_text(self, response_json: Dict) -> str:
        """
        Extract text from API response (handles different formats).

        Args:
            response_json: API response

        Returns:
            Response text string
        """
        # Handle Anthropic format
        if isinstance(response_json, dict):
            if "content" in response_json:
                content = response_json["content"]
                if isinstance(content, list):
                    # Extract text from content blocks
                    return " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and "text" in block
                    )
                return str(content)
            elif "message" in response_json and "content" in response_json["message"]:
                return response_json["message"]["content"]
            elif "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]

        # Fallback
        return str(response_json)


# Utility functions

def create_multi_character_session(
    character_ids: List[str],
    character_manager
) -> MultiCharacterConversation:
    """
    Create a multi-character conversation session from character IDs.

    Args:
        character_ids: List of character IDs to include
        character_manager: CharacterManager instance

    Returns:
        MultiCharacterConversation instance
    """
    characters = []
    for char_id in character_ids:
        char = character_manager.get_character(char_id)
        if char:
            characters.append(char)
        else:
            logger.warning(f"Character not found: {char_id}")

    if not characters:
        raise ValueError("No valid characters found for multi-character session")

    return MultiCharacterConversation(characters)
