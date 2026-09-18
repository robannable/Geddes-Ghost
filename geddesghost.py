"""
GeddesGhost AI Assistant System
--------------------------------
A Retrieval-Augmented Generation (RAG) application that simulates interactions
with Patrick Geddes, the Scottish polymath. The system currently includes:

- TF-IDF-based retrieval over `documents/`, `history/`, and `students/`
- Dynamic cognitive modes mapped to a response depth band
- Capability-driven model selection: whichever generation control the chosen
  model accepts (temperature, or output_config.effort) is the one that is sent
- Context-aware response generation via Anthropic Claude (default) or Ollama
- Comprehensive logging, including token usage, and an Admin Dashboard

Admin Dashboard views:
1. Performance, Document Usage, User Analysis, Response Metrics
2. Topics Map (topic heatmap and doc-to-topic contribution)
3. Reflections (sentiment/keywords and action items)
4. Interventions (auto-generated teaching plan)

Document categories used for retrieval:
1. Authoritative documents (core knowledge)
2. Historical records (past interactions)
3. Student-specific content (personalized context)

Responses are generated using a structured context assembly process and
cognitive mode selection system inspired by Geddes' teaching approach.

Author: Rob Annable
Last Updated: 08-09-2025
Version: 1.1
"""

import re
import streamlit as st
import requests
import json
import pygame
import os
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
from pypdf import PdfReader
try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import CharacterTextSplitter
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import html
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
import dotenv

# First, define the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

import logging
import time

# Set up logging
log_dir = os.path.join(script_dir, "debug_logs")
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.now().strftime("%d-%m-%Y")
log_file = os.path.join(log_dir, f"{current_date}_rag_loading.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)  # Add this line to create the logger instance

# Initialize directories
sound_dir = os.path.join(script_dir, 'sounds')
prompts_dir = os.path.join(script_dir, 'prompts')
about_file_path = os.path.join(script_dir, 'about.txt')

# Initialize pygame for audio (gracefully handle VPS/headless environments)
audio_available = False
ding_sound = None
try:
    pygame.mixer.init()
    ding_sound = pygame.mixer.Sound(os.path.join(sound_dir, 'ding2.wav'))
    audio_available = True
    logger.info("Audio initialized successfully")
except pygame.error as e:
    logger.warning(f"Audio initialization failed (running in headless/VPS mode): {e}")
except Exception as e:
    logger.warning(f"Audio initialization failed: {e}")

# Define constants at the top of the file
CONTEXT_WEIGHTS = {
    'student_specific': 1.5,  # Highest priority
    'project': 1.3,          # Project-related content
    'historical': 1.2,       # Historical context
    'general': 1.0          # Base documents
}

# Add these classes after your existing imports but before any function definitions
@dataclass
class ContextItem:
    content: str
    timestamp: datetime  # Changed from datetime.datetime
    source: str
    relevance_score: float = 0.0

class EnhancedContextManager:
    def __init__(self, max_memory_items: int = 10):
        self.max_memory_items = max_memory_items
        self.conversation_memory: List[ContextItem] = []
        self.context_weights = CONTEXT_WEIGHTS
    
    def add_conversation(self, content: str, source: str):
        context_item = ContextItem(
            content=content,
            timestamp=datetime.now(),  # Changed from datetime.datetime.now()
            source=source
        )
        self.conversation_memory.append(context_item)
        if len(self.conversation_memory) > self.max_memory_items:
            self.conversation_memory.pop(0)
    
    def get_weighted_context(self, query: str, user_name: str) -> Dict[str, List[ContextItem]]:
        categorized_context = {
            'student_specific': [],
            'recent_conversation': [],
            'historical': [],
            'general': []
        }
        
        # Categorize conversation memory
        for item in self.conversation_memory:
            if user_name.lower() in item.source.lower():
                categorized_context['student_specific'].append(item)
            else:
                categorized_context['recent_conversation'].append(item)
        
        return categorized_context

class GeddesCognitiveModes:
    def __init__(self):
        self.modes = {
            'survey': {
                'keywords': [
                    'what', 'describe', 'analyze', 'observe', 'examine', 'study',
                    'investigate', 'explore', 'map', 'document', 'record', 'measure',
                    'identify', 'catalogue', 'survey', 'inspect', 'review', 'assess',
                    'where', 'when', 'who', 'which', 'look', 'find', 'discover'
                ],
                'prompt_prefix': 'Let us first survey and observe...',
                'temperature': 0.7,
                'effort': 'medium'
            },
            'synthesis': {
                'keywords': [
                    'how', 'connect', 'relate', 'integrate', 'combine', 'synthesize',
                    'weave', 'blend', 'merge', 'link', 'bridge', 'join', 'unite',
                    'pattern', 'relationship', 'network', 'system', 'structure',
                    'framework', 'together', 'between', 'across', 'through',
                    'interconnect', 'associate', 'correlate'
                ],
                'prompt_prefix': 'Now, let us weave together these disparate threads...',
                'temperature': 0.8,
                'effort': 'high'
            },
            'proposition': {
                'keywords': [
                    'why', 'propose', 'suggest', 'could', 'might', 'imagine',
                    'envision', 'create', 'design', 'develop', 'innovate', 'transform',
                    'improve', 'enhance', 'advance', 'future', 'potential', 'possible',
                    'alternative', 'solution', 'strategy', 'plan', 'vision',
                    'hypothesis', 'theory', 'concept'
                ],
                'prompt_prefix': 'Let us venture forth with a proposition...',
                'temperature': 0.9,
                'effort': 'xhigh'
            }
        }
        logger.info("Initializing new GeddesCognitiveModes")

    def get_mode_parameters(self, prompt: str) -> dict:
        # Convert prompt to lowercase for matching
        prompt_lower = prompt.lower()
        
        # Count keyword matches for each mode with weighted scoring
        mode_scores = {}
        for mode, params in self.modes.items():
            # Count exact keyword matches
            exact_matches = sum(
                1 for keyword in params['keywords'] 
                if f" {keyword} " in f" {prompt_lower} "  # Add spaces to ensure whole word matching
            )
            
            # Count partial matches (for compound words or variations)
            partial_matches = sum(
                0.5 for keyword in params['keywords']
                if keyword in prompt_lower and f" {keyword} " not in f" {prompt_lower} "
            )
            
            # Combine scores
            mode_scores[mode] = exact_matches + partial_matches
        
        # Select mode with highest score (default to 'survey' if tied or no matches)
        selected_mode = max(
            mode_scores.items(),
            key=lambda x: (x[1], x[0] == 'survey')  # Prioritize survey mode in ties
        )[0]
        
        # Log the selected mode and score
        logger.info(f"Selected mode: {selected_mode} (score: {mode_scores[selected_mode]})")
        
        return {
            'mode': selected_mode,
            'prompt_prefix': self.modes[selected_mode]['prompt_prefix'],
            'temperature': self.modes[selected_mode]['temperature'],
            'effort': self.modes[selected_mode]['effort']
        }

# ---------------------------------------------------------------------------
# Response depth
# ---------------------------------------------------------------------------
# Survey / synthesis / proposition is a claim about how much thinking an answer
# deserves, not about randomness. Temperature was only ever a proxy for that,
# and current Anthropic models no longer accept it. "Depth" is the provider
# neutral middle term: whichever control the selected model supports is mapped
# onto it, and the prompt guidance keys off the depth band. On a model that
# accepts no generation controls at all, the prompt is still the steer.

MODE_DEPTH = {
    'survey': 'focused',
    'synthesis': 'balanced',
    'proposition': 'expansive',
}

EFFORT_DEPTH = {
    'low': 'focused',
    'medium': 'focused',
    'high': 'balanced',
    'xhigh': 'expansive',
    'max': 'expansive',
}

DEPTH_GUIDANCE = {
    'focused': "\n\nIn this moment, focus on diagnostic precision and careful observation. Be economical with words and deliberate in your analysis.",
    'balanced': "\n\nRespond with your natural voice, balancing observation with interpretation as the question warrants.",
    'expansive': "\n\nIn this moment, allow yourself to venture into bold speculation and unexpected connections. Let the response breathe and expand where the ideas demand it. Embrace creative risk.",
}


def resolve_depth(mode, temperature=None, effort=None):
    """Map whichever generation control is in play onto a depth band."""
    if temperature is not None:
        if temperature >= 0.85:
            return 'expansive'
        if temperature <= 0.5:
            return 'focused'
        return 'balanced'
    if effort is not None:
        return EFFORT_DEPTH.get(effort, 'balanced')
    return MODE_DEPTH.get(mode, 'balanced')


def format_depth_control(generation_info):
    """Human-readable summary of the control that shaped this response."""
    depth = generation_info.get('depth', 'balanced')
    if generation_info.get('temperature') is not None:
        return f"{depth} (temperature {generation_info['temperature']})"
    if generation_info.get('effort') is not None:
        return f"{depth} (effort {generation_info['effort']})"
    return f"{depth} (prompt only)"


def format_usage(usage):
    """Human-readable token usage, or a note when the provider reports none."""
    if not usage:
        return "not reported"
    parts = []
    if usage.get('input_tokens') is not None:
        parts.append(f"{usage['input_tokens']} in")
    if usage.get('output_tokens') is not None:
        parts.append(f"{usage['output_tokens']} out")
    if usage.get('cache_read_input_tokens'):
        parts.append(f"{usage['cache_read_input_tokens']} cached")
    return ' / '.join(parts) if parts else "not reported"


# ---------------------------------------------------------------------------
# Model registry and capabilities
# ---------------------------------------------------------------------------
# Anthropic removed the sampling parameters (temperature / top_p / top_k) from
# Opus 4.7 onwards: sending any of them to a current model returns a 400.
# Reasoning depth is now steered with `output_config.effort` instead. Rather
# than hard-code one knob, each model declares what it accepts and both the
# request payload and the sidebar are built from that declaration.
#
# Unknown models default to "no sampling parameters, no effort". Omitting a
# parameter is always valid; sending one the model rejects is not.

EFFORT_LEVELS_FULL = ["low", "medium", "high", "xhigh", "max"]

ANTHROPIC_MODELS = {
    "claude-opus-5": {
        "display_name": "Claude Opus 5",
        "sampling": False,
        "effort_levels": EFFORT_LEVELS_FULL,
    },
    "claude-sonnet-5": {
        "display_name": "Claude Sonnet 5",
        "sampling": False,
        "effort_levels": EFFORT_LEVELS_FULL,
    },
    "claude-opus-4-8": {
        "display_name": "Claude Opus 4.8",
        "sampling": False,
        "effort_levels": EFFORT_LEVELS_FULL,
    },
    "claude-sonnet-4-6": {
        "display_name": "Claude Sonnet 4.6",
        "sampling": True,
        "effort_levels": ["low", "medium", "high", "max"],
    },
    "claude-haiku-4-5": {
        "display_name": "Claude Haiku 4.5",
        "sampling": True,
        "effort_levels": [],
    },
    "claude-sonnet-4-20250514": {
        "display_name": "Claude Sonnet 4 (deprecated)",
        "sampling": True,
        "effort_levels": [],
        "deprecated": True,
    },
}

UNKNOWN_MODEL_CAPABILITIES = {
    "display_name": None,
    "sampling": False,
    "effort_levels": [],
}


def get_model_capabilities(provider, model):
    """Return the capability declaration for a provider/model pair.

    Falls back to the conservative default for models we do not recognise, so
    a newly released model never causes a 400 from a parameter it rejects.
    """
    if provider == "ollama":
        # Ollama passes sampling options straight through to the local runtime.
        return {"display_name": model, "sampling": True, "effort_levels": []}

    discovered = st.session_state.get("discovered_anthropic_models", {})
    if model in discovered:
        return discovered[model]
    if model in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model]

    caps = dict(UNKNOWN_MODEL_CAPABILITIES)
    caps["display_name"] = model
    return caps


# Load model config from file or notepad (for now, hardcode as a dict)
MODEL_CONFIG = {
    "current_provider": "anthropic",
    "providers": {
        "anthropic": {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "max_tokens": 4000,
            # Only used by models that still accept sampling parameters.
            "temperature": 0.7,
            # Used by models that accept output_config.effort.
            "effort": "high",
            "message_retention": "no_retention",
            "api_endpoint": "https://api.anthropic.com/v1/messages",
            "models_endpoint": "https://api.anthropic.com/v1/models",
            "api_key_env": "ANTHROPIC_API_KEY",
            "headers": {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
        },
        "ollama": {
            "provider": "ollama",
            "model": "cogito:latest",
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "api_endpoint": "http://localhost:11434/api/generate",
            "models_endpoint": "http://localhost:11434/api/tags",
            "headers": {
                "Content-Type": "application/json"
            }
        }
    },
    # (connect, read) seconds. Without this a stalled connection hangs Streamlit
    # indefinitely.
    "timeout": (10, 120),
}


class ModelResponse:
    """Normalised result of a generation request, whatever the provider."""

    def __init__(self, text, usage=None, stop_reason=None, model=None, raw=None):
        self.text = text
        self.usage = usage or {}
        self.stop_reason = stop_reason
        self.model = model
        self.raw = raw or {}


def build_http_session():
    """A requests session that retries transient failures with backoff.

    POST is not retried by default, so it has to be named explicitly. 429 and
    529 are Anthropic's rate-limit and overloaded responses.
    """
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504, 529],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class ModelAPIHandler:
    def __init__(self, config):
        self.config = config
        self.provider = config["current_provider"]
        self.provider_config = config["providers"][self.provider]
        self.api_key_env = self.provider_config.get("api_key_env")
        self.api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        self.api_endpoint = self.provider_config["api_endpoint"]
        self.timeout = config.get("timeout", (10, 120))
        self.headers = self.provider_config["headers"].copy()
        if self.api_key:
            # Add API key to headers if needed
            if self.provider == "anthropic":
                self.headers["x-api-key"] = self.api_key
        self.session = build_http_session()

    @property
    def capabilities(self):
        return get_model_capabilities(self.provider, self.provider_config["model"])

    def get_available_ollama_models(self):
        """Fetch available models from Ollama server"""
        try:
            response = self.session.get(
                self.provider_config.get("models_endpoint", "http://localhost:11434/api/tags"),
                timeout=self.timeout,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def get_available_anthropic_models(self):
        """Fetch the live model list from Anthropic and derive capabilities.

        GET /v1/models costs nothing and reports each model's id, display name
        and capability tree, so the sidebar no longer depends on a list baked
        into this file. Effort support is read from the capability tree;
        sampling support is not reported by the API, so it comes from the
        static registry and defaults to False for anything unrecognised.
        """
        endpoint = self.provider_config.get("models_endpoint")
        if not endpoint or not self.api_key:
            return {}
        try:
            response = self.session.get(endpoint, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            logger.error(f"Error fetching Anthropic models: {str(e)}")
            return {}

        discovered = {}
        for entry in payload.get("data", []):
            model_id = entry.get("id")
            if not model_id:
                continue
            known = ANTHROPIC_MODELS.get(model_id, {})
            effort_caps = (entry.get("capabilities") or {}).get("effort") or {}
            effort_levels = [
                level for level in EFFORT_LEVELS_FULL
                if (effort_caps.get(level) or {}).get("supported")
            ]
            if not effort_levels:
                effort_levels = known.get("effort_levels", [])
            discovered[model_id] = {
                "display_name": entry.get("display_name") or known.get("display_name") or model_id,
                "sampling": known.get("sampling", False),
                "effort_levels": effort_levels,
                "max_tokens": entry.get("max_tokens"),
                "max_input_tokens": entry.get("max_input_tokens"),
                "deprecated": known.get("deprecated", False),
            }
        return discovered

    def build_payload(self, prompt, system_prompt=None, temperature=None, effort=None):
        """Assemble a provider-specific request body from the model's capabilities."""
        caps = self.capabilities

        if self.provider == "anthropic":
            payload = {
                "model": self.provider_config["model"],
                "max_tokens": self.provider_config["max_tokens"],
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            if system_prompt:
                payload["system"] = system_prompt
            # Send temperature only where it is still accepted, and never
            # alongside top_p: passing both errors on every Claude 4+ model.
            if caps.get("sampling"):
                effective_temperature = (
                    temperature if temperature is not None
                    else self.provider_config.get("temperature")
                )
                if effective_temperature is not None:
                    payload["temperature"] = effective_temperature
            # Effort replaces temperature as the depth control on current models.
            if caps.get("effort_levels"):
                effective_effort = effort or self.provider_config.get("effort")
                if effective_effort in caps["effort_levels"]:
                    payload["output_config"] = {"effort": effective_effort}
            return payload

        if self.provider == "ollama":
            effective_temperature = (
                temperature if temperature is not None
                else self.provider_config.get("temperature", 0.7)
            )
            return {
                "model": self.provider_config["model"],
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "stream": False,
                "options": {
                    "temperature": effective_temperature,
                    "top_p": self.provider_config.get("top_p", 0.9),
                    "num_predict": self.provider_config["max_tokens"]
                }
            }

        raise ValueError(f"Unsupported provider: {self.provider}")

    def make_request(self, prompt, system_prompt=None, temperature=None, effort=None):
        """Send a generation request and return a normalised ModelResponse."""
        payload = self.build_payload(
            prompt, system_prompt=system_prompt, temperature=temperature, effort=effort
        )

        try:
            response = self.session.post(
                self.api_endpoint, headers=self.headers, json=payload, timeout=self.timeout
            )
        except RequestException as e:
            logger.error(f"Request to {self.provider} failed: {str(e)}")
            raise ValueError(f"Could not reach the {self.provider} API: {str(e)}")

        if response.status_code >= 400:
            # The API reports what it disliked in the body; a bare status code
            # is not enough to debug a rejected parameter.
            detail = response.text[:500]
            logger.error(f"{self.provider} API returned {response.status_code}: {detail}")
            raise ValueError(f"{self.provider} API error {response.status_code}: {detail}")

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {self.provider} response: {str(e)}")
            raise ValueError(f"Error decoding {self.provider} response: {str(e)}")

        if self.provider == "anthropic":
            return self._parse_anthropic_response(data)
        return self._parse_ollama_response(data)

    def _parse_anthropic_response(self, data):
        stop_reason = data.get("stop_reason")

        # A refusal arrives as HTTP 200 with no usable content, so it has to be
        # checked before reading the content blocks.
        if stop_reason == "refusal":
            details = data.get("stop_details") or {}
            category = details.get("category") or "unspecified"
            raise ValueError(
                f"The model declined to answer this request (category: {category})."
            )

        blocks = data.get("content")
        if not isinstance(blocks, list):
            logger.error(f"Unexpected Anthropic response format: {data}")
            raise ValueError(f"Unexpected Anthropic API response format: {data}")

        # Text blocks only: thinking blocks carry no `text` key and are skipped.
        text = " ".join(
            block.get("text", "") for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()

        if stop_reason == "max_tokens":
            logger.warning("Response truncated: hit max_tokens")

        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            usage={
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            },
            stop_reason=stop_reason,
            model=data.get("model"),
            raw=data,
        )

    def _parse_ollama_response(self, data):
        if not isinstance(data, dict) or "response" not in data:
            logger.error(f"Unexpected Ollama response format: {data}")
            raise ValueError(f"Unexpected Ollama response format: {data}")

        return ModelResponse(
            text=data["response"].strip(),
            usage={
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "cache_read_input_tokens": None,
            },
            stop_reason=data.get("done_reason"),
            model=data.get("model"),
            raw=data,
        )

dotenv.load_dotenv()

def check_api_connection():
    """Check we can reach the provider without spending tokens on it."""
    try:
        api_handler = ModelAPIHandler(MODEL_CONFIG)
        endpoint = api_handler.provider_config.get("models_endpoint")
        if not endpoint:
            return False
        response = api_handler.session.get(
            endpoint, headers=api_handler.headers, timeout=api_handler.timeout
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API connection error: {str(e)}")
        return False

@st.cache_data
def get_patrick_prompt():
    prompt_file_path = os.path.join(prompts_dir, 'patrick_geddes_prompt.txt')
    try:
        with open(prompt_file_path, 'r') as file:
            prompt = file.read().strip()
        prompt += "\n\nWhen responding to users, consider their name and potential gender implications. Avoid making assumptions based on stereotypes and strive for inclusive language. Adapt your language and examples to be appropriate for all users, regardless of their perceived gender."
        return prompt
    except FileNotFoundError:
        st.warning(f"'{prompt_file_path}' not found. Using default prompt.")
        return "You are Patrick Geddes, a Scottish biologist, sociologist, and town planner. When responding to users, consider their name and potential gender implications. Avoid making assumptions based on stereotypes and strive for inclusive language. Adapt your language and examples to be appropriate for all users, regardless of their perceived gender."

@st.cache_data
def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True  # Contains HTML
    except FileNotFoundError:
        st.warning(f"'{about_file_path}' not found. Using default about info.")
        return "This app uses advanced AI models to simulate a conversation with Patrick Geddes...", False

@st.cache_data
def load_documents(directories=['documents', 'history', 'students']):
    total_start_time = time.time()
    texts = []
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    # Define system directories to ignore
    ignore_dirs = {
        '__pycache__',
        '.ipynb_checkpoints',
        '.git',
        'debug_logs',
        'logs',
        'sounds',
        'prompts',
        'images'
    }
    
    for directory in directories:
        dir_path = os.path.join(script_dir, directory)
        if os.path.exists(dir_path):
            dir_start_time = time.time()
            files_processed = 0
            
            for item in os.listdir(dir_path):
                # Skip if item is in ignored directories or is hidden
                if item in ignore_dirs or item.startswith('.'):
                    continue
                    
                filepath = os.path.join(dir_path, item)
                
                # Skip if it's a directory
                if os.path.isdir(filepath):
                    continue
                    
                # Skip files with today's date in the history folder
                if directory == 'history' and current_date in item:
                    continue
                
                file_start_time = time.time()
                
                try:
                    if item.endswith('.pdf'):
                        with open(filepath, 'rb') as file:
                            pdf_reader = PdfReader(file)
                            for page in pdf_reader.pages:
                                texts.append((page.extract_text(), item))
                    elif item.endswith(('.txt', '.md')):
                        with open(filepath, 'r', encoding='utf-8') as file:
                            texts.append((file.read(), item))
                    elif item.endswith(('.png', '.jpg', '.jpeg')):
                        image = Image.open(filepath)
                        text = pytesseract.image_to_string(image)
                        texts.append((text, item))
                        
                    file_time = time.time() - file_start_time
                    logging.info(f"Loaded {item} in {file_time:.2f} seconds")
                    files_processed += 1
                    
                except Exception as e:
                    logging.error(f"Failed to load {item}: {str(e)}")
            
            dir_time = time.time() - dir_start_time
            logging.info(f"Directory {directory}: processed {files_processed} files in {dir_time:.2f} seconds")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks_with_filenames = [(chunk, filename) for text, filename in texts for chunk in text_splitter.split_text(text)]
    
    total_time = time.time() - total_start_time
    logging.info(f"Total RAG loading completed in {total_time:.2f} seconds - Created {len(chunks_with_filenames)} chunks from {len(texts)} documents")
    
    return chunks_with_filenames


@st.cache_resource
def compute_tfidf_matrix(document_chunks):
    documents = [chunk for chunk, _ in document_chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Load document chunks and compute TF-IDF matrix at startup
document_chunks_with_filenames = load_documents(['documents', 'history', 'students'])
vectorizer, tfidf_matrix = compute_tfidf_matrix(document_chunks_with_filenames)

def is_name_match(query_name: str, target_name: str) -> bool:
    """
    Check if names match, handling partial matches and common variations.
    Returns True if there's a match, False otherwise.
    """
    # Convert to lowercase and strip whitespace
    query_name = query_name.lower().strip()
    target_name = target_name.lower().strip()
    
    # Split names into parts
    query_parts = query_name.split()
    target_parts = target_name.split()
    
    # Check for exact match
    if query_name == target_name:
        return True
        
    # Check if first name matches
    if query_parts[0] == target_parts[0]:
        return True
        
    # Check if any part of the query name is in the target name
    for query_part in query_parts:
        if query_part in target_name:
            return True
            
    return False

# Add a context weighting system to prioritize different types of documents
def weight_context_chunks(prompt, chunks_with_filenames, vectorizer, tfidf_matrix):
    # Convert prompt to TF-IDF vector
    prompt_vector = vectorizer.transform([prompt])
    
    # Compute cosine similarities
    similarities = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
    
    # Apply weights based on document type using global constants
    weighted_similarities = similarities.copy()
    for i, (chunk, filename) in enumerate(chunks_with_filenames):
        # Extract student name from filename (remove .txt and path)
        filename_base = os.path.splitext(os.path.basename(filename))[0]
        
        # Check if this is a student file (either in filename or content)
        is_student_file = (
            "students/" in filename.lower() or  # Check if file is in students directory
            any(name.lower() in filename.lower() for name in ["student:", "student name:", "name:"]) or  # Check for student name markers
            any(name.lower() in chunk.lower() for name in ["student:", "student name:", "name:"])  # Check content for student name markers
        )
        
        if is_student_file:
            weighted_similarities[i] *= CONTEXT_WEIGHTS['student_specific']
        elif "project" in filename.lower() or "project" in chunk.lower():
            weighted_similarities[i] *= CONTEXT_WEIGHTS['project']
        elif "history" in filename.lower():
            weighted_similarities[i] *= CONTEXT_WEIGHTS['historical']
        else:
            weighted_similarities[i] *= CONTEXT_WEIGHTS['general']
    
    # Log the weighting process
    logger.info(f"Applied context weights: {CONTEXT_WEIGHTS}")
    
    return weighted_similarities

def initialize_log_files():
    """Initialize or get existing log files"""
    current_date = datetime.now().strftime("%d-%m-%Y")
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    csv_file = os.path.join(logs_dir, f"{current_date}_response_log.csv")
    json_file = os.path.join(logs_dir, f"{current_date}_response_log.json")
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'date', 'time', 'name', 'question', 'response',
                'unique_files', 'chunk1_score', 'chunk2_score', 'chunk3_score',
                'cognitive_mode', 'response_length', 'creative_markers', 'temperature',
                'actual_temperature', 'temperature_source', 'detected_mode',
                'model_provider', 'model_name',
                'effort', 'depth', 'input_tokens', 'output_tokens',
                'cache_read_input_tokens', 'stop_reason'
            ], quoting=csv.QUOTE_ALL)
            writer.writeheader()
    
    return csv_file, json_file

def write_markdown_history(user_name, question, response, csv_file):
    history_dir = os.path.join(script_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    md_file = os.path.join(history_dir, f"{current_date}_conversation_history.md")
    
    # Get current model information
    current_provider = MODEL_CONFIG["current_provider"]
    current_model = MODEL_CONFIG["providers"][current_provider]["model"]
    
    with open(md_file, 'a', encoding='utf-8') as f:
        f.write(f"## Date: {current_date} | Time: {current_time}\n\n")
        f.write(f"### User: {user_name}\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(f"**Patrick Geddes:** {response}\n\n")
        f.write(f"**Model Used:** {current_provider} - {current_model}\n\n")
        f.write("---\n\n")

def update_chat_logs(user_name, question, response, unique_files, chunk_info, csv_file, json_file, generation_info=None):
    """Update both CSV and JSON logs with chat data"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Use generation_info if provided, otherwise fall back to cognitive mode detection
    if generation_info:
        current_mode = generation_info['mode']
        temperature = generation_info.get('temperature')
        effort = generation_info.get('effort')
        depth = generation_info.get('depth', 'balanced')
        temperature_source = generation_info['source']
        usage = generation_info.get('usage') or {}
        stop_reason = generation_info.get('stop_reason')
    else:
        mode_params = st.session_state.cognitive_modes.get_mode_parameters(question)
        current_mode = mode_params['mode']
        temperature = mode_params['temperature']
        effort = mode_params['effort']
        depth = resolve_depth(current_mode, temperature=temperature)
        temperature_source = "auto (legacy)"
        usage = {}
        stop_reason = None

    current_provider = MODEL_CONFIG["current_provider"]
    current_model = MODEL_CONFIG["providers"][current_provider]["model"]

    # Get evaluation with accumulated metrics
    evaluation_results = st.session_state.response_evaluator.evaluate_response(
        response=response,
        mode=current_mode,
        temperature=temperature,
        temperature_source=temperature_source,
        effort=effort
    )
    
    # Prepare CSV row with full metrics
    csv_row = {
        'date': current_date,
        'time': current_time,
        'name': user_name,
        'question': question,
        'response': response,
        'unique_files': ' - '.join(unique_files),
        'chunk1_score': chunk_info[0] if len(chunk_info) > 0 else '',
        'chunk2_score': chunk_info[1] if len(chunk_info) > 1 else '',
        'chunk3_score': chunk_info[2] if len(chunk_info) > 2 else '',
        'cognitive_mode': str(evaluation_results['mode_distribution']),
        'response_length': evaluation_results['avg_response_length'],
        'creative_markers': str(evaluation_results['creative_markers_frequency']),
        'temperature': str(evaluation_results['temperature_effectiveness']),
        'actual_temperature': temperature,  # The actual temperature used
        'temperature_source': temperature_source,  # auto/manual
        'detected_mode': current_mode,  # The cognitive mode detected/used
        'model_provider': current_provider,
        'model_name': current_model,
        'effort': effort,
        'depth': depth,
        'input_tokens': usage.get('input_tokens'),
        'output_tokens': usage.get('output_tokens'),
        'cache_read_input_tokens': usage.get('cache_read_input_tokens'),
        'stop_reason': stop_reason
    }
    
    # Write to CSV with proper quoting to handle multi-line responses
    fieldnames = list(csv_row.keys())
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)
    
    # Prepare JSON entry
    json_entry = {
        'date': current_date,
        'time': current_time,
        'name': user_name,
        'question': question,
        'response': response,
        'unique_files': unique_files,
        'chunk_info': chunk_info,
        'cognitive_mode': current_mode,
        'evaluation': evaluation_results,
        'actual_temperature': temperature,
        'temperature_source': temperature_source,
        'effort': effort,
        'depth': depth,
        'usage': usage,
        'stop_reason': stop_reason,
        'model_provider': current_provider,
        'model_name': current_model
    }
    
    # Update JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = []
    
    chat_history.append(json_entry)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, indent=2, ensure_ascii=False)
    
    return response

def get_all_chat_history(user_name, logs_dir):
    history = []
    for filename in os.listdir(logs_dir):
        if filename.endswith('_response_log.csv'):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)  # Changed to DictReader
                    for row in reader:
                        # Update date parsing to handle both formats
                        try:
                            date = datetime.strptime(row['date'], '%d-%m-%Y').strftime('%d-%m-%Y')
                        except ValueError:
                            try:
                                date = datetime.strptime(row['date'], '%Y-%m-%d').strftime('%d-%m-%Y')
                            except ValueError:
                                continue
                            
                        if row['name'] == user_name:
                            history.append({
                                "name": row['name'],
                                "date": date,
                                "time": row.get('time', ""),
                                "question": row.get('question', ""),
                                "response": row.get('response', ""),
                                "unique_files": row.get('unique_files', ""),
                                "chunk_info": [
                                    row.get('chunk1_score', ""),
                                    row.get('chunk2_score', ""),
                                    row.get('chunk3_score', "")
                                ]
                            })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                continue
                
    return sorted(history, key=lambda x: (
        datetime.strptime(x['date'], '%d-%m-%Y'),
        x['time']
    ), reverse=True)

def load_today_history():
    current_date = datetime.now().strftime("%d-%m-%Y")
    history_dir = os.path.join(script_dir, "history")
    today_file = os.path.join(history_dir, f"{current_date}_conversation_history.md")
    

    if os.path.exists(today_file):
        try:
            with open(today_file, 'r', encoding='utf-8') as file:
                content = file.read()
                logging.info(f"Successfully loaded today's history: {len(content)} characters")
                return content
        except Exception as e:
            logging.error(f"Error reading today's history: {str(e)}", exc_info=True)
            return ""
    else:
        logging.info("No history file found for today")
        return ""
    
def get_temporal_context(today_history, max_history_chunks=5):
    """Process conversation history with temporal weighting"""
    history_chunks = today_history.split("##")
    
    # Sort chunks by timestamp (assuming they start with timestamp)
    history_chunks.sort(key=lambda x: x.split("|")[0] if "|" in x else "", reverse=True)
    
    # Take most recent chunks and apply temporal weighting
    recent_chunks = []
    for i, chunk in enumerate(history_chunks[:max_history_chunks]):
        temporal_weight = 1 / (i + 1)  # More recent = higher weight
        recent_chunks.append({
            'content': chunk,
            'weight': temporal_weight
        })
    
    return recent_chunks    

def assemble_enhanced_context(
    user_name: str,
    prompt: str,
    context_manager: EnhancedContextManager,
    top_chunks: List[tuple],
    today_history: str
) -> dict:
    """
    Assembles context with improved structure and weighting
    """
    # Get weighted context from memory
    categorized_context = context_manager.get_weighted_context(prompt, user_name)
    
    # Process and categorize RAG chunks with relevance scores
    rag_context = {
        'authoritative': [],
        'historical': [],
        'student_specific': [],
        'recent_interactions': []
    }
    
    # Process RAG chunks with scores
    for chunk, filename in top_chunks:
        # Calculate chunk relevance (assuming cosine similarity score is available)
        relevance_score = cosine_similarity(
            vectorizer.transform([prompt]), 
            vectorizer.transform([chunk])
        )[0][0]
        
        context_item = {
            'content': chunk,
            'source': filename,
            'relevance': relevance_score
        }
        
        # Extract student name from filename (remove .txt and path)
        filename_base = os.path.splitext(os.path.basename(filename))[0]
        
        # Check if this is a student file (either in filename or content)
        is_student_file = (
            "students/" in filename.lower() or  # Check if file is in students directory
            any(name.lower() in filename.lower() for name in ["student:", "student name:", "name:"]) or  # Check for student name markers
            any(name.lower() in chunk.lower() for name in ["student:", "student name:", "name:"])  # Check content for student name markers
        )
        
        # Categorize based on source and content
        if 'documents' in filename.lower():
            rag_context['authoritative'].append(context_item)
        elif 'history' in filename.lower():
            rag_context['historical'].append(context_item)
        elif (is_student_file or 
              is_name_match(user_name, filename_base) or 
              any(is_name_match(user_name, name) for name in chunk.split())):
            rag_context['student_specific'].append(context_item)
    
    # Sort each category by relevance
    for category in rag_context:
        rag_context[category] = sorted(
            rag_context[category],
            key=lambda x: x['relevance'],
            reverse=True
        )[:3]  # Keep top 3 most relevant chunks per category
    
    return rag_context

def get_ai_response(user_name, prompt, manual_temperature=None, manual_effort=None):
    try:
        # Log the model being used
        current_provider = MODEL_CONFIG["current_provider"]
        current_model = MODEL_CONFIG["providers"][current_provider]["model"]
        logger.info(f"Using model: {current_provider} - {current_model}")

        # Get document chunks and compute relevance
        weighted_similarities = weight_context_chunks(
            prompt,
            document_chunks_with_filenames,
            vectorizer,
            tfidf_matrix
        )

        # Get top chunks based on weighted similarities
        top_indices = weighted_similarities.argsort()[-5:][::-1]  # Get top 5 chunks
        top_chunks = [document_chunks_with_filenames[i] for i in top_indices]

        # Extract unique filenames from top chunks
        unique_files = list(set(filename for _, filename in top_chunks))

        today_history = load_today_history()
        api_handler = ModelAPIHandler(MODEL_CONFIG)

        # Get mode parameters with explicit mode handling
        mode_params = st.session_state.cognitive_modes.get_mode_parameters(prompt)
        selected_mode = mode_params.get('mode', 'survey')  # Default to survey if mode is missing

        # Resolve the generation controls this model actually accepts. Sending
        # temperature to a model that rejects it returns a 400, so capability
        # comes first and the requested value second.
        capabilities = api_handler.capabilities
        supports_sampling = bool(capabilities.get('sampling'))
        supports_effort = bool(capabilities.get('effort_levels'))

        effective_temperature = None
        effective_effort = None

        if supports_sampling:
            if manual_temperature is not None:
                effective_temperature = manual_temperature
                control_source = "manual (temperature)"
            else:
                effective_temperature = mode_params['temperature']
                control_source = f"auto ({selected_mode})"
        elif supports_effort:
            if manual_effort is not None:
                effective_effort = manual_effort
                control_source = "manual (effort)"
            else:
                effective_effort = mode_params['effort']
                if effective_effort not in capabilities['effort_levels']:
                    effective_effort = 'high'
                control_source = f"auto ({selected_mode})"
        else:
            # No generation controls available on this model: the prompt does
            # all the steering.
            control_source = f"auto ({selected_mode}, prompt only)"

        depth = resolve_depth(
            selected_mode, temperature=effective_temperature, effort=effective_effort
        )
        logger.info(
            f"Generation controls: temperature={effective_temperature} "
            f"effort={effective_effort} depth={depth} source={control_source}"
        )

        
        # Get enhanced context structure
        rag_context = assemble_enhanced_context(
            user_name=user_name,
            prompt=prompt,
            context_manager=st.session_state.context_manager,
            top_chunks=top_chunks,
            today_history=today_history
        )
        
        # Get the character prompt
        character_prompt = get_patrick_prompt()

        # Depth-aware dynamic instructions, driven by the resolved depth band
        # rather than a raw temperature value.
        character_prompt += DEPTH_GUIDANCE[depth]

        # Add cognitive mode-specific subtle guidance (only in Auto mode)
        if manual_temperature is None and manual_effort is None:
            mode_guidance_map = {
                'survey': " The question calls for careful observation and diagnosis.",
                'synthesis': " The question invites connection-making across domains.",
                'proposition': " The question opens space for speculative intervention."
            }
            mode_guidance = mode_guidance_map.get(selected_mode, "")
            character_prompt += mode_guidance

        # Construct prompt with organic context integration
        # Combine all context without rigid labeling
        all_context = []
        all_context.extend(chunk['content'] for chunk in rag_context['authoritative'])
        all_context.extend(chunk['content'] for chunk in rag_context['student_specific'])
        all_context.extend(chunk['content'] for chunk in rag_context['historical'])

        context_text = '\n\n'.join(all_context) if all_context else ""

        structured_prompt = f"""{context_text}

{user_name} asks: {prompt}"""

        # Prepare API request with structured prompt and character prompt.
        # The handler returns a normalised ModelResponse whatever the provider,
        # so there is no response-shape guessing left to do here.
        model_response = api_handler.make_request(
            structured_prompt,
            system_prompt=character_prompt,
            temperature=effective_temperature,
            effort=effective_effort,
        )
        response_content = model_response.text

        if model_response.stop_reason == "max_tokens":
            st.warning(
                "The response was cut short by the max_tokens limit. "
                "Raise max_tokens in MODEL_CONFIG if this keeps happening."
            )

        # Parse XML-style tags if present, but don't force them
        import re
        think_match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', response_content, re.DOTALL)

        if think_match and answer_match:
            # Structured response with reasoning
            reasoning = think_match.group(1).strip()
            answer = answer_match.group(1).strip()
        else:
            # Free-form response - no forced structure
            reasoning = ""
            answer = response_content.strip()

        # Clean up any remaining markdown or special characters
        answer = answer.replace("\\n", "\n").replace("\\'", "')")
        
        # Log evaluation with explicit mode
        logger.info(f"Starting response evaluation for mode: {selected_mode}")
        evaluation_results = st.session_state.response_evaluator.evaluate_response(
            response=answer,  # Only evaluate the answer portion
            mode=selected_mode,  # Pass the explicit mode
            temperature=effective_temperature,
            effort=effective_effort
        )
        logger.info(f"Evaluation results: {evaluation_results}")
        
        # Create chunk info with scores
        chunk_info = [
            f"{filename} (score: {weighted_similarities[idx]:.4f})"
            for idx, (_, filename) in enumerate(top_chunks)
        ]

        # Return the generation metadata alongside the response. Token usage is
        # carried through so the dashboard can report spend per response, not
        # just response length.
        generation_info = {
            'temperature': effective_temperature,
            'effort': effective_effort,
            'depth': depth,
            'source': control_source,
            'mode': selected_mode,
            'provider': current_provider,
            'model': current_model,
            'usage': model_response.usage,
            'stop_reason': model_response.stop_reason,
        }

        return (reasoning, answer), unique_files, chunk_info, generation_info

    except Exception as e:
        logger.error(f"Error in get_ai_response: {str(e)}")
        return (
            f"An unexpected error occurred: {str(e)}",
            [],
            [],
            {
                'temperature': None,
                'effort': None,
                'depth': 'balanced',
                'source': 'error',
                'mode': 'unknown',
                'provider': MODEL_CONFIG['current_provider'],
                'model': MODEL_CONFIG['providers'][MODEL_CONFIG['current_provider']]['model'],
                'usage': {},
                'stop_reason': None,
            },
        )

# Initialize session state objects
if 'cognitive_modes' not in st.session_state:
    logger.info("Initializing new GeddesCognitiveModes")
    st.session_state.cognitive_modes = GeddesCognitiveModes()

if 'context_manager' not in st.session_state:
    st.session_state.context_manager = EnhancedContextManager()

if 'response_evaluator' not in st.session_state:
    logger.info("Initializing new ResponseEvaluator")
    from admin_dashboard import ResponseEvaluator
    st.session_state.response_evaluator = ResponseEvaluator()

# Streamlit UI
st.title("The Ghost of Geddes...")

# Sidebar for About information and model selection
about_content, contains_html = get_about_info()
st.sidebar.header("About")
if contains_html:
    st.sidebar.markdown(about_content, unsafe_allow_html=True)
else:
    st.sidebar.info(about_content)

# Model selection dropdown
st.sidebar.header("Model Settings")
selected_provider = st.sidebar.selectbox(
    "Select AI Model",
    options=list(MODEL_CONFIG["providers"].keys()),
    index=list(MODEL_CONFIG["providers"].keys()).index(MODEL_CONFIG["current_provider"])
)

# Apply the provider choice before anything reads the capabilities, so the
# controls below describe the model that will actually be called.
if selected_provider != MODEL_CONFIG["current_provider"]:
    MODEL_CONFIG["current_provider"] = selected_provider
    st.sidebar.success(f"Switched to {selected_provider} model")

api_handler = ModelAPIHandler(MODEL_CONFIG)

if selected_provider == "ollama":
    available_models = api_handler.get_available_ollama_models()
    if available_models:
        current = MODEL_CONFIG["providers"]["ollama"]["model"]
        selected_model = st.sidebar.selectbox(
            "Select Ollama Model",
            options=available_models,
            index=available_models.index(current) if current in available_models else 0
        )
        MODEL_CONFIG["providers"]["ollama"]["model"] = selected_model
    else:
        st.sidebar.warning("Could not fetch available Ollama models. Please ensure Ollama server is running.")
else:
    # Ask Anthropic what it offers rather than trusting a list baked into this
    # file. Falls back to the static registry when the call fails or no key is
    # configured.
    if "discovered_anthropic_models" not in st.session_state:
        st.session_state.discovered_anthropic_models = api_handler.get_available_anthropic_models()

    discovered = st.session_state.discovered_anthropic_models
    catalogue = discovered or ANTHROPIC_MODELS
    model_ids = sorted(catalogue.keys())
    current = MODEL_CONFIG["providers"]["anthropic"]["model"]
    if current not in model_ids:
        model_ids.insert(0, current)

    def _label(model_id):
        entry = catalogue.get(model_id, {})
        label = entry.get("display_name") or model_id
        return f"{label} (deprecated)" if entry.get("deprecated") else label

    selected_model = st.sidebar.selectbox(
        "Select Anthropic Model",
        options=model_ids,
        index=model_ids.index(current),
        format_func=_label,
    )
    MODEL_CONFIG["providers"]["anthropic"]["model"] = selected_model

    if not discovered:
        st.sidebar.caption("Live model list unavailable - showing the built-in registry.")
    if st.sidebar.button("Refresh model list"):
        st.session_state.discovered_anthropic_models = api_handler.get_available_anthropic_models()
        st.rerun()

# Rebuild the handler so the capability read below reflects the chosen model.
api_handler = ModelAPIHandler(MODEL_CONFIG)
model_capabilities = api_handler.capabilities

# Response depth control. Which control appears depends on what the selected
# model accepts: current Anthropic models reject temperature and take
# output_config.effort instead.
st.sidebar.header("Response Depth")
manual_temperature = None
manual_effort = None

if model_capabilities.get("sampling"):
    temperature_mode = st.sidebar.radio(
        "Temperature Mode",
        options=["Auto (Cognitive Mode)", "Manual"],
        help="Auto uses temperature based on cognitive mode (Survey: 0.7, Synthesis: 0.8, Proposition: 0.9). Manual lets you set a custom temperature."
    )
    if temperature_mode == "Manual":
        manual_temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values (0.8-1.0) = more creative/random. Lower values (0.0-0.5) = more focused/deterministic."
        )
        st.sidebar.caption(f"Current: {manual_temperature:.2f}")
    else:
        st.sidebar.caption("Temperature will be set automatically based on query type")
elif model_capabilities.get("effort_levels"):
    levels = model_capabilities["effort_levels"]
    effort_mode = st.sidebar.radio(
        "Effort Mode",
        options=["Auto (Cognitive Mode)", "Manual"],
        help="Auto sets effort from the cognitive mode (Survey: medium, Synthesis: high, Proposition: xhigh). Effort controls how much the model thinks before answering."
    )
    if effort_mode == "Manual":
        default_effort = MODEL_CONFIG["providers"]["anthropic"].get("effort", "high")
        manual_effort = st.sidebar.select_slider(
            "Effort",
            options=levels,
            value=default_effort if default_effort in levels else levels[-1],
            help="Higher effort means deeper reasoning and more tokens spent."
        )
        st.sidebar.caption(f"Current: {manual_effort}")
    else:
        st.sidebar.caption("Effort will be set automatically based on query type")
    st.sidebar.caption("This model does not accept a temperature setting.")
else:
    st.sidebar.caption(
        "This model accepts no generation controls - depth is steered by the "
        "prompt alone."
    )

# Sidebar: Data controls
st.sidebar.header("Data Controls")
if st.sidebar.button("Reload documents (RAG)"):
    try:
        # Clear caches to force reload
        st.cache_data.clear()
        st.cache_resource.clear()

        # Reload documents and recompute TF-IDF
        document_chunks_with_filenames = load_documents(['documents', 'history', 'students'])
        vectorizer, tfidf_matrix = compute_tfidf_matrix(document_chunks_with_filenames)
        st.sidebar.success("Documents reloaded and index recomputed.")
        logger.info("RAG documents reloaded and TF-IDF recomputed via sidebar control")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {str(e)}")
        logger.error(f"RAG reload failed: {str(e)}")

# Introduction section with image and personal introduction
col1, col2 = st.columns([0.8, 3.2])
with col1:
    try:
        st.image("images/patrick_geddes.jpg", width=130)
    except Exception as e:
        st.write("Image not available")

with col2:
    st.markdown("""
    Greetings, dear inquirer! I am Patrick Geddes, a man of many hats - biologist, sociologist, geographer, and yes, a bit of a revolutionary in the realm of town planning, if I do say so myself. 
    
    Now, my eager student, what's your name? And more importantly, what burning question about our shared world shall we explore together? 
    Remember, "By leaves we live" - so let your curiosity bloom and ask away!
    """, unsafe_allow_html=True)

# Input section for user queries
user_name_input = st.text_input("Enter your name:")
prompt_input = st.text_area("Discuss your project with Patrick:")

if st.button('Submit'):
    if user_name_input and prompt_input:
        with st.spinner('Re-animating Geddes Ghost...'):
            try:
                # Get the latest file paths
                csv_file, json_file = initialize_log_files()

                # Get response and update logs
                response_content, unique_files, chunk_info, generation_info = get_ai_response(
                    user_name_input.strip(),
                    prompt_input.strip(),
                    manual_temperature=manual_temperature,
                    manual_effort=manual_effort
                )
                
                # Check for error messages in response
                if isinstance(response_content, str) and "error" in response_content.lower():
                    st.error(response_content)
                    st.stop()
                
                # Unpack reasoning and answer
                reasoning, answer = response_content
                
                # If successful, update logs and display response
                encoded_response = update_chat_logs(
                    user_name=user_name_input.strip(),
                    question=prompt_input.strip(),
                    response=answer,  # Store just the answer portion
                    unique_files=unique_files,
                    chunk_info=chunk_info,
                    csv_file=csv_file,
                    json_file=json_file,
                    generation_info=generation_info
                )

                # Add this line to write markdown history
                write_markdown_history(
                    user_name=user_name_input.strip(),
                    question=prompt_input.strip(),
                    response=answer,  # Store just the answer portion
                    csv_file=csv_file
                )

                # Play sound only on successful response (if audio is available)
                if audio_available and ding_sound:
                    ding_sound.play()
                
                # Add custom CSS for the response sections
                st.markdown("""
                <style>
                .reasoning-section {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-left: 4px solid #4a90e2;
                    padding: 20px;
                    margin-bottom: 25px;
                    border-radius: 5px;
                }
                .answer-section {
                    padding: 20px;
                    margin-bottom: 25px;
                    border-left: 4px solid #ffa500;
                }
                .metadata-section {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    padding: 15px;
                    margin-top: 15px;
                    border-radius: 5px;
                    font-size: 0.9em;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display reasoning section
                if reasoning:
                    st.markdown("""
                    <div class="reasoning-section">
                        <p style="color: #4a90e2; font-weight: bold; margin-bottom: 15px;">🤔 Patrick Geddes thinks:</p>
                        <p style="font-style: italic; color: #495057;">{}</p>
                    </div>
                    """.format(html.escape(reasoning).replace('\n', '<br>')), unsafe_allow_html=True)
                
                # Display answer section - simplified
                st.markdown("### 💭 Patrick Geddes says:")
                st.markdown(f"_{answer}_")
                
                # Display metadata in a cleaner format
                st.markdown("""
                <div class="metadata-section">
                    <p style="color: #666; margin-bottom: 5px;"><strong>📚 Sources:</strong> {}</p>
                    <p style="color: #666; margin-bottom: 5px;"><strong>🔍 Relevance:</strong> {}</p>
                    <p style="color: #666; margin-bottom: 5px;"><strong>🧭 Depth:</strong> {} ({})</p>
                    <p style="color: #666; margin-bottom: 0;"><strong>🧮 Tokens:</strong> {}</p>
                </div>
                """.format(
                    ' • '.join(html.escape(file) for file in unique_files),
                    ' • '.join(html.escape(chunk) for chunk in chunk_info),
                    html.escape(format_depth_control(generation_info)),
                    html.escape(generation_info['source']),
                    html.escape(format_usage(generation_info.get('usage')))
                ), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
    else:
        st.warning("Please enter both your name and a question.")


# Chat history button
if st.button('Show Chat History'):
    logs_dir = os.path.join(script_dir, "logs")
    history = get_all_chat_history(user_name_input, logs_dir)
    for entry in history:
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="color: black; font-weight: bold;">Name: {entry['name']}</p>
        <p style="color: black; font-weight: bold;">Date: {entry['date']} | Time: {entry['time']}</p>
        <p style="color: #FFA500; font-weight: bold;">Question:</p>
        <p>{entry['question']}</p>
        <p style="color: #FFA500; font-weight: bold;">Patrick Geddes:</p>
        <p>{entry['response']}</p>
        <p style="color: black; font-weight: bold;">Sources:</p>
        <p>{entry['unique_files']}</p>
        <p style="color: black; font-weight: bold;">Document relevance:</p>
        {' - '.join(html.escape(str(chunk)) if chunk is not None else '' for chunk in entry['chunk_info'])}
        """, unsafe_allow_html=True)