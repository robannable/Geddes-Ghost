"""
With The Ancients - AI Assistant System
----------------------------------------
A Retrieval-Augmented Generation (RAG) application for conversations with
historical figures. Inspired by Helen de Cruz's work on philosophical dialogue
with the past. The system includes:

- Multi-character conversation system (single, parallel, dialogue modes)
- Character configuration system for easily adding new historical figures
- TF-IDF-based retrieval over character-specific knowledge bases
- Dynamic cognitive modes with temperature variation
- Context-aware response generation via Anthropic Claude (default) or Ollama
- Comprehensive logging and Admin Dashboard for analytics

Conversation Modes:
1. Single Character: Traditional one-on-one dialogue
2. Multiple Perspectives: Get responses from several figures simultaneously
3. Character Dialogue: Watch historical figures discuss topics together

Admin Dashboard views:
1. Performance, Document Usage, User Analysis, Response Metrics
2. Topics Map (topic heatmap and doc-to-topic contribution)
3. Reflections (sentiment/keywords and action items)
4. Interventions (auto-generated teaching plan)

Document categories used for retrieval:
1. Character-specific documents (core knowledge from their writings)
2. Historical records (past interactions)
3. Student-specific content (personalized context)

Author: Rob Annable (Birmingham School of Architecture)
Inspired by: Helen de Cruz's "Thinking with the Ancients"
Last Updated: 2025-01-23
Version: 2.0 - Multi-Character System
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

# Import character configuration system
from character_config import (
    CharacterConfig,
    CharacterManager,
    CognitiveModeSystem,
    CreativeMarkerDetector
)

# Import multi-character conversation system
from multi_character import (
    MultiCharacterConversation,
    MultiCharacterResponseGenerator,
    create_multi_character_session
)
import html
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
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

# Initialize pygame for audio
pygame.mixer.init()

# Load sound file
ding_sound = pygame.mixer.Sound(os.path.join(sound_dir, 'ding2.wav'))

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

# DEPRECATED: Old GeddesCognitiveModes class replaced by character_config.CognitiveModeSystem
# Kept for reference during migration
# class GeddesCognitiveModes:
#     ... (moved to character_config.py as CognitiveModeSystem)

# Load model config from file or notepad (for now, hardcode as a dict)
MODEL_CONFIG = {
    "current_provider": "anthropic",
    "providers": {
        "anthropic": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "message_retention": "no_retention",
            "api_endpoint": "https://api.anthropic.com/v1/messages",
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
            "headers": {
                "Content-Type": "application/json"
            }
        }
    }
}

class ModelAPIHandler:
    def __init__(self, config):
        self.config = config
        self.provider = config["current_provider"]
        self.provider_config = config["providers"][self.provider]
        self.api_key_env = self.provider_config.get("api_key_env")
        self.api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        self.api_endpoint = self.provider_config["api_endpoint"]
        self.headers = self.provider_config["headers"].copy()
        if self.api_key:
            # Add API key to headers if needed
            if self.provider == "anthropic":
                self.headers["x-api-key"] = self.api_key

    def get_available_ollama_models(self):
        """Fetch available models from Ollama server"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def make_request(self, prompt, system_prompt=None, temperature=None):
        # Use provided temperature or fall back to config default
        effective_temperature = temperature if temperature is not None else self.provider_config["temperature"]

        if self.provider == "anthropic":
            payload = {
                "model": self.provider_config["model"],
                "max_tokens": self.provider_config["max_tokens"],
                "temperature": effective_temperature,
                "top_p": self.provider_config["top_p"],
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            if system_prompt:
                payload["system"] = system_prompt
        elif self.provider == "ollama":
            payload = {
                "model": self.provider_config["model"],
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "stream": False,
                "options": {
                    "temperature": effective_temperature,
                    "top_p": self.provider_config["top_p"],
                    "num_predict": self.provider_config["max_tokens"]
                }
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        response = requests.post(self.api_endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        
        # Handle Ollama's response format
        if self.provider == "ollama":
            try:
                response_data = response.json()
                if isinstance(response_data, dict) and "response" in response_data:
                    return {"content": response_data["response"]}
                else:
                    raise ValueError(f"Unexpected Ollama response format: {response_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding Ollama response: {str(e)}")
                raise ValueError(f"Error decoding Ollama response: {str(e)}")
        
        return response.json()

dotenv.load_dotenv()

def check_api_connection():
    """Check if we can connect to the Anthropic API"""
    try:
        api_handler = ModelAPIHandler(MODEL_CONFIG)
        # Do a minimal API call
        response = api_handler.make_request("test")
        return True
    except Exception as e:
        logger.error(f"API connection error: {str(e)}")
        return False

@st.cache_data
def get_character_prompt(character_config: CharacterConfig):
    """
    Get system prompt for a character, with inclusive language guidance.

    Args:
        character_config: CharacterConfig object for the character

    Returns:
        Complete system prompt string
    """
    try:
        prompt = character_config.get_system_prompt()
        # Add inclusive language guidance
        prompt += "\n\nWhen responding to users, consider their name and potential gender implications. Avoid making assumptions based on stereotypes and strive for inclusive language. Adapt your language and examples to be appropriate for all users, regardless of their perceived gender."
        return prompt
    except Exception as e:
        logger.error(f"Error loading character prompt: {e}")
        # Fallback to basic prompt
        return character_config._generate_fallback_prompt() + "\n\nWhen responding to users, consider their name and potential gender implications. Avoid making assumptions based on stereotypes and strive for inclusive language."

# DEPRECATED: Old function kept for compatibility during migration
def get_patrick_prompt():
    """Legacy function - redirects to character config system."""
    if 'current_character' in st.session_state:
        return get_character_prompt(st.session_state.current_character)
    # Emergency fallback
    return "You are a helpful AI assistant engaged in thoughtful conversation."

@st.cache_data
def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True  # Contains HTML
    except FileNotFoundError:
        st.warning(f"'{about_file_path}' not found. Using default about info.")
        return "This app uses advanced AI models to simulate a conversation with Patrick Geddes...", False

@st.cache_data
def load_documents(directories=['documents', 'history', 'students'], character_id=None):
    """
    Load documents for RAG system.

    Args:
        directories: List of directory types to load ('documents', 'history', 'students')
        character_id: If provided, loads character-specific documents from characters/{character_id}/documents

    Returns:
        List of (text, filename) tuples
    """
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
        'images',
        'characters'  # Don't load the entire characters directory
    }

    for directory in directories:
        # Handle character-specific documents directory
        if directory == 'documents' and character_id:
            dir_path = os.path.join(script_dir, 'characters', character_id, 'documents')
        else:
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
# Note: This initial load uses 'patrick_geddes' as default. Will reload when character changes.
document_chunks_with_filenames = load_documents(['documents', 'history', 'students'], character_id='patrick_geddes')
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
                'model_provider', 'model_name'
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

def update_chat_logs(user_name, question, response, unique_files, chunk_info, csv_file, json_file, temperature_info=None):
    """Update both CSV and JSON logs with chat data"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Use temperature_info if provided, otherwise fall back to cognitive mode detection
    if temperature_info:
        current_mode = temperature_info['mode']
        temperature = temperature_info['temperature']
        temperature_source = temperature_info['source']
    else:
        mode_params = st.session_state.cognitive_modes.get_mode_parameters(question)
        current_mode = mode_params['mode']
        temperature = mode_params['temperature']
        temperature_source = "auto (legacy)"

    current_provider = MODEL_CONFIG["current_provider"]
    current_model = MODEL_CONFIG["providers"][current_provider]["model"]

    # Get evaluation with accumulated metrics
    evaluation_results = st.session_state.response_evaluator.evaluate_response(
        response=response,
        mode=current_mode,
        temperature=temperature,
        temperature_source=temperature_source
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
        'model_name': current_model
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

def get_ai_response(user_name, prompt, manual_temperature=None):
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

        # Determine which temperature to use
        if manual_temperature is not None:
            effective_temperature = manual_temperature
            temperature_source = "manual"
            logger.info(f"Using manual temperature: {effective_temperature}")
        else:
            effective_temperature = mode_params['temperature']
            temperature_source = f"auto ({selected_mode})"
            logger.info(f"Using cognitive mode temperature: {effective_temperature} (mode: {selected_mode})")
        
        # Get enhanced context structure
        rag_context = assemble_enhanced_context(
            user_name=user_name,
            prompt=prompt,
            context_manager=st.session_state.context_manager,
            top_chunks=top_chunks,
            today_history=today_history
        )
        
        # Get the character prompt from current character config
        current_char = st.session_state.current_character
        character_prompt = get_character_prompt(current_char)

        # Add temperature-aware dynamic instructions from character config
        temperature_guidance = current_char.get_temperature_guidance(effective_temperature)
        if temperature_guidance:
            character_prompt += "\n\n" + temperature_guidance

        # Add cognitive mode-specific subtle guidance (only in Auto mode)
        if manual_temperature is None:  # Only add mode guidance in Auto mode
            # Get mode guidance from cognitive mode system
            mode_params = st.session_state.cognitive_modes.get_mode_parameters(prompt)
            mode_guidance = mode_params.get('guidance', '')
            if mode_guidance:
                character_prompt += " " + mode_guidance

        # Construct prompt with organic context integration
        # Combine all context without rigid labeling
        all_context = []
        all_context.extend(chunk['content'] for chunk in rag_context['authoritative'])
        all_context.extend(chunk['content'] for chunk in rag_context['student_specific'])
        all_context.extend(chunk['content'] for chunk in rag_context['historical'])

        context_text = '\n\n'.join(all_context) if all_context else ""

        structured_prompt = f"""{context_text}

{user_name} asks: {prompt}"""

        # Prepare API request with structured prompt and character prompt
        response_json = api_handler.make_request(structured_prompt, system_prompt=character_prompt, temperature=effective_temperature)
        
        # Handle different API response formats
        if api_handler.provider == "anthropic":
            if isinstance(response_json, dict):
                if "content" in response_json:
                    response_content = response_json["content"]
                elif "message" in response_json and "content" in response_json["message"]:
                    response_content = response_json["message"]["content"]
                elif "choices" in response_json and len(response_json["choices"]) > 0:
                    response_content = response_json["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Unexpected Anthropic response format: {response_json}")
                    raise ValueError(f"Unexpected Anthropic API response format: {response_json}")
            else:
                logger.error(f"Unexpected response type: {type(response_json)}")
                raise ValueError(f"Unexpected response type: {type(response_json)}")
        elif api_handler.provider == "ollama":
            if isinstance(response_json, dict) and "content" in response_json:
                response_content = response_json["content"]
            else:
                raise ValueError(f"Unexpected Ollama API response format: {response_json}")
        else:
            raise ValueError(f"Unsupported provider: {api_handler.provider}")
        
        # Handle Anthropic's content block format
        if isinstance(response_content, list):
            # Extract text from content blocks
            response_content = " ".join(
                block.get("text", "") for block in response_content 
                if isinstance(block, dict) and "text" in block
            )
        
        # Ensure response_content is a string
        if not isinstance(response_content, str):
            logger.error(f"Response content is not a string: {type(response_content)}")
            response_content = str(response_content)

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
            temperature=mode_params['temperature']
        )
        logger.info(f"Evaluation results: {evaluation_results}")
        
        # Create chunk info with scores
        chunk_info = [
            f"{filename} (score: {weighted_similarities[idx]:.4f})"
            for idx, (_, filename) in enumerate(top_chunks)
        ]

        # Return temperature info along with response
        temperature_info = {
            'temperature': effective_temperature,
            'source': temperature_source,
            'mode': selected_mode
        }

        return (reasoning, answer), unique_files, chunk_info, temperature_info

    except Exception as e:
        logger.error(f"Error in get_ai_response: {str(e)}")
        return f"An unexpected error occurred: {str(e)}", [], [], {'temperature': 0.7, 'source': 'error', 'mode': 'unknown'}


def get_multi_character_responses(character_ids, user_query, context_text=""):
    """
    Get responses from multiple characters in parallel.

    Args:
        character_ids: List of character IDs to query
        user_query: User's question
        context_text: RAG context

    Returns:
        Formatted multi-character response string
    """
    try:
        # Create multi-character session
        multi_conv = create_multi_character_session(character_ids, st.session_state.character_manager)

        # Create response generator
        api_handler = ModelAPIHandler(MODEL_CONFIG)
        generator = MultiCharacterResponseGenerator(api_handler)

        # Generate parallel responses
        responses = generator.generate_parallel_responses(
            multi_conv,
            user_query,
            context_text=context_text
        )

        # Format for display
        formatted = multi_conv.format_parallel_responses(responses)

        return formatted

    except Exception as e:
        logger.error(f"Error in multi-character response: {str(e)}")
        return f"Error generating multi-character responses: {str(e)}"


def get_character_dialogue(character_ids, topic, context_text="", num_turns=2):
    """
    Generate a dialogue between multiple characters.

    Args:
        character_ids: List of character IDs to participate
        topic: Discussion topic
        context_text: RAG context
        num_turns: Number of turns per character

    Returns:
        Formatted dialogue string
    """
    try:
        # Create multi-character session
        multi_conv = create_multi_character_session(character_ids, st.session_state.character_manager)

        # Create response generator
        api_handler = ModelAPIHandler(MODEL_CONFIG)
        generator = MultiCharacterResponseGenerator(api_handler)

        # Generate dialogue
        conversation_history = generator.generate_dialogue(
            multi_conv,
            topic,
            context_text=context_text,
            num_turns=num_turns
        )

        # Format for display
        formatted = multi_conv.format_dialogue(conversation_history, topic)

        return formatted

    except Exception as e:
        logger.error(f"Error in character dialogue: {str(e)}")
        return f"Error generating character dialogue: {str(e)}"


# Initialize Character Manager (manages all available characters)
if 'character_manager' not in st.session_state:
    logger.info("Initializing CharacterManager")
    st.session_state.character_manager = CharacterManager(characters_dir='characters')

# Initialize current character (default to Patrick Geddes)
if 'current_character' not in st.session_state:
    logger.info("Loading default character (Patrick Geddes)")
    st.session_state.current_character = st.session_state.character_manager.get_character('patrick_geddes')
    if st.session_state.current_character is None:
        st.error("Failed to load default character. Please check character configuration.")
        st.stop()

# Initialize cognitive modes for current character
if 'cognitive_modes' not in st.session_state or st.session_state.get('reload_character', False):
    logger.info(f"Initializing CognitiveModeSystem for {st.session_state.current_character.get_name()}")
    st.session_state.cognitive_modes = CognitiveModeSystem(st.session_state.current_character)
    st.session_state.reload_character = False

if 'context_manager' not in st.session_state:
    st.session_state.context_manager = EnhancedContextManager()

if 'response_evaluator' not in st.session_state:
    logger.info("Initializing new ResponseEvaluator")
    from admin_dashboard import ResponseEvaluator
    st.session_state.response_evaluator = ResponseEvaluator()

# Streamlit UI - use current character's page title
current_char = st.session_state.current_character
st.title(current_char.get_page_title())

# Sidebar - Conversation Mode Selection
st.sidebar.header("Conversation Mode")

conversation_mode = st.sidebar.radio(
    "Select mode:",
    ["Single Character", "Multiple Perspectives", "Character Dialogue"],
    help="Single: One character responds | Multiple: Several characters answer the same question | Dialogue: Characters discuss a topic"
)

# Initialize session state for selected characters
if 'selected_character_ids' not in st.session_state:
    st.session_state.selected_character_ids = ['patrick_geddes']

st.sidebar.markdown("---")

# Character Selection (different UI based on mode)
st.sidebar.header("Character Selection")

character_display_list = st.session_state.character_manager.get_character_display_list()

if conversation_mode == "Single Character":
    # Single character dropdown
    if character_display_list:
        display_to_id = {display: char_id for display, char_id in character_display_list}
        current_display = f"{current_char.get_name()} ({current_char.get_lifespan()})"

        selected_display = st.sidebar.selectbox(
            "Select Historical Figure",
            options=[display for display, _ in character_display_list],
            index=[display for display, _ in character_display_list].index(current_display) if current_display in [d for d, _ in character_display_list] else 0
        )

        # If character changed, update session state
        selected_id = display_to_id[selected_display]
        if selected_id != st.session_state.current_character.character_id:
            st.session_state.current_character = st.session_state.character_manager.get_character(selected_id)
            st.session_state.reload_character = True
            st.session_state.cognitive_modes = CognitiveModeSystem(st.session_state.current_character)
            st.session_state.selected_character_ids = [selected_id]
            st.rerun()

else:
    # Multiple character checkboxes
    st.sidebar.write("Select characters to include:")

    selected_chars = []
    for display, char_id in character_display_list:
        # Use session state to remember selections
        default_checked = char_id in st.session_state.selected_character_ids
        if st.sidebar.checkbox(display, value=default_checked, key=f"char_{char_id}"):
            selected_chars.append(char_id)

    # Update session state
    if selected_chars != st.session_state.selected_character_ids:
        st.session_state.selected_character_ids = selected_chars
        # If switching from single to multi, keep current character selected
        if len(selected_chars) == 0 and current_char.character_id not in selected_chars:
            st.session_state.selected_character_ids = [current_char.character_id]

    # Show count
    st.sidebar.caption(f"{len(st.session_state.selected_character_ids)} character(s) selected")

st.sidebar.markdown("---")

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

# Add Ollama model selection when Ollama is selected
if selected_provider == "ollama":
    api_handler = ModelAPIHandler(MODEL_CONFIG)
    available_models = api_handler.get_available_ollama_models()
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Select Ollama Model",
            options=available_models,
            index=available_models.index(MODEL_CONFIG["providers"]["ollama"]["model"]) if MODEL_CONFIG["providers"]["ollama"]["model"] in available_models else 0
        )
        MODEL_CONFIG["providers"]["ollama"]["model"] = selected_model
    else:
        st.sidebar.warning("Could not fetch available Ollama models. Please ensure Ollama server is running.")

# Update the current provider if changed
if selected_provider != MODEL_CONFIG["current_provider"]:
    MODEL_CONFIG["current_provider"] = selected_provider
    st.sidebar.success(f"Switched to {selected_provider} model")

# Temperature control section
st.sidebar.header("Temperature Control")
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
    manual_temperature = None
    st.sidebar.caption("Temperature will be set automatically based on query type")

# Sidebar: Data controls
st.sidebar.header("Data Controls")
if st.sidebar.button("Reload documents (RAG)"):
    try:
        # Clear caches to force reload
        st.cache_data.clear()
        st.cache_resource.clear()

        # Reload documents for current character and recompute TF-IDF
        current_character_id = st.session_state.current_character.character_id
        document_chunks_with_filenames = load_documents(['documents', 'history', 'students'], character_id=current_character_id)
        vectorizer, tfidf_matrix = compute_tfidf_matrix(document_chunks_with_filenames)
        st.sidebar.success(f"Documents reloaded for {st.session_state.current_character.get_name()}")
        logger.info(f"RAG documents reloaded for {current_character_id} and TF-IDF recomputed via sidebar control")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {str(e)}")
        logger.error(f"RAG reload failed: {str(e)}")

# Introduction section - changes based on conversation mode
if conversation_mode == "Single Character":
    # Single character introduction
    col1, col2 = st.columns([0.8, 3.2])
    with col1:
        try:
            portrait_path = current_char.get_portrait_path()
            if os.path.exists(portrait_path):
                st.image(portrait_path, width=130)
            else:
                st.write("Portrait not available")
        except Exception as e:
            st.write("Portrait not available")
            logger.error(f"Error loading portrait: {e}")

    with col2:
        st.markdown(current_char.get_greeting(), unsafe_allow_html=True)

elif conversation_mode == "Multiple Perspectives":
    # Show selected characters
    st.markdown("### Multiple Perspectives Mode")
    st.write(f"Ask a question and get responses from {len(st.session_state.selected_character_ids)} different historical figures:")

    # Show mini portraits if available
    if st.session_state.selected_character_ids:
        cols = st.columns(len(st.session_state.selected_character_ids))
        for idx, char_id in enumerate(st.session_state.selected_character_ids):
            char = st.session_state.character_manager.get_character(char_id)
            if char:
                with cols[idx]:
                    portrait_path = char.get_portrait_path()
                    if os.path.exists(portrait_path):
                        st.image(portrait_path, width=100)
                    st.caption(f"**{char.get_name()}**")

else:  # Character Dialogue
    st.markdown("### Character Dialogue Mode")
    st.write(f"Watch {len(st.session_state.selected_character_ids)} historical figures discuss a topic:")

    # Show selected characters
    if st.session_state.selected_character_ids:
        char_names = []
        for char_id in st.session_state.selected_character_ids:
            char = st.session_state.character_manager.get_character(char_id)
            if char:
                char_names.append(char.get_name())
        st.info(f"**Participants:** {', '.join(char_names)}")

# Input section for user queries (changes based on mode)
user_name_input = st.text_input("Enter your name:")

if conversation_mode == "Character Dialogue":
    prompt_input = st.text_area("Enter a topic for discussion:", placeholder="e.g., 'The role of density in creating vibrant urban neighborhoods'")
    num_turns = st.slider("Number of turns per character:", min_value=1, max_value=5, value=2)
else:
    if conversation_mode == "Single Character":
        prompt_input = st.text_area(current_char.get_input_label())
    else:
        prompt_input = st.text_area("Ask your question:")

if st.button('Submit'):
    if user_name_input and prompt_input:
        # Determine spinner message based on mode
        if conversation_mode == "Single Character":
            spinner_msg = current_char.get_loading_message()
        elif conversation_mode == "Multiple Perspectives":
            spinner_msg = f"Getting responses from {len(st.session_state.selected_character_ids)} characters..."
        else:
            spinner_msg = "Generating dialogue..."

        with st.spinner(spinner_msg):
            try:
                # Get the latest file paths
                csv_file, json_file = initialize_log_files()

                # Handle different conversation modes
                if conversation_mode == "Single Character":
                    # Original single character logic
                    response_content, unique_files, chunk_info, temperature_info = get_ai_response(
                        user_name_input.strip(),
                        prompt_input.strip(),
                        manual_temperature=manual_temperature
                    )

                    # Check for error messages in response
                    if isinstance(response_content, str) and "error" in response_content.lower():
                        st.error(response_content)
                        st.stop()

                    # Unpack reasoning and answer
                    reasoning, answer = response_content

                elif conversation_mode == "Multiple Perspectives":
                    # Multi-character parallel responses
                    if len(st.session_state.selected_character_ids) == 0:
                        st.error("Please select at least one character.")
                        st.stop()

                    # Get RAG context for the query
                    query_vector = vectorizer.transform([prompt_input.strip()])
                    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                    top_indices = similarities.argsort()[-5:][::-1]
                    top_chunks = [(document_chunks_with_filenames[i][0], document_chunks_with_filenames[i][1])
                                  for i in top_indices]
                    context_text = '\n\n'.join([chunk for chunk, _ in top_chunks])

                    # Generate multi-character responses
                    answer = get_multi_character_responses(
                        st.session_state.selected_character_ids,
                        prompt_input.strip(),
                        context_text=context_text
                    )

                    # For multi-character, we don't have reasoning/unique files like single mode
                    reasoning = ""
                    unique_files = []
                    chunk_info = []
                    temperature_info = {'temperature': 0.8, 'source': 'multi-character', 'mode': 'parallel'}

                else:  # Character Dialogue
                    # Multi-character dialogue
                    if len(st.session_state.selected_character_ids) < 2:
                        st.error("Please select at least two characters for a dialogue.")
                        st.stop()

                    # Get RAG context for the topic
                    query_vector = vectorizer.transform([prompt_input.strip()])
                    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                    top_indices = similarities.argsort()[-5:][::-1]
                    top_chunks = [(document_chunks_with_filenames[i][0], document_chunks_with_filenames[i][1])
                                  for i in top_indices]
                    context_text = '\n\n'.join([chunk for chunk, _ in top_chunks])

                    # Generate dialogue
                    answer = get_character_dialogue(
                        st.session_state.selected_character_ids,
                        prompt_input.strip(),
                        context_text=context_text,
                        num_turns=num_turns if 'num_turns' in locals() else 2
                    )

                    # For dialogue, we don't have reasoning/unique files like single mode
                    reasoning = ""
                    unique_files = []
                    chunk_info = []
                    temperature_info = {'temperature': 0.85, 'source': 'dialogue', 'mode': 'dialogue'}
                
                # If successful, update logs and display response
                encoded_response = update_chat_logs(
                    user_name=user_name_input.strip(),
                    question=prompt_input.strip(),
                    response=answer,  # Store just the answer portion
                    unique_files=unique_files,
                    chunk_info=chunk_info,
                    csv_file=csv_file,
                    json_file=json_file,
                    temperature_info=temperature_info
                )

                # Add this line to write markdown history
                write_markdown_history(
                    user_name=user_name_input.strip(),
                    question=prompt_input.strip(),
                    response=answer,  # Store just the answer portion
                    csv_file=csv_file
                )

                # Play sound only on successful response
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
                
                # Display response based on conversation mode
                if conversation_mode == "Single Character":
                    # Display reasoning section for single character
                    if reasoning:
                        st.markdown(f"""
                        <div class="reasoning-section">
                            <p style="color: #4a90e2; font-weight: bold; margin-bottom: 15px;">🤔 {html.escape(current_char.get_name())} thinks:</p>
                            <p style="font-style: italic; color: #495057;">{html.escape(reasoning).replace('\n', '<br>')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display answer section
                    st.markdown(f"### 💭 {current_char.get_name()} says:")
                    st.markdown(f"_{answer}_")

                    # Display metadata
                    st.markdown(f"""
                    <div class="metadata-section">
                        <p style="color: #666; margin-bottom: 5px;"><strong>📚 Sources:</strong> {' • '.join(html.escape(file) for file in unique_files)}</p>
                        <p style="color: #666; margin-bottom: 5px;"><strong>🔍 Relevance:</strong> {' • '.join(html.escape(chunk) for chunk in chunk_info)}</p>
                        <p style="color: #666; margin-bottom: 0;"><strong>🌡️ Temperature:</strong> {temperature_info['temperature']} ({html.escape(temperature_info['source'])})</p>
                    </div>
                    """, unsafe_allow_html=True)

                elif conversation_mode == "Multiple Perspectives":
                    # Multi-character response - answer already formatted
                    st.markdown(answer, unsafe_allow_html=True)

                else:  # Character Dialogue
                    # Dialogue response - answer already formatted
                    st.markdown(answer, unsafe_allow_html=True)
                
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