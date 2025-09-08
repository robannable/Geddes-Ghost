# admin_dashboard.py

import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta
import json
import csv
import os
import numpy as np
from collections import defaultdict
import time
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Get the script directory and create debug_logs folder
script_dir = os.path.dirname(os.path.abspath(__file__))
debug_logs_dir = os.path.join(script_dir, "debug_logs")
os.makedirs(debug_logs_dir, exist_ok=True)

# Configure logging to write to debug_logs folder
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(debug_logs_dir, 'admin_dashboard.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    def __init__(self):
        self.metrics = {
            'mode_distribution': {'survey': 0, 'synthesis': 0, 'proposition': 0},
            'response_lengths': [],
            'creative_markers': {
                'metaphor': 0,
                'ecological_reference': 0,
                'speculative_proposition': 0,
                'cross-disciplinary': 0
            },
            'temperature_effectiveness': {0.7: [], 0.8: [], 0.9: []}
        }
        logger.info("ResponseEvaluator initialized")
    
    def evaluate_response(self, response: str, mode: str, temperature: float) -> dict:
        # Update mode distribution
        self.metrics['mode_distribution'][mode] = self.metrics['mode_distribution'].get(mode, 0) + 1
        
        # Update response length
        response_length = len(response.split())
        self.metrics['response_lengths'].append(response_length)
        
        # Update temperature effectiveness
        if temperature in self.metrics['temperature_effectiveness']:
            self.metrics['temperature_effectiveness'][temperature].append(response_length)
        
        # Analyze for creative markers
        lower_response = response.lower()
        if any(word in lower_response for word in ['like', 'as', 'metaphor', 'akin']):
            self.metrics['creative_markers']['metaphor'] += 1
        if any(word in lower_response for word in ['ecology', 'nature', 'environment', 'organic']):
            self.metrics['creative_markers']['ecological_reference'] += 1
        if any(word in lower_response for word in ['could', 'might', 'suggest', 'propose']):
            self.metrics['creative_markers']['speculative_proposition'] += 1
        if any(word in lower_response for word in ['across', 'between', 'integrate', 'combine']):
            self.metrics['creative_markers']['cross-disciplinary'] += 1
        
        # Calculate averages for temperature effectiveness
        temp_effectiveness = {
            temp: sum(lengths) / len(lengths) if lengths else 0
            for temp, lengths in self.metrics['temperature_effectiveness'].items()
        }
        
        return {
            'mode_distribution': self.metrics['mode_distribution'],
            'avg_response_length': sum(self.metrics['response_lengths']) / len(self.metrics['response_lengths']),
            'creative_markers_frequency': dict(self.metrics['creative_markers']),
            'temperature_effectiveness': temp_effectiveness
        }
    
    def _check_creative_marker(self, response: str, marker: str) -> bool:
        """Check for presence of creative markers in response"""
        marker_patterns = {
            'metaphor': r'like|as if|resembles',
            'cross-disciplinary': r'biology|sociology|economics|art',
            'historical_parallel': r'historically|in the past|reminds me of',
            'ecological_reference': r'nature|ecosystem|organic',
            'speculative_proposition': r'what if|imagine|consider'
        }
        return bool(re.search(marker_patterns[marker], response.lower()))
    
    def _generate_evaluation_report(self) -> dict:
        """Generate summary report of metrics"""
        return {
            'mode_distribution': dict(self.metrics['mode_distribution']),
            'avg_response_length': np.mean(self.metrics['response_lengths']),
            'creative_markers_frequency': dict(self.metrics['creative_markers']),
            'temperature_effectiveness': {
                temp: np.mean(lengths)
                for temp, lengths in self.metrics['temperature_effectiveness'].items()
            }
        }

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == os.getenv("ADMIN_PASSWORD"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

def load_response_data(logs_dir):
    """Load and combine response data from all CSV files"""
    all_data = []
    csv_files = [f for f in os.listdir(logs_dir) if f.endswith('_response_log.csv')]
    
    if not csv_files:
        logger.error("No CSV files found in logs directory")
        return pd.DataFrame()
    
    for filename in csv_files:
        file_path = os.path.join(logs_dir, filename)
        logger.info(f"Processing file: {filename}")
        
        try:
            # Try reading with default settings first
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Successfully read {filename} with default settings")
            except Exception as e:
                logger.warning(f"Failed to read {filename} with default settings: {str(e)}")
                # Try different encodings
                encodings = ['utf-8-sig', 'latin1', 'utf-8']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            on_bad_lines='skip',
                            quoting=csv.QUOTE_MINIMAL,
                            escapechar='\\',
                            lineterminator='\n',
                            engine='python'
                        )
                        logger.info(f"Successfully read {filename} with {encoding} encoding")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to read {filename} with {encoding}: {str(e)}")
                        continue
            
            if df is None:
                logger.error(f"Could not read {filename} with any settings")
                continue
            
            # Log the columns found in the DataFrame
            logger.info(f"Columns in {filename}: {df.columns.tolist()}")
            
            # Check for missing columns and add them with None values
            new_columns = ['cognitive_mode', 'response_length', 'creative_markers', 'temperature', 'model_provider', 'model_name']
            for col in new_columns:
                if col not in df.columns:
                    df[col] = None
                    logger.info(f"Added missing column: {col}")
            
            # Basic data cleaning
            if 'date' in df.columns:
                try:
                    # Convert date strings to datetime objects
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # Clean up any problematic strings in the DataFrame
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).replace({r'\n': ' ', r'\r': ' '}, regex=True)
                            # Remove any null bytes or other problematic characters
                            df[col] = df[col].str.replace('\x00', '', regex=False)
                    all_data.append(df)
                    logger.info(f"Successfully processed {filename} with {len(df)} rows")
                except Exception as e:
                    logger.error(f"Error processing dates in {filename}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
            continue
    
    try:
        if all_data:
            # Ensure all DataFrames have the same columns
            columns = set()
            for df in all_data:
                columns.update(df.columns)
            
            logger.info(f"Combining DataFrames with columns: {columns}")
            
            # Add missing columns to each DataFrame
            for i in range(len(all_data)):
                for col in columns:
                    if col not in all_data[i].columns:
                        all_data[i][col] = None
            
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('date')
            logger.info(f"Total combined rows: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No data frames were successfully loaded")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error combining data frames: {str(e)}")
        return pd.DataFrame()

def analyze_chunk_scores(df):
    """Analyze document chunk relevance scores"""
    scores = []
    files = []
    
    # Check if required columns exist
    chunk_columns = ['chunk1_score', 'chunk2_score', 'chunk3_score']
    if not all(col in df.columns for col in chunk_columns):
        logger.warning("Missing chunk score columns in DataFrame")
        return pd.DataFrame({'score': [], 'file': []})
    
    for _, row in df.iterrows():
        for chunk_col in chunk_columns:
            try:
                if pd.notna(row[chunk_col]):
                    # Handle the score format: "filename (score: X.XXXX)"
                    score_str = str(row[chunk_col])
                    if '(' in score_str and ')' in score_str:
                        # Extract score and filename
                        score_part = score_str.split('(')[1].split(')')[0]
                        score = float(score_part.split(':')[1].strip())
                        filename = score_str.split('(')[0].strip()
                        scores.append(score)
                        files.append(filename)
            except Exception as e:
                logger.warning(f"Error processing chunk score: {str(e)}")
                continue
    
    return pd.DataFrame({'score': scores, 'file': files})

def display_chunk_analysis(df):
    st.subheader("Document Chunk Relevance Analysis")
    
    # Check if DataFrame is empty
    if df.empty:
        st.info("No data available for chunk analysis")
        return
        
    chunk_data = analyze_chunk_scores(df)
    
    if chunk_data.empty:
        st.info("No chunk score data available")
        return
    
    # Create visualizations
    avg_scores = chunk_data.groupby('file')['score'].agg(['mean', 'count']).reset_index()
    avg_scores = avg_scores.sort_values('mean', ascending=False)
    
    fig = px.bar(avg_scores, 
                 x='file', 
                 y='mean',
                 color='count',
                 title="Average Relevance Score by Document",
                 labels={'mean': 'Average Score', 
                        'file': 'Document', 
                        'count': 'Times Used'})
    st.plotly_chart(fig)

    fig = px.histogram(chunk_data, 
                      x='score',
                      nbins=20,
                      title="Distribution of Relevance Scores",
                      labels={'score': 'Relevance Score', 
                             'count': 'Frequency'})
    st.plotly_chart(fig)

def display_response_times(df):
    st.subheader("Response Time Analysis")
    if 'time' in df.columns:
        # Parse times robustly: coerce invalid strings to NaT, then fallback to mixed formats
        times = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
        if times.isna().all():
            # Fallback attempt without explicit format (lets pandas infer/mixed)
            times = pd.to_datetime(df['time'], errors='coerce')
        df['time'] = times.dt.time

        # Only count rows with valid date and time
        valid_mask = df['date'].notna() & df['time'].notna()
        if valid_mask.any():
            response_times = df.loc[valid_mask].groupby(df.loc[valid_mask, 'date'].dt.date)['time'].count().reset_index()
        else:
            response_times = pd.DataFrame({'date': [], 'time': []})
        
        fig = px.line(response_times,
                      x='date',
                      y='time',
                      title="Responses per Day",
                      labels={'date': 'Date', 
                             'time': 'Number of Responses'})
        st.plotly_chart(fig)

def display_user_interactions(df):
    st.subheader("User Interaction Patterns")
    user_stats = df.groupby('name').agg({
        'question': 'count',
        'date': 'nunique'
    }).reset_index()
    user_stats.columns = ['User', 'Total Questions', 'Active Days']
    
    fig = px.bar(user_stats,
                 x='User',
                 y=['Total Questions', 'Active Days'],
                 title="User Engagement Metrics",
                 barmode='group')
    st.plotly_chart(fig)

def display_response_metrics():
    st.subheader("Response Quality Metrics")
    
    # Load data from CSV
    logs_dir = "logs"
    df = load_response_data(logs_dir)
    
    if df.empty:
        st.info("No response data available yet. Try using the system first.")
        return
        
    # Log the data we're working with
    logger.info(f"Loaded dataframe with columns: {df.columns.tolist()}")
    logger.info(f"Sample cognitive_mode: {df['cognitive_mode'].iloc[0] if 'cognitive_mode' in df.columns else None}")
    logger.info(f"Sample creative_markers: {df['creative_markers'].iloc[0] if 'creative_markers' in df.columns else None}")
    logger.info(f"Sample temperature: {df['temperature'].iloc[0] if 'temperature' in df.columns else None}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mode distribution chart
        try:
            mode_counts = defaultdict(int)
            for mode_str in df['cognitive_mode'].dropna():
                try:
                    if isinstance(mode_str, str) and mode_str.strip():
                        mode_dict = eval(mode_str.strip())
                        if isinstance(mode_dict, dict):
                            for mode, count in mode_dict.items():
                                mode_counts[mode] += int(count)
                except:
                    continue
            
            if mode_counts:
                mode_data = pd.DataFrame(
                    list(mode_counts.items()),
                    columns=['Mode', 'Count']
                )
                fig = px.bar(mode_data, x='Mode', y='Count',
                            title="Cognitive Mode Distribution")
                st.plotly_chart(fig)
        except Exception as e:
            logger.error(f"Error processing mode distribution: {str(e)}")
    
    with col2:
        # Response length analysis
        try:
            if 'response_length' in df.columns:
                response_lengths = df['response_length'].dropna()
                if not response_lengths.empty:
                    fig = px.histogram(response_lengths,
                                     title="Response Length Distribution",
                                     labels={'value': 'Length (words)',
                                            'count': 'Frequency'})
                    st.plotly_chart(fig)
        except Exception as e:
            logger.error(f"Error processing response lengths: {str(e)}")
    
    # Creative markers analysis
    try:
        marker_counts = defaultdict(int)
        for markers_str in df['creative_markers'].dropna():
            try:
                if isinstance(markers_str, str) and markers_str.strip():
                    markers_dict = eval(markers_str.strip())
                    if isinstance(markers_dict, dict):
                        for marker, count in markers_dict.items():
                            marker_counts[marker] += int(count)
            except:
                continue
        
        if marker_counts:
            marker_data = pd.DataFrame(
                list(marker_counts.items()),
                columns=['Marker', 'Count']
            )
            fig = px.bar(marker_data, x='Marker', y='Count',
                        title="Creative Markers Frequency")
            st.plotly_chart(fig)
    except Exception as e:
        logger.error(f"Error processing creative markers: {str(e)}")
    
    # Temperature impact analysis
    try:
        temp_data = []
        for idx, row in df.iterrows():
            try:
                if isinstance(row.get('temperature'), str) and row['temperature'].strip():
                    temp_dict = eval(row['temperature'].strip())
                    if isinstance(temp_dict, dict):
                        for temp, length in temp_dict.items():
                            try:
                                value = float(length) if isinstance(length, (int, float)) else float(np.mean(length))
                                temp_data.append({
                                    'Temperature': float(temp),
                                    'Response Length': value,
                                    'Response Number': idx + 1
                                })
                            except:
                                continue
            except:
                continue
        
        if temp_data:
            temp_df = pd.DataFrame(temp_data)
            fig = px.line(temp_df, x='Response Number', y='Response Length',
                         title="Temperature Impact on Response Length")
            st.plotly_chart(fig)
    except Exception as e:
        logger.error(f"Error processing temperature impact: {str(e)}")

def create_performance_dashboard(df):
    st.header("System Performance Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
        )
    
    with col2:
        metrics_type = st.selectbox(
            "Metrics Type",
            ["All", "Chunk Scores", "Response Times", "User Interactions", "Model Usage"]
        )

    if metrics_type in ["All", "Chunk Scores"]:
        display_chunk_analysis(df)
    
    if metrics_type in ["All", "Response Times"]:
        display_response_times(df)
    
    if metrics_type in ["All", "User Interactions"]:
        display_user_interactions(df)
        
    if metrics_type in ["All", "Model Usage"]:
        display_model_usage(df)

def display_model_usage(df):
    st.subheader("Model Usage Analysis")
    
    if 'model_provider' not in df.columns or 'model_name' not in df.columns:
        st.warning("Model usage data not available")
        return
    
    # Count model usage
    model_counts = df.groupby(['model_provider', 'model_name']).size().reset_index(name='count')
    
    # Create visualization
    fig = px.bar(model_counts,
                 x='model_provider',
                 y='count',
                 color='model_name',
                 title="Model Usage Distribution",
                 labels={'model_provider': 'Provider',
                        'count': 'Number of Uses',
                        'model_name': 'Model'})
    st.plotly_chart(fig)

def create_document_usage_analysis(df):
    st.header("Document Usage Analysis")
    
    # Extract document names from chunk scores
    documents = []
    for col in ['chunk1_score', 'chunk2_score', 'chunk3_score']:
        if col in df.columns:
            for score in df[col].dropna():
                try:
                    doc_name = score.split('(')[0].strip()
                    documents.append(doc_name)
                except:
                    continue
    
    if not documents:
        st.warning("No document usage data available")
        return
    
    # Count document usage
    doc_counts = pd.Series(documents).value_counts().reset_index()
    doc_counts.columns = ['Document', 'Usage Count']
    
    # Create visualization
    fig = px.bar(doc_counts,
                 x='Document',
                 y='Usage Count',
                 title="Document Usage Frequency",
                 labels={'Document': 'Document Name',
                        'Usage Count': 'Times Used'})
    st.plotly_chart(fig)

def create_user_analysis(df):
    st.header("User Analysis")
    
    if 'name' not in df.columns:
        st.warning("User data not available")
        return
    
    # Calculate user statistics
    user_stats = df.groupby('name').agg({
        'question': 'count',
        'date': 'nunique'
    }).reset_index()
    user_stats.columns = ['User', 'Total Questions', 'Active Days']
    
    # Create visualization
    fig = px.bar(user_stats,
                 x='User',
                 y=['Total Questions', 'Active Days'],
                 title="User Engagement Metrics",
                 barmode='group',
                 labels={'value': 'Count',
                        'variable': 'Metric'})
    st.plotly_chart(fig)

def _extract_doc_name(score_str):
    try:
        return str(score_str).split('(')[0].strip()
    except Exception:
        return None

def _infer_topics_from_docs(doc_names):
    # Basic heuristic: use cleaned document base names as topics
    topics = {}
    for name in doc_names:
        if not name:
            continue
        topic = os.path.splitext(os.path.basename(name))[0]
        topics[name] = topic
    return topics

def _parse_topic_tags(value):
    # Accept comma-separated or JSON-like list strings
    try:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.startswith('[') and cleaned.endswith(']'):
                try:
                    return [str(v).strip().strip('"\'') for v in json.loads(cleaned)]
                except Exception:
                    pass
            # fallback: comma-separated
            return [t.strip() for t in cleaned.split(',') if t.strip()]
    except Exception:
        return []
    return []

def create_topics_map(df):
    st.header("Topics Map")
    # Collect document names from chunk score fields
    doc_names = []
    for col in ['chunk1_score', 'chunk2_score', 'chunk3_score']:
        if col in df.columns:
            doc_names.extend([_extract_doc_name(x) for x in df[col].dropna()])
    doc_names = [d for d in doc_names if d]
    if not doc_names and 'topic_tags' not in df.columns:
        st.warning("No topic or document usage data available")
        return

    # Map documents to topics (or use provided topic_tags if available)
    doc_to_topic = _infer_topics_from_docs(set(doc_names))

    # Build per-row topic usage
    usage_rows = []
    for _, row in df.iterrows():
        date_val = row.get('date')
        if pd.isna(date_val):
            continue
        topics = []
        # Prefer explicit topic tags if present
        if 'topic_tags' in df.columns and pd.notna(row.get('topic_tags')):
            topics = _parse_topic_tags(row.get('topic_tags'))
        else:
            for col in ['chunk1_score', 'chunk2_score', 'chunk3_score']:
                val = row.get(col)
                if pd.notna(val):
                    doc = _extract_doc_name(val)
                    topic = doc_to_topic.get(doc)
                    if topic:
                        topics.append(topic)
        for topic in set(topics):
            usage_rows.append({
                'date': pd.to_datetime(date_val, errors='coerce').date() if pd.notna(date_val) else None,
                'topic': topic
            })

    if not usage_rows:
        st.info("No topic usage could be derived from data")
        return

    usage_df = pd.DataFrame(usage_rows)
    usage_df = usage_df.dropna(subset=['date', 'topic'])

    # Aggregate by week for a clearer heatmap
    usage_df['week'] = pd.to_datetime(usage_df['date']).dt.to_period('W').apply(lambda r: r.start_time.date())
    heat = usage_df.groupby(['topic', 'week']).size().reset_index(name='count')

    if heat.empty:
        st.info("Insufficient data for topics heatmap")
        return

    pivot = heat.pivot(index='topic', columns='week', values='count').fillna(0)
    fig = px.imshow(pivot.values,
                    labels=dict(x='Week', y='Topic', color='Count'),
                    x=[str(c) for c in pivot.columns],
                    y=list(pivot.index),
                    aspect='auto',
                    title='Topic Usage Heatmap (by week)')
    st.plotly_chart(fig)

    # Document contribution (if chunk scores available)
    if doc_names:
        doc_counts = pd.Series(doc_names).value_counts().reset_index()
        doc_counts.columns = ['Document', 'Usage Count']
        fig2 = px.bar(doc_counts.head(20), x='Document', y='Usage Count',
                      title='Top Document Contributions',
                      labels={'Document': 'Document', 'Usage Count': 'Times Used'})
        st.plotly_chart(fig2)

def _simple_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    txt = text.lower()
    positive = ['clear', 'understand', 'confident', 'helpful', 'insight', 'progress', 'good', 'improve']
    negative = ['confused', 'unclear', 'stuck', 'difficult', 'frustrate', 'bad', 'worse', 'problem']
    score = sum(w in txt for w in positive) - sum(w in txt for w in negative)
    return score

def _extract_keywords(text, top_k=15):
    if not isinstance(text, str):
        return []
    stop = set(['the','and','for','with','that','this','from','into','over','about','your','their','into','when','how','what','why','are','was','were','will','would','could','should','have','has','had','on','in','of','to','a','an','as','it','its'])
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    words = [w for w in words if w not in stop]
    if not words:
        return []
    ser = pd.Series(words).value_counts().head(top_k)
    return list(ser.items())

def _extract_action_items(text):
    if not isinstance(text, str):
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    actions = []
    patterns = [r"^(i will|i'll|next|plan|action|todo|we will)\b",
                r"^[-*]\s+",
                r"\bby (monday|tomorrow|next week|date)\b"]
    for line in lines:
        if any(re.search(p, line, re.IGNORECASE) for p in patterns):
            actions.append(line)
    return actions[:10]

def create_reflections_analysis(df):
    st.header("Reflections Analysis")
    if 'reflection_text' not in df.columns and 'self_rating' not in df.columns:
        st.warning("No reflections data available")
        return

    reflections = df['reflection_text'] if 'reflection_text' in df.columns else pd.Series(dtype=str)
    ratings = df['self_rating'] if 'self_rating' in df.columns else pd.Series(dtype=float)

    # Sentiment distribution
    if not reflections.empty:
        sent_scores = reflections.fillna("").apply(_simple_sentiment)
        fig = px.histogram(sent_scores, nbins=11, title='Reflection Sentiment Distribution', labels={'value': 'Sentiment Score', 'count': 'Frequency'})
        st.plotly_chart(fig)

        # Top keywords
        all_keywords = []
        for txt in reflections.dropna().tolist():
            all_keywords.extend([k for k,_ in _extract_keywords(txt, top_k=5)])
        if all_keywords:
            kw_counts = pd.Series(all_keywords).value_counts().reset_index()
            kw_counts.columns = ['Keyword', 'Count']
            fig2 = px.bar(kw_counts.head(20), x='Keyword', y='Count', title='Top Reflection Keywords')
            st.plotly_chart(fig2)

        # Action items aggregation
        actions = []
        for txt in reflections.dropna().tolist():
            actions.extend(_extract_action_items(txt))
        if actions:
            st.subheader("Common Action Items")
            for a in actions[:20]:
                st.write(f"- {a}")

    # Ratings distribution
    if 'self_rating' in df.columns and df['self_rating'].notna().any():
        fig3 = px.histogram(df['self_rating'].dropna(), nbins=5, title='Self-Ratings Distribution', labels={'value': 'Rating', 'count': 'Frequency'})
        st.plotly_chart(fig3)

def create_interventions(df):
    st.header("Interventions (Suggested Teaching Plan)")
    # Estimate topic gaps: high usage with lower average relevance score
    topic_usage = {}
    topic_scores = {}
    for _, row in df.iterrows():
        topics_here = []
        for col in ['chunk1_score', 'chunk2_score', 'chunk3_score']:
            val = row.get(col)
            if pd.notna(val):
                doc = _extract_doc_name(val)
                topic = os.path.splitext(os.path.basename(doc))[0] if doc else None
                # extract numeric score
                try:
                    score_part = str(val).split('(')[1].split(')')[0]
                    score = float(score_part.split(':')[1].strip())
                except Exception:
                    score = None
                if topic:
                    topics_here.append((topic, score))
        for topic, score in topics_here:
            topic_usage[topic] = topic_usage.get(topic, 0) + 1
            if score is not None:
                topic_scores.setdefault(topic, []).append(score)

    if not topic_usage:
        st.info("Not enough data to generate interventions.")
        return

    topic_avg = {t: (np.mean(scores) if scores else 0) for t, scores in topic_scores.items()}
    # Rank by usage high and score low
    topics_rank = sorted(topic_usage.keys(), key=lambda t: ( -topic_usage.get(t,0), topic_avg.get(t,1)))

    suggestions = []
    for t in topics_rank[:10]:
        usage = topic_usage.get(t, 0)
        avg = topic_avg.get(t, np.nan)
        if np.isnan(avg):
            rationale = f"high interest (uses: {usage})"
        else:
            rationale = f"high interest (uses: {usage}) and lower grounding (avg score: {avg:.2f})"
        suggestions.append(
            f"Topic: {t}\n- Do: 10–15 min mini-lecture clarifying key concepts\n- Practice: Short applied exercise (3 prompts)\n- Resource: Link 2–3 key readings from documents mentioning {t}\n- Assessment: 3-question exit ticket\n- Rationale: {rationale}\n"
        )

    st.subheader("Suggested Priorities")
    for s in suggestions[:5]:
        st.write(s)

    # Download plan
    full_plan = "\n\n".join(suggestions)
    st.download_button(
        label="Download Teaching Plan (txt)",
        data=full_plan.encode('utf-8'),
        file_name="teaching_plan.txt",
        mime="text/plain"
    )

def main():
    st.title("GeddesGhost Admin Dashboard")
    
    if not check_password():
        st.stop()
    
    # Load data
    logs_dir = "logs"
    df = load_response_data(logs_dir)
    
    if df.empty:
        st.warning("No data available. Please use the system first.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Performance", "Document Usage", "User Analysis", "Response Metrics", "Topics Map", "Reflections", "Interventions"
    ])
    
    with tab1:
        create_performance_dashboard(df)
    
    with tab2:
        create_document_usage_analysis(df)
    
    with tab3:
        create_user_analysis(df)
    
    with tab4:
        display_response_metrics()

    with tab5:
        create_topics_map(df)

    with tab6:
        create_reflections_analysis(df)

    with tab7:
        create_interventions(df)

if __name__ == "__main__":
    main()