"""
Configuration file for the Local RAG Pipeline.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
INDEX_PATH = BASE_DIR / "faiss_index"

# Model configurations
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "orca-mini-3b-gguf2-q4_0.gguf"  # Smaller, more reliable model
FALLBACK_LLM_MODEL = "mistral-7b-openorca.Q4_0.gguf"

# Document processing settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MAX_CONTEXT_CHUNKS = 3

# Generation settings
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# Sample documents for download
SAMPLE_DOCUMENTS = {
    "artificial_intelligence.txt": "https://raw.githubusercontent.com/microsoft/AI-For-Beginners/main/lessons/1-Intro/README.md",
    "machine_learning.txt": "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/README.md",
    "data_science.txt": "https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/README.md"
}

# Evaluation settings
EVALUATION_METRICS = [
    'retrieval_metrics',
    'answer_relevance', 
    'context_relevance',
    'answer_quality',
    'hallucination'
]

# Sample queries for testing
SAMPLE_QUERIES = [
    "What is artificial intelligence and how is it used today?",
    "Explain machine learning and its main types.",
    "What are the key concepts in data science?"
]

def ensure_directories():
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(exist_ok=True)

def get_config():
    """Get configuration dictionary."""
    return {
        'base_dir': BASE_DIR,
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'results_dir': RESULTS_DIR,
        'index_path': INDEX_PATH,
        'embedding_model': DEFAULT_EMBEDDING_MODEL,
        'llm_model': DEFAULT_LLM_MODEL,
        'fallback_llm_model': FALLBACK_LLM_MODEL,
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'chunk_overlap': DEFAULT_CHUNK_OVERLAP,
        'max_context_chunks': MAX_CONTEXT_CHUNKS,
        'max_tokens': DEFAULT_MAX_TOKENS,
        'temperature': DEFAULT_TEMPERATURE,
        'top_p': DEFAULT_TOP_P,
        'sample_documents': SAMPLE_DOCUMENTS,
        'sample_queries': SAMPLE_QUERIES
    }
