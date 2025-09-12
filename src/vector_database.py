"""
Vector Database Setup using FAISS and Sentence Transformers
Creates embeddings for document chunks and sets up FAISS index for semantic search.
"""

import os
import json
import numpy as np
import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, 
                 model_name="all-MiniLM-L6-v2",
                 index_path="./models/faiss_index.bin",
                 embeddings_path="./models/embeddings.pkl",
                 documents_path="./documents/processed_chunks.json"):
        """
        Initialize Vector Database with FAISS and SentenceTransformers
        
        Args:
            model_name: Name of the sentence transformer model
            index_path: Path to save FAISS index
            embeddings_path: Path to save embeddings metadata
            documents_path: Path to processed document chunks
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.embeddings_path = Path(embeddings_path)
        self.documents_path = Path(documents_path)
        
        # Create directories
        self.index_path.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None
        
        logger.info(f"Initialized VectorDatabase with model: {model_name}")
    
    def load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return False
    
    def load_documents(self) -> bool:
        """Load processed document chunks"""
        try:
            if not self.documents_path.exists():
                logger.error(f"Documents file not found: {self.documents_path}")
                return False
            
            with open(self.documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            logger.info(f"Loaded {len(self.documents)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False
    
    def create_embeddings(self) -> bool:
        """Create embeddings for all document chunks"""
        try:
            if not self.model:
                logger.error("Embedding model not loaded")
                return False
            
            if not self.documents:
                logger.error("No documents loaded")
                return False
            
            logger.info("Creating embeddings for document chunks...")
            
            # Extract text content from documents
            texts = [doc['content'] for doc in self.documents]
            
            # Create embeddings with progress bar
            self.embeddings = self.model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            
            logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return False
    
    def create_faiss_index(self) -> bool:
        """Create FAISS index from embeddings"""
        try:
            if self.embeddings is None:
                logger.error("No embeddings available")
                return False
            
            logger.info("Creating FAISS index...")
            
            # Get embedding dimensions
            embedding_dim = self.embeddings.shape[1]
            
            # Create FAISS index (using L2 distance)
            self.index = faiss.IndexFlatL2(embedding_dim)
            
            # Add embeddings to index
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False
    
    def save_index(self) -> bool:
        """Save FAISS index and metadata"""
        try:
            if self.index is None:
                logger.error("No index to save")
                return False
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save embeddings and document metadata
            metadata = {
                'embeddings': self.embeddings,
                'documents': self.documents,
                'model_name': self.model_name,
                'embedding_dim': self.embeddings.shape[1],
                'num_documents': len(self.documents)
            }
            
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {self.index_path}")
            logger.info(f"Metadata saved to {self.embeddings_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load existing FAISS index and metadata"""
        try:
            if not self.index_path.exists():
                logger.warning(f"Index file not found: {self.index_path}")
                return False
            
            if not self.embeddings_path.exists():
                logger.warning(f"Metadata file not found: {self.embeddings_path}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.embeddings_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.embeddings = metadata['embeddings']
            self.documents = metadata['documents']
            self.model_name = metadata['model_name']
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            logger.info(f"Model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search using the query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with scores
        """
        try:
            if not self.model:
                logger.error("Embedding model not loaded")
                return []
            
            if not self.index:
                logger.error("FAISS index not loaded")
                return []
            
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search in FAISS index
            distances, indices = self.index.search(
                query_embedding.astype('float32'), 
                top_k
            )
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    result = {
                        'rank': i + 1,
                        'score': float(distance),
                        'similarity': 1.0 / (1.0 + float(distance)),  # Convert distance to similarity
                        'content': self.documents[idx]['content'],
                        'metadata': self.documents[idx]['metadata']
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def get_index_info(self) -> Dict:
        """Get information about the current index"""
        info = {
            'model_name': self.model_name,
            'index_loaded': self.index is not None,
            'num_vectors': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'num_documents': len(self.documents),
            'index_path': str(self.index_path),
            'embeddings_path': str(self.embeddings_path)
        }
        return info
    
    def build_complete_index(self) -> bool:
        """Build complete vector database from scratch"""
        logger.info("=== Building Complete Vector Database ===")
        
        steps = [
            ("Loading embedding model", self.load_embedding_model),
            ("Loading documents", self.load_documents),
            ("Creating embeddings", self.create_embeddings),
            ("Creating FAISS index", self.create_faiss_index),
            ("Saving index", self.save_index)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Failed at step: {step_name}")
                return False
        
        logger.info("Vector database built successfully!")
        return True

def main():
    """Test the vector database setup"""
    db = VectorDatabase()
    
    # Build the complete index
    success = db.build_complete_index()
    
    if success:
        # Test semantic search
        test_queries = [
            "What is attention mechanism?",
            "How does BERT work?",
            "What are residual connections?"
        ]
        
        logger.info("\n=== Testing Semantic Search ===")
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            results = db.semantic_search(query, top_k=3)
            
            for result in results:
                logger.info(f"  Rank {result['rank']}: Score={result['score']:.4f}, "
                          f"Source={result['metadata']['source']}, "
                          f"Preview={result['content'][:100]}...")
        
        # Show index info
        info = db.get_index_info()
        logger.info(f"\nIndex Info: {json.dumps(info, indent=2)}")
    else:
        logger.error("Failed to build vector database")

if __name__ == "__main__":
    main()