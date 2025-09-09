"""
Document processing and indexing module for the local RAG pipeline.
Handles document ingestion, chunking, and FAISS indexing.
"""

import os
import requests
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm


class DocumentProcessor:
    """Handles document ingestion, chunking, and vector indexing."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document processor.
        
        Args:
            embedding_model_name: Name of the sentence-transformers model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def download_sample_documents(self, data_dir: str = "data") -> List[str]:
        """
        Download sample public documents for testing.
        
        Args:
            data_dir: Directory to save documents
            
        Returns:
            List of downloaded file paths
        """
        os.makedirs(data_dir, exist_ok=True)
        
        # Sample documents to download
        documents = {
            "artificial_intelligence.txt": "https://raw.githubusercontent.com/microsoft/AI-For-Beginners/main/lessons/1-Intro/README.md",
            "machine_learning.txt": "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/README.md",
            "python_guide.txt": "https://raw.githubusercontent.com/python-guide/python-guide/master/docs/intro/overview.rst"
        }
        
        downloaded_files = []
        
        for filename, url in documents.items():
            file_path = os.path.join(data_dir, filename)
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                downloaded_files.append(file_path)
                print(f"✓ Downloaded {filename}")
                
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
                
        return downloaded_files
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
        return text
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents and extract chunks.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of dictionaries containing chunk information
        """
        all_chunks = []
        
        for file_path in tqdm(file_paths, desc="Processing documents"):
            file_ext = Path(file_path).suffix.lower()
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                text = self.extract_text_from_docx(file_path)
            else:
                text = self.extract_text_from_txt(file_path)
            
            if not text.strip():
                print(f"Warning: No text extracted from {file_path}")
                continue
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Create chunk metadata
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    'text': chunk,
                    'source': file_path,
                    'chunk_id': f"{Path(file_path).stem}_{i}",
                    'chunk_index': i
                }
                all_chunks.append(chunk_info)
        
        self.chunks = all_chunks
        print(f"Processed {len(file_paths)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all text chunks.
        
        Returns:
            Numpy array of embeddings
        """
        if not self.chunks:
            raise ValueError("No chunks available. Process documents first.")
        
        texts = [chunk['text'] for chunk in self.chunks]
        
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.embeddings = embeddings
        return embeddings
    
    def build_faiss_index(self) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Returns:
            FAISS index
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Create embeddings first.")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(self.embeddings)
        
        self.index = index
        print(f"Built FAISS index with {index.ntotal} vectors")
        return index
    
    def save_index(self, index_path: str = "faiss_index"):
        """
        Save FAISS index and chunks to disk.
        
        Args:
            index_path: Base path for saving index files
        """
        if self.index is None or not self.chunks:
            raise ValueError("No index or chunks to save. Build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{index_path}.index")
        
        # Save chunks metadata
        with open(f"{index_path}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved index and chunks to {index_path}.*")
    
    def load_index(self, index_path: str = "faiss_index"):
        """
        Load FAISS index and chunks from disk.
        
        Args:
            index_path: Base path for loading index files
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{index_path}.index")
        
        # Load chunks metadata
        with open(f"{index_path}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query.
        
        Args:
            query: Search query
            k: Number of top results to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.index is None:
            raise ValueError("No index available. Build or load index first.")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Return results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Download sample documents
    files = processor.download_sample_documents()
    
    if files:
        # Process documents
        chunks = processor.process_documents(files)
        
        # Create embeddings and build index
        processor.create_embeddings()
        processor.build_faiss_index()
        
        # Save index
        processor.save_index()
        
        # Test search
        results = processor.search("What is artificial intelligence?", k=3)
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['similarity_score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Text: {result['text'][:200]}...")
