"""
Document Ingestion Pipeline
Downloads public documents and processes them into chunks for RAG.
"""

import os
import requests
import logging
from pathlib import Path
from typing import List, Dict
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngestion:
    def __init__(self, documents_path="./documents", chunk_size=1000, chunk_overlap=200):
        """
        Initialize document ingestion pipeline
        
        Args:
            documents_path: Path to store documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.documents_path = Path(documents_path)
        self.documents_path.mkdir(exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Sample public documents to download
        self.sample_documents = [
            {
                "name": "attention_is_all_you_need",
                "url": "https://arxiv.org/pdf/1706.03762.pdf",
                "type": "pdf",
                "description": "Attention Is All You Need - Transformer Paper"
            },
            {
                "name": "bert_paper", 
                "url": "https://arxiv.org/pdf/1810.04805.pdf",
                "type": "pdf",
                "description": "BERT: Pre-training of Deep Bidirectional Transformers"
            },
            {
                "name": "resnet_paper",
                "url": "https://arxiv.org/pdf/1512.03385.pdf",
                "type": "pdf", 
                "description": "Deep Residual Learning for Image Recognition"
            }
        ]
    
    def download_document(self, doc_info: Dict) -> bool:
        """
        Download a document from URL
        
        Args:
            doc_info: Dictionary containing document information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading {doc_info['name']} from {doc_info['url']}")
            
            response = requests.get(doc_info['url'], stream=True, timeout=30)
            response.raise_for_status()
            
            file_path = self.documents_path / f"{doc_info['name']}.{doc_info['type']}"
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Successfully downloaded {doc_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {doc_info['name']}: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_document(self, doc_info: Dict) -> List[Dict]:
        """
        Process a document into chunks
        
        Args:
            doc_info: Document information
            
        Returns:
            List of document chunks with metadata
        """
        file_path = self.documents_path / f"{doc_info['name']}.{doc_info['type']}"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            # Extract text based on file type
            if doc_info['type'] == 'pdf':
                text = self.extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return []
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk documents with metadata
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "content": chunk,
                    "metadata": {
                        "source": doc_info['name'],
                        "description": doc_info['description'],
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "file_type": doc_info['type']
                    }
                }
                chunk_docs.append(chunk_doc)
            
            logger.info(f"Processed {doc_info['name']} into {len(chunks)} chunks")
            return chunk_docs
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def download_all_documents(self) -> bool:
        """Download all sample documents"""
        logger.info("Starting document download...")
        success_count = 0
        
        for doc_info in self.sample_documents:
            if self.download_document(doc_info):
                success_count += 1
        
        logger.info(f"Downloaded {success_count}/{len(self.sample_documents)} documents")
        return success_count == len(self.sample_documents)
    
    def process_all_documents(self) -> List[Dict]:
        """
        Process all documents into chunks
        
        Returns:
            List of all document chunks
        """
        logger.info("Processing all documents...")
        all_chunks = []
        
        for doc_info in self.sample_documents:
            chunks = self.process_document(doc_info)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Save chunks to JSON file for later use
        chunks_file = self.documents_path / "processed_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved chunks to {chunks_file}")
        return all_chunks
    
    def get_document_stats(self) -> Dict:
        """Get statistics about processed documents"""
        chunks_file = self.documents_path / "processed_chunks.json"
        
        if not chunks_file.exists():
            return {"error": "No processed chunks found"}
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Calculate statistics
        stats = {
            "total_chunks": len(chunks),
            "documents": {},
            "total_characters": 0
        }
        
        for chunk in chunks:
            source = chunk['metadata']['source']
            if source not in stats["documents"]:
                stats["documents"][source] = {
                    "chunks": 0,
                    "characters": 0,
                    "description": chunk['metadata']['description']
                }
            
            stats["documents"][source]["chunks"] += 1
            chunk_chars = len(chunk['content'])
            stats["documents"][source]["characters"] += chunk_chars
            stats["total_characters"] += chunk_chars
        
        return stats

def main():
    """Test the document ingestion pipeline"""
    ingestion = DocumentIngestion()
    
    # Download documents
    logger.info("=== Starting Document Ingestion Pipeline ===")
    success = ingestion.download_all_documents()
    
    if success:
        # Process documents
        chunks = ingestion.process_all_documents()
        
        # Show statistics
        stats = ingestion.get_document_stats()
        logger.info(f"Processing complete. Statistics: {json.dumps(stats, indent=2)}")
    else:
        logger.error("Failed to download all documents")

if __name__ == "__main__":
    main()