"""
RAG Pipeline Implementation
Combines document retrieval, context augmentation, and local LLM generation.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Import our custom components
from local_llm import LocalLLM
from vector_database import VectorDatabase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structure for RAG pipeline response"""
    query: str
    answer: str
    retrieved_contexts: List[Dict]
    generation_time: float
    retrieval_time: float
    total_time: float
    metadata: Dict

class RAGPipeline:
    def __init__(self, 
                 llm_model_name="mistral-7b-instruct-v0.1.Q4_0.gguf",
                 embedding_model_name="all-MiniLM-L6-v2",
                 top_k_retrieval=5,
                 max_context_length=2000):
        """
        Initialize RAG Pipeline
        
        Args:
            llm_model_name: Name of the local LLM model
            embedding_model_name: Name of the embedding model
            top_k_retrieval: Number of documents to retrieve
            max_context_length: Maximum context length for LLM
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.top_k_retrieval = top_k_retrieval
        self.max_context_length = max_context_length
        
        # Initialize components
        self.llm = None
        self.vector_db = None
        self.is_initialized = False
        
        logger.info("RAG Pipeline initialized")
    
    def initialize(self) -> bool:
        """Initialize all components of the RAG pipeline"""
        try:
            logger.info("=== Initializing RAG Pipeline ===")
            
            # Initialize Local LLM
            logger.info("Initializing Local LLM...")
            self.llm = LocalLLM(model_name=self.llm_model_name)
            if not self.llm.download_model():
                logger.error("Failed to initialize LLM")
                return False
            
            # Initialize Vector Database
            logger.info("Initializing Vector Database...")
            self.vector_db = VectorDatabase(model_name=self.embedding_model_name)
            
            # Load embedding model
            if not self.vector_db.load_embedding_model():
                logger.error("Failed to load embedding model")
                return False
            
            # Load existing index or build new one
            if not self.vector_db.load_index():
                logger.info("No existing index found. Building new index...")
                if not self.vector_db.build_complete_index():
                    logger.error("Failed to build vector database")
                    return False
            
            self.is_initialized = True
            logger.info("RAG Pipeline initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            return False
    
    def retrieve_context(self, query: str) -> Tuple[List[Dict], float]:
        """
        Retrieve relevant context for the query
        
        Args:
            query: User query
            
        Returns:
            Tuple of (retrieved documents, retrieval time)
        """
        import time
        start_time = time.time()
        
        try:
            if not self.vector_db:
                logger.error("Vector database not initialized")
                return [], 0.0
            
            # Perform semantic search
            results = self.vector_db.semantic_search(query, top_k=self.top_k_retrieval)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
            
            return results, retrieval_time
            
        except Exception as e:
            logger.error(f"Error in context retrieval: {e}")
            return [], 0.0
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            content = doc['content'].strip()
            source = doc['metadata']['source']
            chunk_id = doc['metadata']['chunk_id']
            
            # Format document with metadata
            doc_text = f"[Document {i+1} - {source} (chunk {chunk_id})]:\n{content}\n"
            
            # Check if adding this document would exceed max length
            if current_length + len(doc_text) > self.max_context_length:
                if not context_parts:  # Include at least one document
                    context_parts.append(doc_text[:self.max_context_length])
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> Tuple[str, float]:
        """
        Generate response using local LLM with context
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Tuple of (generated response, generation time)
        """
        import time
        start_time = time.time()
        
        try:
            if not self.llm:
                logger.error("LLM not initialized")
                return "Error: LLM not available", 0.0
            
            # Create prompt with context and query
            prompt = self.create_rag_prompt(query, context)
            
            # Generate response
            response = self.llm.generate_response(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7
            )
            
            generation_time = time.time() - start_time
            
            if response:
                logger.info(f"Generated response in {generation_time:.3f}s")
                return response.strip(), generation_time
            else:
                logger.warning("No response generated")
                return "I apologize, but I couldn't generate a response to your query.", generation_time
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return f"Error generating response: {str(e)}", 0.0
    
    def create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create RAG prompt template
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt_template = """Based on the following context documents, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer: """
        
        return prompt_template.format(context=context, query=query)
    
    def query(self, user_query: str) -> RAGResponse:
        """
        Main RAG pipeline query method
        
        Args:
            user_query: User's question
            
        Returns:
            RAGResponse object with complete results
        """
        import time
        total_start_time = time.time()
        
        if not self.is_initialized:
            logger.error("RAG Pipeline not initialized. Call initialize() first.")
            return RAGResponse(
                query=user_query,
                answer="Error: RAG Pipeline not initialized",
                retrieved_contexts=[],
                generation_time=0.0,
                retrieval_time=0.0,
                total_time=0.0,
                metadata={"error": "Not initialized"}
            )
        
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: Retrieve relevant context
        retrieved_docs, retrieval_time = self.retrieve_context(user_query)
        
        # Step 2: Format context
        formatted_context = self.format_context(retrieved_docs)
        
        # Step 3: Generate response
        answer, generation_time = self.generate_response(user_query, formatted_context)
        
        total_time = time.time() - total_start_time
        
        # Create response object
        response = RAGResponse(
            query=user_query,
            answer=answer,
            retrieved_contexts=retrieved_docs,
            generation_time=generation_time,
            retrieval_time=retrieval_time,
            total_time=total_time,
            metadata={
                "num_retrieved_docs": len(retrieved_docs),
                "context_length": len(formatted_context),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Query processed in {total_time:.3f}s total")
        return response
    
    def batch_query(self, queries: List[str]) -> List[RAGResponse]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of user queries
            
        Returns:
            List of RAGResponse objects
        """
        logger.info(f"Processing {len(queries)} queries in batch")
        responses = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            response = self.query(query)
            responses.append(response)
        
        return responses
    
    def save_responses(self, responses: List[RAGResponse], output_path: str):
        """
        Save RAG responses to JSON file
        
        Args:
            responses: List of RAG responses
            output_path: Path to save the responses
        """
        try:
            # Convert responses to serializable format
            serializable_responses = []
            for response in responses:
                serializable_response = {
                    "query": response.query,
                    "answer": response.answer,
                    "retrieved_contexts": response.retrieved_contexts,
                    "generation_time": response.generation_time,
                    "retrieval_time": response.retrieval_time,
                    "total_time": response.total_time,
                    "metadata": response.metadata
                }
                serializable_responses.append(serializable_response)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_responses, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(responses)} responses to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving responses: {e}")
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the RAG pipeline configuration"""
        return {
            "llm_model": self.llm_model_name,
            "embedding_model": self.embedding_model_name,
            "top_k_retrieval": self.top_k_retrieval,
            "max_context_length": self.max_context_length,
            "is_initialized": self.is_initialized,
            "vector_db_info": self.vector_db.get_index_info() if self.vector_db else None
        }

def main():
    """Test the RAG pipeline"""
    # Sample queries for testing
    test_queries = [
        "What is the attention mechanism and how does it work?",
        "How does BERT differ from traditional language models?",
        "What are residual connections and why are they important in deep networks?"
    ]
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Initialize components
    if rag.initialize():
        logger.info("=== Testing RAG Pipeline ===")
        
        # Process queries
        responses = rag.batch_query(test_queries)
        
        # Display results
        for i, response in enumerate(responses):
            print(f"\n{'='*80}")
            print(f"QUERY {i+1}: {response.query}")
            print(f"{'='*80}")
            print(f"ANSWER: {response.answer}")
            print(f"\nRETRIEVAL TIME: {response.retrieval_time:.3f}s")
            print(f"GENERATION TIME: {response.generation_time:.3f}s")
            print(f"TOTAL TIME: {response.total_time:.3f}s")
            print(f"RETRIEVED DOCS: {response.metadata['num_retrieved_docs']}")
            
            print("\nTOP RETRIEVED CONTEXTS:")
            for j, doc in enumerate(response.retrieved_contexts[:3]):
                print(f"  {j+1}. {doc['metadata']['source']} (Score: {doc['score']:.4f})")
                print(f"     Preview: {doc['content'][:100]}...")
        
        # Save responses
        output_path = "evaluation/rag_responses.json"
        Path("evaluation").mkdir(exist_ok=True)
        rag.save_responses(responses, output_path)
        
        # Show pipeline info
        info = rag.get_pipeline_info()
        print(f"\n{'='*80}")
        print("PIPELINE INFORMATION:")
        print(f"{'='*80}")
        print(json.dumps(info, indent=2))
        
    else:
        logger.error("Failed to initialize RAG pipeline")

if __name__ == "__main__":
    main()