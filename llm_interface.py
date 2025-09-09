"""
Local LLM interface module for the RAG pipeline.
Supports GPT4All and llama.cpp models for local inference.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from gpt4all import GPT4All


class LocalLLM:
    """Interface for local LLM models."""
    
    def __init__(self, model_type: str = "gpt4all", model_name: str = "mistral-7b-instruct-v0.1.Q4_0.gguf"):
        """
        Initialize the local LLM.
        
        Args:
            model_type: Type of model ('gpt4all' or 'llamacpp')
            model_name: Name/path of the model to use
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.model_path = None
        
    def download_model(self, models_dir: str = "models") -> str:
        """
        Download a model if it doesn't exist locally.
        
        Args:
            models_dir: Directory to store models
            
        Returns:
            Path to the downloaded model
        """
        os.makedirs(models_dir, exist_ok=True)
        
        if self.model_type == "gpt4all":
            # GPT4All will automatically download the model
            print(f"GPT4All will download {self.model_name} automatically if needed...")
            return os.path.join(models_dir, self.model_name)
        
        # For other model types, you would implement download logic here
        return ""
    
    def load_model(self, models_dir: str = "models"):
        """
        Load the local LLM model.
        
        Args:
            models_dir: Directory containing models
        """
        if self.model_type == "gpt4all":
            try:
                print(f"Loading GPT4All model: {self.model_name}")
                # GPT4All handles model downloading automatically
                self.model = GPT4All(
                    model_name=self.model_name,
                    model_path=models_dir,
                    allow_download=True
                )
                print("✓ Model loaded successfully")
                
            except Exception as e:
                print(f"✗ Error loading GPT4All model: {e}")
                # Fallback to a smaller, more reliable model
                print("Trying fallback model: orca-mini-3b-gguf2-q4_0.gguf")
                try:
                    self.model = GPT4All(
                        model_name="orca-mini-3b-gguf2-q4_0.gguf",
                        model_path=models_dir,
                        allow_download=True
                    )
                    self.model_name = "orca-mini-3b-gguf2-q4_0.gguf"
                    print("✓ Fallback model loaded successfully")
                except Exception as e2:
                    print(f"✗ Error loading fallback model: {e2}")
                    raise
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented yet")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response from the local LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            if self.model_type == "gpt4all":
                response = self.model.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    top_p=top_p,
                    streaming=False
                )
                return response.strip()
            else:
                raise NotImplementedError(f"Generation not implemented for {self.model_type}")
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating a response."


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""
    
    def __init__(self, document_processor, llm: LocalLLM):
        """
        Initialize the RAG pipeline.
        
        Args:
            document_processor: DocumentProcessor instance
            llm: LocalLLM instance
        """
        self.document_processor = document_processor
        self.llm = llm
    
    def create_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a RAG prompt with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([
            f"Document: {chunk['source']}\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""You are a helpful AI assistant. Use the following context information to answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context Information:
{context_text}

Question: {query}

Answer: Based on the provided context, """
        
        return prompt
    
    def query(
        self, 
        question: str, 
        k: int = 3, 
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            max_tokens: Maximum tokens for generation
            temperature: Generation temperature
            
        Returns:
            Dictionary containing query results
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.document_processor.search(question, k=k)
        
        # Create RAG prompt
        rag_prompt = self.create_rag_prompt(question, retrieved_chunks)
        
        # Generate response
        response = self.llm.generate_response(
            prompt=rag_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            'question': question,
            'retrieved_chunks': retrieved_chunks,
            'rag_prompt': rag_prompt,
            'response': response,
            'num_retrieved': len(retrieved_chunks)
        }
    
    def batch_query(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple queries through the RAG pipeline.
        
        Args:
            questions: List of questions
            **kwargs: Additional arguments for query method
            
        Returns:
            List of query results
        """
        results = []
        for question in questions:
            print(f"\nProcessing: {question}")
            result = self.query(question, **kwargs)
            results.append(result)
        return results


if __name__ == "__main__":
    # Example usage
    from document_processor import DocumentProcessor
    
    # Initialize components
    processor = DocumentProcessor()
    llm = LocalLLM()
    
    # Load models and index
    llm.load_model()
    
    # Check if index exists, otherwise create it
    if os.path.exists("faiss_index.index"):
        processor.load_index()
    else:
        print("No existing index found. Creating new index...")
        files = processor.download_sample_documents()
        if files:
            processor.process_documents(files)
            processor.create_embeddings()
            processor.build_faiss_index()
            processor.save_index()
    
    # Create RAG pipeline
    rag = RAGPipeline(processor, llm)
    
    # Test queries
    test_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the benefits of using Python for programming?"
    ]
    
    # Process queries
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        result = rag.query(question)
        
        print(f"\nRetrieved {result['num_retrieved']} relevant chunks:")
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"\n{i}. Score: {chunk['similarity_score']:.4f}")
            print(f"Source: {Path(chunk['source']).name}")
            print(f"Text: {chunk['text'][:150]}...")
        
        print(f"\nGenerated Response:")
        print(result['response'])
