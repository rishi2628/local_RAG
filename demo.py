"""
Local RAG Demo Script
Quick demonstration of the complete RAG pipeline.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline

def main():
    """Demo the RAG pipeline with interactive queries"""
    print("🤖 Local RAG Pipeline Demo")
    print("=" * 50)
    print("Initializing components...")
    
    # Initialize pipeline
    rag = RAGPipeline()
    
    if not rag.initialize():
        print("❌ Failed to initialize RAG pipeline")
        return
    
    print("✅ RAG pipeline ready!")
    print("\nYou can ask questions about:")
    print("- Attention mechanisms and Transformers")
    print("- BERT and language models")
    print("- ResNet and residual connections")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            query = input("🔍 Ask a question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query
            print("\n⏳ Processing...")
            start_time = time.time()
            
            response = rag.query(query)
            
            # Display results
            print(f"\n📝 **Answer** ({response.total_time:.1f}s):")
            print("-" * 40)
            print(response.answer)
            
            print(f"\n📚 **Sources** ({len(response.retrieved_contexts)} documents):")
            print("-" * 40)
            for i, doc in enumerate(response.retrieved_contexts[:3]):
                source = doc['metadata']['source']
                score = doc['score']
                print(f"{i+1}. {source} (relevance: {1/(1+score):.3f})")
            
            print(f"\n⚡ **Performance:**")
            print(f"   Retrieval: {response.retrieval_time:.3f}s")
            print(f"   Generation: {response.generation_time:.3f}s")
            print(f"   Total: {response.total_time:.3f}s")
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()