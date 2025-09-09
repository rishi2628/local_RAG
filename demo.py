"""
Simple demonstration script for the Local RAG Pipeline.
A focused example that showcases core RAG functionality.
"""

import os
from pathlib import Path
from document_processor import DocumentProcessor
from llm_interface import LocalLLM, RAGPipeline
from config import get_config, ensure_directories


def simple_demo():
    """Run a simple RAG demonstration."""
    
    print("🚀 Simple Local RAG Demo")
    print("=" * 40)
    
    # Setup configuration and directories
    config = get_config()
    ensure_directories()
    
    # Step 1: Setup document processor
    print("\n1️⃣ Setting up document processor...")
    processor = DocumentProcessor(config['embedding_model'])
    
    # Check if we have an existing index
    index_path = str(config['index_path'])
    if os.path.exists(f"{index_path}.index"):
        print("📂 Loading existing index...")
        processor.load_index(index_path)
    else:
        print("📥 Creating new index from sample documents...")
        
        # Download and process documents
        files = processor.download_sample_documents(str(config['data_dir']))
        if files:
            processor.process_documents(files)
            processor.create_embeddings()
            processor.build_faiss_index()
            processor.save_index(index_path)
            print("✅ Index created and saved!")
        else:
            print("❌ Failed to download documents")
            return
    
    # Step 2: Test retrieval
    print("\n2️⃣ Testing document retrieval...")
    test_query = "What is machine learning?"
    results = processor.search(test_query, k=2)
    
    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} relevant chunks:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['similarity_score']:.4f}")
        print(f"   Source: {os.path.basename(result['source'])}")
        print(f"   Text: {result['text'][:150]}...")
    
    # Step 3: Setup LLM (optional, graceful failure)
    print("\n3️⃣ Setting up local LLM...")
    print("⚠️  Note: This may take a few minutes for first-time model download...")
    
    try:
        llm = LocalLLM(model_name=config['llm_model'])
        llm.load_model(str(config['models_dir']))
        print("✅ LLM loaded successfully!")
        
        # Step 4: Create and test RAG pipeline
        print("\n4️⃣ Testing RAG pipeline...")
        rag = RAGPipeline(processor, llm)
        
        result = rag.query(
            test_query, 
            k=config['max_context_chunks'], 
            max_tokens=200,
            temperature=config['temperature']
        )
        
        print(f"\n🤖 RAG Response:")
        print("-" * 40)
        print(result['response'])
        print("-" * 40)
        
        print(f"\n📊 Context used:")
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            source = Path(chunk['source']).name
            score = chunk['similarity_score']
            print(f"  {i}. [{source}] Relevance: {score:.4f}")
        
    except Exception as e:
        print(f"⚠️  LLM setup failed: {e}")
        print("\n💡 Tips for LLM setup:")
        print("  • Ensure you have at least 4GB free RAM")
        print("  • Check internet connection for model download")
        print("  • Try running again - first download can be slow")
        print("  • Retrieval functionality is still working!")
    
    print("\n✅ Demo completed!")
    print("\n🔍 What happened:")
    print("  ✓ Downloaded sample documents about AI/ML")
    print("  ✓ Created semantic embeddings using sentence-transformers")
    print("  ✓ Built FAISS index for fast similarity search")
    print("  ✓ Successfully retrieved relevant context for queries")
    if 'llm' in locals():
        print("  ✓ Generated contextual responses using local LLM")
    
    print(f"\n📁 Files created:")
    print(f"  • {config['data_dir']}/  - Downloaded documents")
    print(f"  • faiss_index.*  - Vector search index")
    if os.path.exists(str(config['models_dir'])):
        models = list(Path(config['models_dir']).glob("*.gguf"))
        if models:
            print(f"  • {config['models_dir']}/  - Downloaded LLM models")


def test_retrieval_only():
    """Test just the retrieval functionality without LLM."""
    print("🔍 Testing Retrieval-Only Mode")
    print("=" * 40)
    
    config = get_config()
    ensure_directories()
    
    processor = DocumentProcessor(config['embedding_model'])
    
    # Load or create index
    index_path = str(config['index_path'])
    if os.path.exists(f"{index_path}.index"):
        processor.load_index(index_path)
    else:
        files = processor.download_sample_documents(str(config['data_dir']))
        if files:
            processor.process_documents(files)
            processor.create_embeddings()
            processor.build_faiss_index()
            processor.save_index(index_path)
    
    # Test multiple queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "Explain data science concepts"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = processor.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            source = Path(result['source']).name
            score = result['similarity_score']
            text = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
            print(f"  {i}. [{source}] {score:.4f} - {text}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--retrieval-only":
        test_retrieval_only()
    else:
        simple_demo()
