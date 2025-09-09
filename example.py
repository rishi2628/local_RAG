"""
Comprehensive RAG Pipeline Example
This script demonstrates the complete workflow with evaluation.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from document_processor import DocumentProcessor
from llm_interface import LocalLLM, RAGPipeline
from rag_evaluator import RAGEvaluator
from config import get_config, ensure_directories


def run_rag_example():
    """Run a comprehensive RAG pipeline example."""
    
    print("🚀 Comprehensive Local RAG Pipeline Example")
    print("=" * 60)
    
    # Setup
    config = get_config()
    ensure_directories()
    
    # Sample queries for evaluation
    sample_queries = [
        "What is artificial intelligence and how is it defined?",
        "Explain the main types of machine learning algorithms.",
        "What are the applications of AI in today's world?"
    ]
    
    # Step 1: Document Processing
    print("\n📚 Step 1: Document Processing and Indexing")
    print("-" * 40)
    
    processor = DocumentProcessor(config['embedding_model'])
    index_path = str(config['index_path'])
    
    if os.path.exists(f"{index_path}.index"):
        print("Loading existing document index...")
        processor.load_index(index_path)
    else:
        print("Creating new document index...")
        files = processor.download_sample_documents(str(config['data_dir']))
        if files:
            chunks = processor.process_documents(files)
            embeddings = processor.create_embeddings()
            index = processor.build_faiss_index()
            processor.save_index(index_path)
            print(f"✅ Processed {len(files)} documents into {len(chunks)} chunks")
    
    # Step 2: Test Retrieval
    print(f"\n🔍 Step 2: Testing Semantic Retrieval")
    print("-" * 40)
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: {query}")
        results = processor.search(query, k=3)
        
        for j, result in enumerate(results, 1):
            source = Path(result['source']).name
            score = result['similarity_score']
            preview = result['text'][:80] + "..." if len(result['text']) > 80 else result['text']
            print(f"  {j}. [{source}] {score:.4f} - {preview}")
    
    # Step 3: RAG Pipeline (with graceful LLM handling)
    print(f"\n🤖 Step 3: RAG Pipeline with Local LLM")
    print("-" * 40)
    
    rag_results = []
    llm_available = False
    
    try:
        print("Attempting to load local LLM (this may take a while)...")
        llm = LocalLLM(model_name=config['llm_model'])
        llm.load_model(str(config['models_dir']))
        rag = RAGPipeline(processor, llm)
        llm_available = True
        print("✅ LLM loaded successfully!")
        
        # Process queries through RAG pipeline
        for i, query in enumerate(sample_queries, 1):
            print(f"\n--- RAG Query {i} ---")
            print(f"Question: {query}")
            
            result = rag.query(
                query, 
                k=config['max_context_chunks'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            
            print(f"\nRetrieved Context ({result['num_retrieved']} chunks):")
            for j, chunk in enumerate(result['retrieved_chunks'], 1):
                source = Path(chunk['source']).name
                score = chunk['similarity_score']
                print(f"  {j}. [{source}] Relevance: {score:.4f}")
            
            print(f"\nGenerated Response:")
            print(f"{result['response']}")
            
            rag_results.append(result)
            
    except Exception as e:
        print(f"⚠️ LLM not available: {e}")
        print("Continuing with retrieval-only demonstration...")
        
        # Create mock results for evaluation
        for query in sample_queries:
            chunks = processor.search(query, k=config['max_context_chunks'])
            mock_result = {
                'question': query,
                'response': f"[Mock response for: {query}] Based on the retrieved context, this would be a generated answer using the local LLM.",
                'retrieved_chunks': chunks,
                'num_retrieved': len(chunks)
            }
            rag_results.append(mock_result)
    
    # Step 4: Evaluation
    print(f"\n📊 Step 4: Pipeline Evaluation")
    print("-" * 40)
    
    if rag_results:
        evaluator = RAGEvaluator()
        evaluation = evaluator.comprehensive_evaluation(rag_results)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = config['results_dir'] / f"evaluation_report_{timestamp}.txt"
        report = evaluator.generate_evaluation_report(evaluation, str(report_file))
        
        print("\n" + report)
        
        # Save detailed results
        results_file = config['results_dir'] / f"rag_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert to JSON-serializable format
            json_results = []
            for result in rag_results:
                json_result = {
                    'question': result['question'],
                    'response': result['response'],
                    'num_retrieved': result['num_retrieved'],
                    'retrieved_chunks': [
                        {
                            'text': chunk['text'],
                            'source': chunk['source'],
                            'similarity_score': float(chunk['similarity_score'])
                        }
                        for chunk in result['retrieved_chunks']
                    ]
                }
                json_results.append(json_result)
            
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    # Step 5: Summary
    print(f"\n🎉 Pipeline Demonstration Complete!")
    print("=" * 60)
    
    print(f"\n✅ What was accomplished:")
    print(f"  • Document ingestion and chunking")
    print(f"  • Semantic embedding creation")
    print(f"  • FAISS vector index construction")
    print(f"  • Query-based document retrieval")
    if llm_available:
        print(f"  • Local LLM response generation")
    print(f"  • Comprehensive pipeline evaluation")
    
    print(f"\n📈 Key metrics from evaluation:")
    if rag_results and 'evaluation' in locals():
        retrieval_metrics = evaluation.get('retrieval_metrics', {})
        answer_relevance = evaluation.get('answer_relevance', {})
        
        print(f"  • Avg retrieval score: {retrieval_metrics.get('avg_retrieval_score', 0):.4f}")
        print(f"  • Avg answer relevance: {answer_relevance.get('mean', 0):.4f}")
        print(f"  • Documents per query: {retrieval_metrics.get('avg_num_retrieved', 0):.1f}")
    
    print(f"\n📁 Generated files:")
    print(f"  • {config['data_dir']}/ - Sample documents")
    print(f"  • faiss_index.* - Vector search index")
    print(f"  • {config['results_dir']}/ - Evaluation results")
    if os.path.exists(str(config['models_dir'])):
        models = list(Path(config['models_dir']).glob("*.gguf"))
        if models:
            print(f"  • {config['models_dir']}/ - LLM models")
    
    print(f"\n🔍 To explore further:")
    print(f"  • Modify queries in the script")
    print(f"  • Add your own documents to data/")
    print(f"  • Experiment with different embedding models")
    print(f"  • Try different LLM models")


def demonstrate_individual_components():
    """Demonstrate each component individually for educational purposes."""
    
    print("🎓 Individual Component Demonstration")
    print("=" * 50)
    
    config = get_config()
    ensure_directories()
    
    # 1. Document Processing
    print("\n1. Document Processing Component:")
    processor = DocumentProcessor()
    print(f"  • Embedding model: {config['embedding_model']}")
    print(f"  • Chunk size: {config['chunk_size']} characters")
    print(f"  • Chunk overlap: {config['chunk_overlap']} characters")
    
    # 2. Vector Search
    print("\n2. Vector Search Component:")
    if os.path.exists("faiss_index.index"):
        processor.load_index()
        print(f"  • Index loaded with {processor.index.ntotal} vectors")
        
        test_query = "neural networks"
        results = processor.search(test_query, k=2)
        print(f"  • Test search for '{test_query}': {len(results)} results")
        for result in results:
            print(f"    - Score: {result['similarity_score']:.4f}")
    
    # 3. LLM Interface
    print("\n3. Local LLM Component:")
    print(f"  • Model: {config['llm_model']}")
    print(f"  • Model directory: {config['models_dir']}")
    print(f"  • Generation settings: temp={config['temperature']}, max_tokens={config['max_tokens']}")
    
    # 4. Evaluation
    print("\n4. Evaluation Component:")
    evaluator = RAGEvaluator()
    print(f"  • Evaluation embedding model: {evaluator.embedding_model}")
    print(f"  • Available metrics: retrieval, relevance, quality, hallucination")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--components":
        demonstrate_individual_components()
    else:
        run_rag_example()
