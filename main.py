"""
Main script for the Local RAG Pipeline demonstration.
This script orchestrates the complete RAG pipeline workflow.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from document_processor import DocumentProcessor
from llm_interface import LocalLLM, RAGPipeline
from rag_evaluator import evaluate_rag_pipeline


def main():
    """Main function to run the complete RAG pipeline demonstration."""
    
    print("🚀 Starting Local RAG Pipeline Demonstration")
    print("=" * 60)
    
    # Configuration
    MODELS_DIR = "models"
    INDEX_PATH = "faiss_index"
    RESULTS_DIR = "results"
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Initialize Document Processor
    print("\n📚 Step 1: Initializing Document Processor")
    processor = DocumentProcessor()
    
    # Step 2: Download and process documents
    print("\n📥 Step 2: Downloading and Processing Documents")
    
    if os.path.exists(f"{INDEX_PATH}.index") and os.path.exists(f"{INDEX_PATH}_chunks.pkl"):
        print("Found existing index, loading...")
        processor.load_index(INDEX_PATH)
    else:
        print("No existing index found, creating new one...")
        
        # Download sample documents
        files = processor.download_sample_documents("data")
        
        if not files:
            print("❌ No documents downloaded. Cannot proceed.")
            return
        
        # Process documents into chunks
        chunks = processor.process_documents(files)
        print(f"✓ Processed {len(files)} documents into {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = processor.create_embeddings()
        print(f"✓ Created embeddings with shape: {embeddings.shape}")
        
        # Build FAISS index
        index = processor.build_faiss_index()
        print(f"✓ Built FAISS index with {index.ntotal} vectors")
        
        # Save index for future use
        processor.save_index(INDEX_PATH)
        print(f"✓ Saved index to {INDEX_PATH}")
    
    # Step 3: Initialize Local LLM
    print("\n🤖 Step 3: Initializing Local LLM")
    llm = LocalLLM(model_type="gpt4all")
    
    try:
        llm.load_model(MODELS_DIR)
        print("✓ Local LLM loaded successfully")
    except Exception as e:
        print(f"❌ Error loading LLM: {e}")
        print("Please ensure you have enough disk space and a stable internet connection.")
        return
    
    # Step 4: Create RAG Pipeline
    print("\n🔗 Step 4: Creating RAG Pipeline")
    rag = RAGPipeline(processor, llm)
    print("✓ RAG pipeline created")
    
    # Step 5: Define Sample Queries
    print("\n❓ Step 5: Processing Sample Queries")
    
    sample_queries = [
        "What is artificial intelligence and how is it used today?",
        "Explain machine learning and its main types.",
        "What are the advantages of Python programming language?"
    ]
    
    print(f"Processing {len(sample_queries)} sample queries...")
    
    # Process queries and collect results
    results = []
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = rag.query(
                question=query,
                k=3,  # Retrieve top 3 relevant chunks
                max_tokens=300,
                temperature=0.7
            )
            
            # Display results
            print(f"\n📄 Retrieved {result['num_retrieved']} relevant chunks:")
            for j, chunk in enumerate(result['retrieved_chunks'], 1):
                source_name = Path(chunk['source']).name
                score = chunk['similarity_score']
                text_preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                print(f"  {j}. [{source_name}] Score: {score:.4f}")
                print(f"     {text_preview}\n")
            
            print(f"🤖 Generated Response:")
            print(f"{result['response']}\n")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            continue
    
    # Step 6: Evaluate Results
    print("\n📊 Step 6: Evaluating RAG Pipeline Performance")
    print("=" * 60)
    
    if results:
        try:
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(RESULTS_DIR, f"rag_results_{timestamp}.json")
            
            # Convert results to JSON-serializable format
            json_results = []
            for result in results:
                json_result = {
                    'question': result['question'],
                    'response': result['response'],
                    'num_retrieved': result['num_retrieved'],
                    'retrieved_chunks': [
                        {
                            'text': chunk['text'],
                            'source': chunk['source'],
                            'similarity_score': float(chunk['similarity_score']),
                            'chunk_id': chunk.get('chunk_id', ''),
                        }
                        for chunk in result['retrieved_chunks']
                    ]
                }
                json_results.append(json_result)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results saved to: {results_file}")
            
            # Perform evaluation
            evaluation_file = os.path.join(RESULTS_DIR, f"evaluation_report_{timestamp}.txt")
            evaluation = evaluate_rag_pipeline(results, evaluation_file)
            
            # Save evaluation metrics as JSON
            eval_json_file = os.path.join(RESULTS_DIR, f"evaluation_metrics_{timestamp}.json")
            with open(eval_json_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, default=str)
            
            print(f"✓ Evaluation metrics saved to: {eval_json_file}")
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
    
    # Step 7: Summary
    print("\n🎉 Pipeline Demonstration Complete!")
    print("=" * 60)
    print("\nWhat was accomplished:")
    print("✓ Downloaded and processed public documents")
    print("✓ Created semantic embeddings using sentence-transformers")
    print("✓ Built FAISS vector index for efficient retrieval")
    print("✓ Loaded and used a local LLM (GPT4All)")
    print("✓ Implemented complete RAG pipeline")
    print("✓ Processed 3 sample queries with retrieval and generation")
    print("✓ Evaluated pipeline performance using multiple metrics")
    
    print(f"\n📁 Results and evaluations saved in: {RESULTS_DIR}/")
    print("\nNext steps:")
    print("• Review the evaluation report for insights")
    print("• Experiment with different queries")
    print("• Try different embedding models or LLMs")
    print("• Add more documents to improve knowledge coverage")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please check the error details above and try again.")
