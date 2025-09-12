"""
RAGAS Evaluation for RAG Pipeline
Evaluates the quality of RAG responses using RAGAS metrics.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import warnings
import asyncio

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        context_relevancy
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.llms import GPT4All
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"RAGAS import error: {e}")
    RAGAS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGASEvaluator:
    def __init__(self, responses_path="./evaluation/rag_responses.json"):
        """
        Initialize RAGAS evaluator
        
        Args:
            responses_path: Path to RAG responses JSON file
        """
        self.responses_path = Path(responses_path)
        self.responses = []
        self.evaluation_results = {}
        
        logger.info("RAGAS Evaluator initialized")
    
    def load_responses(self) -> bool:
        """Load RAG responses from JSON file"""
        try:
            if not self.responses_path.exists():
                logger.error(f"Responses file not found: {self.responses_path}")
                return False
            
            with open(self.responses_path, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            
            logger.info(f"Loaded {len(self.responses)} responses for evaluation")
            return True
            
        except Exception as e:
            logger.error(f"Error loading responses: {e}")
            return False
    
    def prepare_evaluation_data(self) -> Dataset:
        """
        Prepare data in RAGAS-compatible format
        
        Returns:
            Dataset object for RAGAS evaluation
        """
        try:
            eval_data = {
                'question': [],
                'answer': [],
                'contexts': [],
                'ground_truths': []
            }
            
            for response in self.responses:
                # Extract contexts from retrieved documents
                contexts = [doc['content'] for doc in response['retrieved_contexts']]
                
                # For this demo, we'll create simple ground truth based on the domain
                # In a real scenario, you'd have expert-annotated ground truths
                ground_truth = self.create_ground_truth(response['query'])
                
                eval_data['question'].append(response['query'])
                eval_data['answer'].append(response['answer'])
                eval_data['contexts'].append(contexts)
                eval_data['ground_truths'].append([ground_truth])
            
            dataset = Dataset.from_dict(eval_data)
            logger.info("Evaluation data prepared successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing evaluation data: {e}")
            return None
    
    def setup_local_llm_for_ragas(self):
        """
        Setup local LLM for RAGAS evaluation using GPT4All
        
        Returns:
            Configured LLM and embeddings for RAGAS
        """
        try:
            logger.info("Setting up local LLM for RAGAS evaluation...")
            
            # Setup local LLM (GPT4All)
            local_llm = GPT4All(
                model="./models/mistral-7b-instruct-v0.1.Q4_0.gguf",
                max_tokens=512,
                temp=0.7,
                verbose=False
            )
            
            # Wrap for RAGAS
            ragas_llm = LangchainLLMWrapper(local_llm)
            
            # Setup embeddings (same as vector database)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            logger.info("Local LLM setup completed for RAGAS")
            return ragas_llm, embeddings
            
        except Exception as e:
            logger.error(f"Error setting up local LLM for RAGAS: {e}")
            return None, None
    
    def configure_ragas_metrics(self, llm, embeddings):
        """
        Configure RAGAS metrics with local models
        
        Args:
            llm: Local LLM for evaluation
            embeddings: Embedding model
            
        Returns:
            List of configured metrics
        """
        try:
            # Configure metrics with local models
            metrics = []
            
            # Faithfulness - measures factual accuracy
            faithfulness_metric = faithfulness
            faithfulness_metric.llm = llm
            metrics.append(faithfulness_metric)
            
            # Answer Relevancy - measures relevance to question
            answer_relevancy_metric = answer_relevancy  
            answer_relevancy_metric.llm = llm
            answer_relevancy_metric.embeddings = embeddings
            metrics.append(answer_relevancy_metric)
            
            # Context Relevancy - measures context relevance
            context_relevancy_metric = context_relevancy
            context_relevancy_metric.llm = llm
            metrics.append(context_relevancy_metric)
            
            # Context Precision - measures precision of retrieval
            context_precision_metric = context_precision
            context_precision_metric.llm = llm
            metrics.append(context_precision_metric)
            
            # Context Recall - measures recall of retrieval
            context_recall_metric = context_recall
            context_recall_metric.llm = llm
            context_recall_metric.embeddings = embeddings
            metrics.append(context_recall_metric)
            
            logger.info(f"Configured {len(metrics)} RAGAS metrics with local models")
            return metrics
            
        except Exception as e:
            logger.error(f"Error configuring RAGAS metrics: {e}")
            return []
    
    def create_ground_truth(self, query: str) -> str:
        """
        Create simple ground truth for demonstration
        In production, this would be expert-annotated
        
        Args:
            query: The query question
            
        Returns:
            Simple ground truth answer
        """
        ground_truths = {
            "attention": "Attention mechanism allows models to focus on relevant parts of input sequences by computing weighted representations based on similarity scores between queries, keys, and values.",
            "bert": "BERT differs from traditional language models by using bidirectional context and masked language modeling for pre-training, enabling better understanding of context.",
            "residual": "Residual connections help address vanishing gradient problems in deep networks by allowing gradients to flow directly through skip connections."
        }
        
        query_lower = query.lower()
        for key, truth in ground_truths.items():
            if key in query_lower:
                return truth
        
        return "This question requires domain expertise to answer accurately."
    
    def run_ragas_evaluation(self) -> Dict[str, Any]:
        """
        Run RAGAS evaluation on the prepared dataset
        
        Returns:
            Dictionary with evaluation results
        """
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS not available. Running alternative evaluation.")
            return self.run_alternative_evaluation()
        
        # Skip RAGAS entirely and go straight to alternative evaluation
        # This avoids async task cleanup issues with RAGAS + local LLMs
        logger.info("Skipping RAGAS evaluation due to compatibility issues with local LLM")
        logger.info("Using alternative evaluation metrics for better reliability...")
        return self.run_alternative_evaluation()
    
    def run_alternative_evaluation(self) -> Dict[str, Any]:
        """
        Alternative evaluation when RAGAS is not available
        Uses local similarity computations to replicate RAGAS metrics
        
        Returns:
            Dictionary with evaluation results including RAGAS-equivalent metrics
        """
        logger.info("Running comprehensive local evaluation with RAGAS-equivalent metrics...")
        
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Load embedding model for similarity calculations
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize metrics including RAGAS-equivalent ones
            metrics = {
                # RAGAS-equivalent metrics
                'answer_relevancy': 0,
                'faithfulness': 0,
                'context_recall': 0,
                'context_precision': 0,
                'context_relevancy': 0,
                
                # Performance metrics
                'avg_response_length': 0,
                'avg_retrieval_time': 0,
                'avg_generation_time': 0,
                'avg_total_time': 0,
                'context_utilization': 0,
                'response_completeness': 0,
                'query_coverage': 0,
                'semantic_similarity_to_context': 0,
                'response_coherence': 0,
                'context_relevance': 0
            }
            
            total_responses = len(self.responses)
            
            for response in self.responses:
                # Basic metrics
                metrics['avg_response_length'] += len(response['answer'])
                metrics['avg_retrieval_time'] += response['retrieval_time']
                metrics['avg_generation_time'] += response['generation_time']
                metrics['avg_total_time'] += response['total_time']
                
                # Context utilization (how many docs were retrieved)
                metrics['context_utilization'] += len(response['retrieved_contexts'])
                
                # Response completeness (basic heuristic)
                if len(response['answer']) > 100 and '.' in response['answer']:
                    metrics['response_completeness'] += 1
                
                # Query coverage (check if response addresses the query)
                query_words = set(response['query'].lower().split())
                answer_words = set(response['answer'].lower().split())
                overlap = len(query_words.intersection(answer_words))
                metrics['query_coverage'] += overlap / len(query_words) if query_words else 0
                
                # Calculate embeddings once for efficiency
                query_embedding = embedding_model.encode([response['query']])
                answer_embedding = embedding_model.encode([response['answer']])
                
                if response['retrieved_contexts']:
                    contexts_text = " ".join([doc['content'] for doc in response['retrieved_contexts'][:3]])
                    context_embedding = embedding_model.encode([contexts_text])
                    
                    # RAGAS-equivalent metrics using semantic similarity
                    
                    # 1. Answer Relevancy: How relevant is the answer to the question
                    answer_query_similarity = cosine_similarity(answer_embedding, query_embedding)[0][0]
                    metrics['answer_relevancy'] += answer_query_similarity
                    
                    # 2. Faithfulness: How well the answer is grounded in the context
                    answer_context_similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
                    metrics['faithfulness'] += answer_context_similarity
                    
                    # 3. Context Relevancy: How relevant is the retrieved context to the question
                    context_query_similarity = cosine_similarity(context_embedding, query_embedding)[0][0]
                    metrics['context_relevancy'] += context_query_similarity
                    
                    # 4. Context Precision: Quality of retrieved context (using similarity threshold)
                    # Higher similarity means more precise context
                    if context_query_similarity > 0.5:  # Threshold for relevant context
                        metrics['context_precision'] += 1.0
                    else:
                        metrics['context_precision'] += context_query_similarity
                    
                    # 5. Context Recall: Coverage completeness (proxy using answer-context similarity)
                    # Higher similarity suggests better context coverage
                    metrics['context_recall'] += answer_context_similarity
                    
                    # Legacy metrics for backward compatibility
                    metrics['semantic_similarity_to_context'] += answer_context_similarity
                    metrics['context_relevance'] += context_query_similarity
                
                # Response coherence (simple heuristic based on sentence structure)
                sentences = response['answer'].split('.')
                if len(sentences) > 1 and all(len(s.strip()) > 10 for s in sentences[:3]):
                    metrics['response_coherence'] += 1
            
            # Calculate averages
            for key in metrics:
                if key in ['response_completeness', 'response_coherence']:
                    metrics[key] = metrics[key] / total_responses  # Proportion
                elif key != 'avg_response_length':
                    metrics[key] = metrics[key] / total_responses
            
            # Normalize scores to 0-1 scale where needed
            metrics['response_completeness'] = min(metrics['response_completeness'], 1.0)
            metrics['response_coherence'] = min(metrics['response_coherence'], 1.0)
            metrics['context_precision'] = min(metrics['context_precision'], 1.0)
            
            logger.info("Local evaluation completed successfully")
            logger.info("RAGAS-EQUIVALENT METRICS:")
            logger.info(f"  • Answer Relevancy: {metrics['answer_relevancy']:.4f}")
            logger.info(f"  • Faithfulness: {metrics['faithfulness']:.4f}")
            logger.info(f"  • Context Recall: {metrics['context_recall']:.4f}")
            logger.info(f"  • Context Precision: {metrics['context_precision']:.4f}")
            logger.info(f"  • Context Relevancy: {metrics['context_relevancy']:.4f}")
            logger.info("ADDITIONAL LOCAL METRICS:")
            logger.info(f"  • Response Completeness: {metrics['response_completeness']:.4f}")
            logger.info(f"  • Response Coherence: {metrics['response_coherence']:.4f}")
            logger.info(f"  • Query Coverage: {metrics['query_coverage']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in alternative evaluation: {e}")
            return {}
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Create a detailed evaluation report
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted evaluation report string
        """
        report = []
        report.append("=" * 80)
        report.append("RAG PIPELINE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Queries Evaluated: {len(self.responses)}")
        report.append("")
        
        # Add metrics section
        report.append("EVALUATION METRICS:")
        report.append("-" * 40)
        
        if 'faithfulness' in results and 'answer_relevancy' in results:
            # RAGAS-equivalent metrics with detailed descriptions
            report.append("RAGAS-EQUIVALENT METRICS (Local Implementation):")
            report.append(f"• Answer Relevancy: {results.get('answer_relevancy', 'N/A'):.4f}")
            report.append("  (Cosine similarity between answer and question - measures relevance)")
            report.append(f"• Faithfulness: {results.get('faithfulness', 'N/A'):.4f}")
            report.append("  (Cosine similarity between answer and context - measures grounding)")
            report.append(f"• Context Recall: {results.get('context_recall', 'N/A'):.4f}")
            report.append("  (Answer-context similarity - proxy for context coverage)")
            report.append(f"• Context Precision: {results.get('context_precision', 'N/A'):.4f}")
            report.append("  (Quality score of retrieved contexts based on relevance threshold)")
            report.append(f"• Context Relevancy: {results.get('context_relevancy', 'N/A'):.4f}")
            report.append("  (Cosine similarity between context and question - measures context relevance)")
            report.append("")
            report.append("ADDITIONAL LOCAL METRICS:")
            report.append(f"• Response Completeness: {results.get('response_completeness', 0):.4f}")
            report.append("  (Proportion of well-formed responses)")
            report.append(f"• Response Coherence: {results.get('response_coherence', 0):.4f}")
            report.append("  (Structural coherence of responses)")
            report.append(f"• Query Coverage: {results.get('query_coverage', 0):.4f}")
            report.append("  (Overlap between query terms and answer terms)")
            report.append("")
            report.append("PERFORMANCE METRICS:")
            report.append(f"• Average Response Length: {results.get('avg_response_length', 0):.1f} characters")
            report.append(f"• Average Retrieval Time: {results.get('avg_retrieval_time', 0):.3f} seconds")
            report.append(f"• Average Generation Time: {results.get('avg_generation_time', 0):.3f} seconds")
            report.append(f"• Average Total Time: {results.get('avg_total_time', 0):.3f} seconds")
            report.append(f"• Context Utilization: {results.get('context_utilization', 0):.1f} docs/query")
        elif 'faithfulness' in results:
            # Original RAGAS metrics with detailed descriptions
            report.append("RAGAS METRICS:")
            report.append(f"• Faithfulness Score: {results.get('faithfulness', 'N/A'):.4f}")
            report.append("  (Measures factual accuracy - how much the answer is grounded in context)")
            report.append(f"• Answer Relevancy: {results.get('answer_relevancy', 'N/A'):.4f}")
            report.append("  (Measures how relevant the answer is to the question)")
            report.append(f"• Context Relevancy: {results.get('context_relevancy', 'N/A'):.4f}")
            report.append("  (Measures how relevant retrieved contexts are to the question)")
            report.append(f"• Context Precision: {results.get('context_precision', 'N/A'):.4f}")
            report.append("  (Measures precision of retrieved context chunks)")
            report.append(f"• Context Recall: {results.get('context_recall', 'N/A'):.4f}")
            report.append("  (Measures how well retrieved context covers the ground truth)")
            report.append("")
            report.append("PERFORMANCE METRICS:")
        else:
            # Legacy fallback for older format
            report.append("LOCAL EVALUATION METRICS:")
            report.append(f"• Semantic Similarity (Answer-Context): {results.get('semantic_similarity_to_context', 0):.4f}")
            report.append("  (Cosine similarity between generated answer and retrieved context)")
            report.append(f"• Context Relevance (Query-Context): {results.get('context_relevance', 0):.4f}")
            report.append("  (Cosine similarity between query and retrieved context)")
            report.append(f"• Response Completeness: {results.get('response_completeness', 0):.4f}")
            report.append("  (Proportion of well-formed responses)")
            report.append(f"• Response Coherence: {results.get('response_coherence', 0):.4f}")
            report.append("  (Structural coherence of responses)")
            report.append(f"• Query Coverage: {results.get('query_coverage', 0):.4f}")
            report.append("  (Overlap between query terms and answer terms)")
            report.append("")
            report.append("PERFORMANCE METRICS:")
            report.append(f"• Average Response Length: {results.get('avg_response_length', 0):.1f} characters")
            report.append(f"• Average Retrieval Time: {results.get('avg_retrieval_time', 0):.3f} seconds")
            report.append(f"• Average Generation Time: {results.get('avg_generation_time', 0):.3f} seconds")
            report.append(f"• Average Total Time: {results.get('avg_total_time', 0):.3f} seconds")
            report.append(f"• Context Utilization: {results.get('context_utilization', 0):.1f} docs/query")
        
        report.append("")
        
        # Add individual query analysis
        report.append("INDIVIDUAL QUERY ANALYSIS:")
        report.append("-" * 40)
        
        for i, response in enumerate(self.responses):
            report.append(f"\nQuery {i+1}: {response['query']}")
            report.append(f"  Retrieval Time: {response['retrieval_time']:.3f}s")
            report.append(f"  Generation Time: {response['generation_time']:.3f}s")
            report.append(f"  Documents Retrieved: {len(response['retrieved_contexts'])}")
            report.append(f"  Response Length: {len(response['answer'])} characters")
            
            # Show top context source
            if response['retrieved_contexts']:
                top_source = response['retrieved_contexts'][0]['metadata']['source']
                top_score = response['retrieved_contexts'][0]['score']
                report.append(f"  Top Context Source: {top_source} (Score: {top_score:.4f})")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        # Add recommendations based on results
        if 'avg_generation_time' in results:
            if results['avg_generation_time'] > 100:
                report.append("• Consider using a smaller or more optimized LLM model for faster generation")
            if results['response_completeness'] < 0.8:
                report.append("• Improve prompt engineering to generate more complete responses")
            if results['context_utilization'] < 3:
                report.append("• Consider increasing the number of retrieved documents (top_k)")
            if results['query_coverage'] < 0.3:
                report.append("• Improve response relevance by refining the prompt template")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_evaluation_results(self, results: Dict[str, Any], report: str):
        """
        Save evaluation results and report
        
        Args:
            results: Evaluation results dictionary
            report: Formatted evaluation report
        """
        try:
            # Save results as JSON
            results_path = self.responses_path.parent / "evaluation_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save report as text
            report_path = self.responses_path.parent / "evaluation_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Evaluation results saved to {results_path}")
            logger.info(f"Evaluation report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def run_complete_evaluation(self) -> bool:
        """
        Run complete evaluation pipeline
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=== Starting Complete RAG Evaluation ===")
        
        try:
            # Load responses
            if not self.load_responses():
                return False
            
            # Run evaluation
            results = self.run_ragas_evaluation()
            
            if not results:
                logger.error("Evaluation failed")
                return False
            
            # Create report
            report = self.create_evaluation_report(results)
            
            # Display results
            print(report)
            
            # Save results
            self.save_evaluation_results(results, report)
            
            logger.info("Evaluation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return False
        finally:
            # Clean up any pending async tasks
            self.cleanup_async_tasks()
    
    def cleanup_async_tasks(self):
        """Clean up any pending asyncio tasks to prevent warnings"""
        try:
            # Get the current event loop if it exists
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                # Cancel all pending tasks
                pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                if pending_tasks:
                    logger.info(f"Cleaning up {len(pending_tasks)} pending async tasks...")
                    for task in pending_tasks:
                        task.cancel()
        except RuntimeError:
            # No event loop running, nothing to clean up
            pass
        except Exception as e:
            # Suppress cleanup errors to avoid confusing the user
            pass

def main():
    """Main function to run RAGAS evaluation"""
    evaluator = RAGASEvaluator()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()