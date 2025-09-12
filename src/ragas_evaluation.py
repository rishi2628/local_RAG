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
        
        try:
            # Prepare dataset
            dataset = self.prepare_evaluation_data()
            if dataset is None:
                return {}
            
            logger.info("Starting RAGAS evaluation...")
            
            # Define metrics to evaluate
            metrics = [
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_precision,
                context_recall
            ]
            
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
            )
            
            logger.info("RAGAS evaluation completed")
            return dict(result)
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            logger.info("Falling back to alternative evaluation...")
            return self.run_alternative_evaluation()
    
    def run_alternative_evaluation(self) -> Dict[str, Any]:
        """
        Alternative evaluation when RAGAS is not available
        Uses simple heuristics to assess response quality
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Running alternative evaluation using simple metrics...")
        
        try:
            metrics = {
                'avg_response_length': 0,
                'avg_retrieval_time': 0,
                'avg_generation_time': 0,
                'avg_total_time': 0,
                'context_utilization': 0,
                'response_completeness': 0,
                'query_coverage': 0
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
            
            # Calculate averages
            for key in metrics:
                if key in ['response_completeness']:
                    metrics[key] = metrics[key] / total_responses  # Proportion
                elif key != 'avg_response_length':
                    metrics[key] = metrics[key] / total_responses
            
            # Normalize completeness to 0-1 scale
            metrics['response_completeness'] = min(metrics['response_completeness'], 1.0)
            
            logger.info("Alternative evaluation completed")
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
        
        if 'faithfulness' in results:
            # RAGAS metrics
            report.append(f"Faithfulness Score: {results.get('faithfulness', 'N/A'):.4f}")
            report.append(f"Answer Relevancy: {results.get('answer_relevancy', 'N/A'):.4f}")
            report.append(f"Context Relevancy: {results.get('context_relevancy', 'N/A'):.4f}")
            report.append(f"Context Precision: {results.get('context_precision', 'N/A'):.4f}")
            report.append(f"Context Recall: {results.get('context_recall', 'N/A'):.4f}")
        else:
            # Alternative metrics
            report.append(f"Average Response Length: {results.get('avg_response_length', 0):.1f} characters")
            report.append(f"Average Retrieval Time: {results.get('avg_retrieval_time', 0):.3f} seconds")
            report.append(f"Average Generation Time: {results.get('avg_generation_time', 0):.3f} seconds")
            report.append(f"Average Total Time: {results.get('avg_total_time', 0):.3f} seconds")
            report.append(f"Context Utilization: {results.get('context_utilization', 0):.1f} docs/query")
            report.append(f"Response Completeness: {results.get('response_completeness', 0):.2f}")
            report.append(f"Query Coverage: {results.get('query_coverage', 0):.2f}")
        
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

def main():
    """Main function to run RAGAS evaluation"""
    evaluator = RAGASEvaluator()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()