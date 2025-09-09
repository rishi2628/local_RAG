"""
RAG evaluation module using RAGAS framework.
Evaluates retrieval quality and generation performance.
"""

import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


class RAGEvaluator:
    """Evaluates RAG pipeline performance using various metrics."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG evaluator.
        
        Args:
            embedding_model_name: Model for computing semantic similarities
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
    
    def compute_retrieval_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute retrieval-specific metrics.
        
        Args:
            results: List of RAG query results
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {
            'avg_num_retrieved': 0,
            'avg_retrieval_score': 0,
            'retrieval_score_std': 0
        }
        
        all_scores = []
        num_retrieved_list = []
        
        for result in results:
            chunks = result.get('retrieved_chunks', [])
            num_retrieved_list.append(len(chunks))
            
            if chunks:
                scores = [chunk.get('similarity_score', 0) for chunk in chunks]
                all_scores.extend(scores)
        
        if all_scores:
            metrics['avg_retrieval_score'] = np.mean(all_scores)
            metrics['retrieval_score_std'] = np.std(all_scores)
        
        if num_retrieved_list:
            metrics['avg_num_retrieved'] = np.mean(num_retrieved_list)
        
        return metrics
    
    def compute_answer_relevance(self, questions: List[str], answers: List[str]) -> List[float]:
        """
        Compute answer relevance scores using semantic similarity.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            
        Returns:
            List of relevance scores
        """
        if len(questions) != len(answers):
            raise ValueError("Number of questions and answers must match")
        
        # Encode questions and answers
        question_embeddings = self.embedding_model.encode(questions)
        answer_embeddings = self.embedding_model.encode(answers)
        
        # Compute cosine similarities
        relevance_scores = []
        for i in range(len(questions)):
            similarity = cosine_similarity(
                [question_embeddings[i]], 
                [answer_embeddings[i]]
            )[0][0]
            relevance_scores.append(float(similarity))
        
        return relevance_scores
    
    def compute_context_relevance(self, questions: List[str], contexts: List[List[str]]) -> List[float]:
        """
        Compute context relevance scores.
        
        Args:
            questions: List of questions
            contexts: List of context chunks for each question
            
        Returns:
            List of context relevance scores
        """
        relevance_scores = []
        
        for question, context_list in zip(questions, contexts):
            if not context_list:
                relevance_scores.append(0.0)
                continue
            
            # Encode question and contexts
            question_embedding = self.embedding_model.encode([question])
            context_embeddings = self.embedding_model.encode(context_list)
            
            # Compute similarities and take the mean
            similarities = cosine_similarity(question_embedding, context_embeddings)[0]
            avg_similarity = np.mean(similarities)
            relevance_scores.append(float(avg_similarity))
        
        return relevance_scores
    
    def evaluate_answer_quality(self, answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate answer quality using simple heuristics.
        
        Args:
            answers: List of generated answers
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'avg_answer_length': 0,
            'avg_word_count': 0,
            'completeness_score': 0,
            'informativeness_score': 0
        }
        
        if not answers:
            return metrics
        
        lengths = [len(answer) for answer in answers]
        word_counts = [len(answer.split()) for answer in answers]
        
        metrics['avg_answer_length'] = np.mean(lengths)
        metrics['avg_word_count'] = np.mean(word_counts)
        
        # Simple completeness heuristic (answers with reasonable length)
        complete_answers = sum(1 for length in lengths if length > 50)
        metrics['completeness_score'] = complete_answers / len(answers)
        
        # Simple informativeness heuristic (answers with diverse vocabulary)
        total_unique_words = set()
        for answer in answers:
            words = answer.lower().split()
            total_unique_words.update(words)
        
        if word_counts:
            metrics['informativeness_score'] = len(total_unique_words) / np.mean(word_counts)
        
        return metrics
    
    def check_hallucination(self, answers: List[str], contexts: List[List[str]]) -> List[float]:
        """
        Simple hallucination detection using context overlap.
        
        Args:
            answers: List of generated answers
            contexts: List of context chunks for each answer
            
        Returns:
            List of hallucination scores (lower is better)
        """
        hallucination_scores = []
        
        for answer, context_list in zip(answers, contexts):
            if not context_list:
                hallucination_scores.append(1.0)  # High hallucination if no context
                continue
            
            # Simple word overlap metric
            answer_words = set(answer.lower().split())
            context_words = set()
            
            for context in context_list:
                context_words.update(context.lower().split())
            
            if not answer_words:
                hallucination_scores.append(1.0)
                continue
            
            # Calculate overlap ratio
            overlap = len(answer_words.intersection(context_words))
            overlap_ratio = overlap / len(answer_words)
            
            # Hallucination score is inverse of overlap (higher overlap = lower hallucination)
            hallucination_score = 1.0 - overlap_ratio
            hallucination_scores.append(hallucination_score)
        
        return hallucination_scores
    
    def comprehensive_evaluation(self, rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of RAG results.
        
        Args:
            rag_results: List of RAG pipeline results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Extract data from results
        questions = [result['question'] for result in rag_results]
        answers = [result['response'] for result in rag_results]
        contexts = []
        
        for result in rag_results:
            chunks = result.get('retrieved_chunks', [])
            context_texts = [chunk['text'] for chunk in chunks]
            contexts.append(context_texts)
        
        # Compute various metrics
        evaluation = {}
        
        # Retrieval metrics
        evaluation['retrieval_metrics'] = self.compute_retrieval_metrics(rag_results)
        
        # Answer relevance
        answer_relevance = self.compute_answer_relevance(questions, answers)
        evaluation['answer_relevance'] = {
            'scores': answer_relevance,
            'mean': np.mean(answer_relevance),
            'std': np.std(answer_relevance)
        }
        
        # Context relevance
        context_relevance = self.compute_context_relevance(questions, contexts)
        evaluation['context_relevance'] = {
            'scores': context_relevance,
            'mean': np.mean(context_relevance),
            'std': np.std(context_relevance)
        }
        
        # Answer quality
        evaluation['answer_quality'] = self.evaluate_answer_quality(answers)
        
        # Hallucination detection
        hallucination_scores = self.check_hallucination(answers, contexts)
        evaluation['hallucination'] = {
            'scores': hallucination_scores,
            'mean': np.mean(hallucination_scores),
            'std': np.std(hallucination_scores)
        }
        
        return evaluation
    
    def generate_evaluation_report(self, evaluation: Dict[str, Any], output_file: str = None) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation: Evaluation results dictionary
            output_file: Optional file to save the report
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("RAG PIPELINE EVALUATION REPORT")
        report.append("=" * 60)
        
        # Retrieval metrics
        retrieval = evaluation.get('retrieval_metrics', {})
        report.append("\n📋 RETRIEVAL METRICS:")
        report.append(f"  • Average documents retrieved: {retrieval.get('avg_num_retrieved', 0):.2f}")
        report.append(f"  • Average retrieval score: {retrieval.get('avg_retrieval_score', 0):.4f}")
        report.append(f"  • Retrieval score std dev: {retrieval.get('retrieval_score_std', 0):.4f}")
        
        # Answer relevance
        answer_rel = evaluation.get('answer_relevance', {})
        report.append(f"\n🎯 ANSWER RELEVANCE:")
        report.append(f"  • Mean relevance score: {answer_rel.get('mean', 0):.4f}")
        report.append(f"  • Standard deviation: {answer_rel.get('std', 0):.4f}")
        
        # Context relevance
        context_rel = evaluation.get('context_relevance', {})
        report.append(f"\n📄 CONTEXT RELEVANCE:")
        report.append(f"  • Mean context relevance: {context_rel.get('mean', 0):.4f}")
        report.append(f"  • Standard deviation: {context_rel.get('std', 0):.4f}")
        
        # Answer quality
        quality = evaluation.get('answer_quality', {})
        report.append(f"\n✨ ANSWER QUALITY:")
        report.append(f"  • Average answer length: {quality.get('avg_answer_length', 0):.1f} chars")
        report.append(f"  • Average word count: {quality.get('avg_word_count', 0):.1f} words")
        report.append(f"  • Completeness score: {quality.get('completeness_score', 0):.2f}")
        report.append(f"  • Informativeness score: {quality.get('informativeness_score', 0):.2f}")
        
        # Hallucination
        hallucination = evaluation.get('hallucination', {})
        report.append(f"\n🚨 HALLUCINATION DETECTION:")
        report.append(f"  • Mean hallucination score: {hallucination.get('mean', 0):.4f} (lower is better)")
        report.append(f"  • Standard deviation: {hallucination.get('std', 0):.4f}")
        
        # Overall assessment
        report.append(f"\n📊 OVERALL ASSESSMENT:")
        
        # Simple scoring system
        scores = []
        if answer_rel.get('mean', 0) > 0.7:
            scores.append("Answer relevance: GOOD")
        elif answer_rel.get('mean', 0) > 0.5:
            scores.append("Answer relevance: FAIR")
        else:
            scores.append("Answer relevance: POOR")
        
        if context_rel.get('mean', 0) > 0.7:
            scores.append("Context relevance: GOOD")
        elif context_rel.get('mean', 0) > 0.5:
            scores.append("Context relevance: FAIR")
        else:
            scores.append("Context relevance: POOR")
        
        if hallucination.get('mean', 1) < 0.3:
            scores.append("Hallucination: LOW (good)")
        elif hallucination.get('mean', 1) < 0.6:
            scores.append("Hallucination: MODERATE")
        else:
            scores.append("Hallucination: HIGH (concerning)")
        
        for score in scores:
            report.append(f"  • {score}")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            report.append(f"\nReport saved to: {output_file}")
        
        return report_text


def evaluate_rag_pipeline(rag_results: List[Dict[str, Any]], output_file: str = "evaluation_report.txt") -> Dict[str, Any]:
    """
    Convenience function to evaluate RAG pipeline results.
    
    Args:
        rag_results: List of RAG pipeline results
        output_file: File to save evaluation report
        
    Returns:
        Complete evaluation results
    """
    evaluator = RAGEvaluator()
    evaluation = evaluator.comprehensive_evaluation(rag_results)
    report = evaluator.generate_evaluation_report(evaluation, output_file)
    
    print(report)
    
    return evaluation


if __name__ == "__main__":
    # Example usage with dummy data
    from llm_interface import RAGPipeline, LocalLLM
    from document_processor import DocumentProcessor
    
    # This would typically be called after running the RAG pipeline
    print("RAG Evaluator module loaded successfully!")
    print("Use evaluate_rag_pipeline(results) to evaluate your RAG pipeline results.")
