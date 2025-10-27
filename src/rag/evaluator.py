"""
RAG Evaluator Module
Evaluation metrics for RAG system performance
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # Evaluation metrics
    evaluate_retrieval: bool = True
    evaluate_generation: bool = True
    evaluate_end_to_end: bool = True
    
    # Retrieval metrics
    retrieval_metrics: List[str] = None  # precision, recall, f1, mrr, ndcg
    
    # Generation metrics
    generation_metrics: List[str] = None  # bleu, rouge, bert_score, perplexity
    
    # End-to-end metrics
    end_to_end_metrics: List[str] = None  # answer_relevance, context_relevance, faithfulness
    
    def __post_init__(self):
        """Initialize default values"""
        if self.retrieval_metrics is None:
            self.retrieval_metrics = ["precision", "recall", "f1", "mrr"]
        if self.generation_metrics is None:
            self.generation_metrics = ["bleu", "rouge", "perplexity"]
        if self.end_to_end_metrics is None:
            self.end_to_end_metrics = ["answer_relevance", "context_relevance", "faithfulness"]


class RAGEvaluator:
    """Evaluator for RAG system performance"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        logger.info("RAGEvaluator initialized")
    
    def evaluate(self, rag_system, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate RAG system on test data"""
        try:
            logger.info(f"Evaluating RAG system on {len(test_data)} test examples")
            
            evaluation_results = {}
            
            # Evaluate retrieval
            if self.config.evaluate_retrieval:
                retrieval_metrics = self._evaluate_retrieval(rag_system, test_data)
                evaluation_results.update(retrieval_metrics)
            
            # Evaluate generation
            if self.config.evaluate_generation:
                generation_metrics = self._evaluate_generation(rag_system, test_data)
                evaluation_results.update(generation_metrics)
            
            # Evaluate end-to-end
            if self.config.evaluate_end_to_end:
                end_to_end_metrics = self._evaluate_end_to_end(rag_system, test_data)
                evaluation_results.update(end_to_end_metrics)
            
            logger.info(f"Evaluation completed: {evaluation_results}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def _evaluate_retrieval(self, rag_system, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        try:
            retrieval_scores = []
            
            for item in test_data:
                query = item.get('query', '')
                ground_truth_docs = item.get('relevant_documents', [])
                
                if not query:
                    continue
                
                # Retrieve documents
                retrieved_docs = rag_system.retrieve(query, top_k=5)
                
                # Calculate retrieval metrics
                if ground_truth_docs:
                    scores = self._calculate_retrieval_metrics(
                        retrieved_docs, ground_truth_docs
                    )
                    retrieval_scores.append(scores)
            
            # Average metrics
            if retrieval_scores:
                avg_metrics = {}
                for metric in self.config.retrieval_metrics:
                    values = [scores.get(metric, 0.0) for scores in retrieval_scores]
                    avg_metrics[f"avg_{metric}"] = np.mean(values)
                    avg_metrics[f"std_{metric}"] = np.std(values)
                
                return avg_metrics
            else:
                return {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1": 0.0}
                
        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {e}")
            return {}
    
    def _calculate_retrieval_metrics(self, retrieved_docs: List, ground_truth_docs: List) -> Dict[str, float]:
        """Calculate retrieval metrics for a single query"""
        try:
            # Get retrieved document IDs
            retrieved_ids = [doc.id for doc in retrieved_docs]
            ground_truth_ids = [doc.get('id', '') for doc in ground_truth_docs]
            
            # Calculate precision, recall, F1
            if retrieved_ids:
                true_positives = len(set(retrieved_ids) & set(ground_truth_ids))
                precision = true_positives / len(retrieved_ids)
                recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                precision = recall = f1 = 0.0
            
            # Calculate MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, doc in enumerate(retrieved_docs):
                if doc.id in ground_truth_ids:
                    mrr = 1.0 / (i + 1)
                    break
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mrr": mrr
            }
            
        except Exception as e:
            logger.error(f"Retrieval metrics calculation failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}
    
    def _evaluate_generation(self, rag_system, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate generation performance"""
        try:
            generation_scores = []
            
            for item in test_data:
                query = item.get('query', '')
                ground_truth_response = item.get('response', '')
                
                if not query:
                    continue
                
                # Generate response
                start_time = time.time()
                generated_response = rag_system.retrieve_and_generate(query)
                generation_time = time.time() - start_time
                
                response_text = generated_response.get('response', '')
                
                # Calculate generation metrics
                scores = self._calculate_generation_metrics(
                    response_text, ground_truth_response
                )
                scores['generation_time'] = generation_time
                generation_scores.append(scores)
            
            # Average metrics
            if generation_scores:
                avg_metrics = {}
                for metric in self.config.generation_metrics:
                    values = [scores.get(metric, 0.0) for scores in generation_scores]
                    avg_metrics[f"avg_{metric}"] = np.mean(values)
                    avg_metrics[f"std_{metric}"] = np.std(values)
                
                # Add generation time
                times = [scores.get('generation_time', 0.0) for scores in generation_scores]
                avg_metrics['avg_generation_time'] = np.mean(times)
                
                return avg_metrics
            else:
                return {"avg_bleu": 0.0, "avg_rouge": 0.0, "avg_perplexity": 0.0}
                
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
            return {}
    
    def _calculate_generation_metrics(self, generated_text: str, ground_truth_text: str) -> Dict[str, float]:
        """Calculate generation metrics for a single response"""
        try:
            # BLEU score (simplified)
            bleu_score = self._calculate_bleu(generated_text, ground_truth_text)
            
            # ROUGE score (simplified)
            rouge_score = self._calculate_rouge(generated_text, ground_truth_text)
            
            # Perplexity (simplified)
            perplexity = self._calculate_perplexity(generated_text)
            
            return {
                "bleu": bleu_score,
                "rouge": rouge_score,
                "perplexity": perplexity
            }
            
        except Exception as e:
            logger.error(f"Generation metrics calculation failed: {e}")
            return {"bleu": 0.0, "rouge": 0.0, "perplexity": 0.0}
    
    def _calculate_bleu(self, generated: str, reference: str) -> float:
        """Calculate simplified BLEU score"""
        try:
            # Simple n-gram overlap for demonstration
            gen_words = generated.lower().split()
            ref_words = reference.lower().split()
            
            if not gen_words or not ref_words:
                return 0.0
            
            # 1-gram precision
            gen_1grams = set(gen_words)
            ref_1grams = set(ref_words)
            overlap_1grams = len(gen_1grams & ref_1grams)
            precision_1 = overlap_1grams / len(gen_1grams) if gen_1grams else 0.0
            
            # 2-gram precision
            gen_2grams = set(zip(gen_words, gen_words[1:]))
            ref_2grams = set(zip(ref_words, ref_words[1:]))
            overlap_2grams = len(gen_2grams & ref_2grams)
            precision_2 = overlap_2grams / len(gen_2grams) if gen_2grams else 0.0
            
            # Simple BLEU approximation
            bleu = (precision_1 * precision_2) ** 0.5
            return min(bleu, 1.0)
            
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0
    
    def _calculate_rouge(self, generated: str, reference: str) -> float:
        """Calculate simplified ROUGE score"""
        try:
            gen_words = generated.lower().split()
            ref_words = reference.lower().split()
            
            if not gen_words or not ref_words:
                return 0.0
            
            # ROUGE-L (Longest Common Subsequence)
            def lcs_length(a, b):
                m, n = len(a), len(b)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if a[i-1] == b[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]
            
            lcs_len = lcs_length(gen_words, ref_words)
            rouge_l = lcs_len / len(ref_words) if ref_words else 0.0
            
            return min(rouge_l, 1.0)
            
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return 0.0
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate simplified perplexity"""
        try:
            words = text.split()
            if not words:
                return 0.0
            
            # Simple perplexity based on word frequency
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Calculate entropy
            total_words = len(words)
            entropy = 0.0
            for count in word_counts.values():
                prob = count / total_words
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            
            # Perplexity = 2^entropy
            perplexity = 2 ** entropy
            return perplexity
            
        except Exception as e:
            logger.error(f"Perplexity calculation failed: {e}")
            return 0.0
    
    def _evaluate_end_to_end(self, rag_system, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate end-to-end performance"""
        try:
            end_to_end_scores = []
            
            for item in test_data:
                query = item.get('query', '')
                expected_response = item.get('response', '')
                
                if not query:
                    continue
                
                # Get end-to-end response
                response = rag_system.retrieve_and_generate(query)
                generated_text = response.get('response', '')
                retrieved_docs = response.get('retrieved_documents', [])
                
                # Calculate end-to-end metrics
                scores = self._calculate_end_to_end_metrics(
                    query, generated_text, retrieved_docs, expected_response
                )
                end_to_end_scores.append(scores)
            
            # Average metrics
            if end_to_end_scores:
                avg_metrics = {}
                for metric in self.config.end_to_end_metrics:
                    values = [scores.get(metric, 0.0) for scores in end_to_end_scores]
                    avg_metrics[f"avg_{metric}"] = np.mean(values)
                    avg_metrics[f"std_{metric}"] = np.std(values)
                
                return avg_metrics
            else:
                return {"avg_answer_relevance": 0.0, "avg_context_relevance": 0.0, "avg_faithfulness": 0.0}
                
        except Exception as e:
            logger.error(f"End-to-end evaluation failed: {e}")
            return {}
    
    def _calculate_end_to_end_metrics(self, query: str, generated_text: str, 
                                    retrieved_docs: List, expected_response: str) -> Dict[str, float]:
        """Calculate end-to-end metrics for a single response"""
        try:
            # Answer relevance (how well the answer addresses the query)
            answer_relevance = self._calculate_answer_relevance(query, generated_text)
            
            # Context relevance (how relevant are the retrieved documents)
            context_relevance = self._calculate_context_relevance(query, retrieved_docs)
            
            # Faithfulness (how faithful is the answer to the retrieved context)
            faithfulness = self._calculate_faithfulness(generated_text, retrieved_docs)
            
            return {
                "answer_relevance": answer_relevance,
                "context_relevance": context_relevance,
                "faithfulness": faithfulness
            }
            
        except Exception as e:
            logger.error(f"End-to-end metrics calculation failed: {e}")
            return {"answer_relevance": 0.0, "context_relevance": 0.0, "faithfulness": 0.0}
    
    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """Calculate how relevant the answer is to the query"""
        try:
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            
            if not query_words or not answer_words:
                return 0.0
            
            # Word overlap
            overlap = len(query_words & answer_words)
            relevance = overlap / len(query_words)
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Answer relevance calculation failed: {e}")
            return 0.0
    
    def _calculate_context_relevance(self, query: str, retrieved_docs: List) -> float:
        """Calculate how relevant the retrieved documents are to the query"""
        try:
            if not retrieved_docs:
                return 0.0
            
            # Average similarity score of retrieved documents
            similarities = [doc.get('similarity_score', 0.0) for doc in retrieved_docs]
            avg_similarity = np.mean(similarities)
            
            return min(avg_similarity, 1.0)
            
        except Exception as e:
            logger.error(f"Context relevance calculation failed: {e}")
            return 0.0
    
    def _calculate_faithfulness(self, generated_text: str, retrieved_docs: List) -> float:
        """Calculate how faithful the generated text is to the retrieved context"""
        try:
            if not retrieved_docs:
                return 0.0
            
            # Combine retrieved context
            context_text = " ".join([doc.get('content', '') for doc in retrieved_docs])
            
            # Calculate word overlap between generated text and context
            gen_words = set(generated_text.lower().split())
            context_words = set(context_text.lower().split())
            
            if not gen_words:
                return 0.0
            
            overlap = len(gen_words & context_words)
            faithfulness = overlap / len(gen_words)
            
            return min(faithfulness, 1.0)
            
        except Exception as e:
            logger.error(f"Faithfulness calculation failed: {e}")
            return 0.0