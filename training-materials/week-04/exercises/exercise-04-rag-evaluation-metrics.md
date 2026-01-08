# Exercise 4: RAG Evaluation Metrics

**Week 4 - RAG Fundamentals**

## Overview

"You can't improve what you don't measure." This exercise focuses on building a comprehensive evaluation framework for RAG systems. You'll implement retrieval metrics, generation quality metrics, end-to-end evaluation, and create a testing harness for continuous quality monitoring.

**Time:** 90 minutes  
**Difficulty:** Advanced

## Learning Objectives

By completing this exercise, you will:
- Implement standard retrieval metrics (Recall, Precision, MRR, NDCG)
- Measure generation quality (faithfulness, relevance, completeness)
- Build automated evaluation pipelines
- Create test datasets for RAG systems
- Design A/B testing frameworks
- Implement continuous quality monitoring

## Prerequisites

- Completed Week 4 Labs 1-3
- Completed Exercises 1-3
- Understanding of evaluation metrics
- OpenAI API access (for LLM-as-judge)

## Setup

```python
import openai
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime

client = openai.OpenAI()
```

## Part 1: Retrieval Metrics

### Task 1.1: Implement Core Retrieval Metrics

```python
@dataclass
class RetrievalTestCase:
    """Test case for retrieval evaluation."""
    query_id: str
    query: str
    relevant_doc_ids: List[str]  # Ground truth relevant documents
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """Result from retrieval system."""
    query_id: str
    retrieved_doc_ids: List[str]  # In ranked order
    scores: List[float]  # Relevance scores
    latency_ms: float

class RetrievalMetrics:
    """Calculate retrieval quality metrics."""
    
    @staticmethod
    def recall_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        TODO: Calculate Recall@K.
        
        Recall@K = (# relevant docs in top-k) / (# total relevant docs)
        
        Measures: What % of all relevant documents were retrieved?
        
        Example:
        - relevant_docs = ["doc1", "doc2", "doc3"]
        - retrieved_docs = ["doc1", "doc4", "doc2", "doc5"]
        - k = 3
        - recall@3 = 2/3 = 0.667 (found doc1 and doc2, missed doc3)
        """
        pass
    
    @staticmethod
    def precision_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        TODO: Calculate Precision@K.
        
        Precision@K = (# relevant docs in top-k) / k
        
        Measures: What % of retrieved documents are relevant?
        
        Example:
        - relevant_docs = ["doc1", "doc2", "doc3"]
        - retrieved_docs = ["doc1", "doc4", "doc2", "doc5"]
        - k = 3
        - precision@3 = 2/3 = 0.667 (2 of top-3 are relevant)
        """
        pass
    
    @staticmethod
    def f1_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        TODO: Calculate F1@K.
        
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
        
        Harmonic mean balancing precision and recall.
        """
        pass
    
    @staticmethod
    def mean_reciprocal_rank(
        relevant_docs: List[str],
        retrieved_docs: List[str]
    ) -> float:
        """
        TODO: Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / rank of first relevant document
        
        Measures: How high is the first relevant result?
        
        Example:
        - relevant_docs = ["doc1", "doc2"]
        - retrieved_docs = ["doc4", "doc1", "doc5"]
        - First relevant doc (doc1) is at rank 2
        - MRR = 1/2 = 0.5
        
        Return 0 if no relevant docs found.
        """
        pass
    
    @staticmethod
    def average_precision(
        relevant_docs: List[str],
        retrieved_docs: List[str]
    ) -> float:
        """
        TODO: Calculate Average Precision (AP).
        
        AP = (sum of Precision@k for each relevant doc k) / (# relevant docs)
        
        Only sum precision at ranks where relevant docs appear.
        
        Example:
        - relevant_docs = ["doc1", "doc2", "doc3"]
        - retrieved_docs = ["doc4", "doc1", "doc5", "doc2"]
        - Relevant docs at ranks [2, 4]
        - Precision@2 = 1/2 = 0.5 (1 relevant in top 2)
        - Precision@4 = 2/4 = 0.5 (2 relevant in top 4)
        - AP = (0.5 + 0.5) / 3 = 0.333
        
        Better than MRR because it considers all relevant docs.
        """
        pass
    
    @staticmethod
    def ndcg_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        TODO: Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        DCG@K = sum_{i=1}^{k} (relevance_i / log2(i + 1))
        NDCG@K = DCG@K / IDCG@K
        
        where IDCG@K is DCG of ideal ranking (all relevant docs at top)
        
        If relevance_scores not provided, use binary relevance:
        - 1 if doc is relevant
        - 0 if not relevant
        
        NDCG accounts for ranking quality, not just presence.
        
        Example:
        - relevant_docs = ["doc1", "doc2"]
        - retrieved_docs = ["doc1", "doc3", "doc2"]
        - k = 3
        - DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
        - IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
        - NDCG = 1.5 / 1.631 = 0.920
        """
        pass
    
    @staticmethod
    def mean_average_precision(
        test_cases: List[RetrievalTestCase],
        results: List[RetrievalResult]
    ) -> float:
        """
        TODO: Calculate Mean Average Precision (MAP) across all queries.
        
        MAP = average of AP scores for all queries
        
        Standard metric for overall retrieval quality.
        """
        pass

# TODO: Create test dataset
test_cases = [
    RetrievalTestCase(
        query_id="q1",
        query="What is machine learning?",
        relevant_doc_ids=["doc1", "doc2", "doc5"],
        metadata={"difficulty": "easy"}
    ),
    RetrievalTestCase(
        query_id="q2",
        query="How do transformers work?",
        relevant_doc_ids=["doc7", "doc9"],
        metadata={"difficulty": "hard"}
    ),
    # Add more test cases...
]

# TODO: Implement all metrics
# TODO: Test on sample retrieval results
# TODO: Verify metric calculations are correct
```

### Task 1.2: Build Retrieval Evaluator

```python
class RetrievalEvaluator:
    """Comprehensive retrieval evaluation."""
    
    def __init__(self, test_cases: List[RetrievalTestCase]):
        """TODO: Initialize evaluator with test dataset."""
        self.test_cases = {tc.query_id: tc for tc in test_cases}
        self.metrics = RetrievalMetrics()
    
    def evaluate(
        self,
        retrieval_system,
        k_values: List[int] = [1, 3, 5, 10],
        return_per_query: bool = False
    ) -> Dict[str, Any]:
        """
        TODO: Evaluate retrieval system on all test cases.
        
        For each test case:
        1. Run retrieval
        2. Calculate all metrics
        3. Track latency
        
        Aggregate metrics:
        - Mean of each metric across all queries
        - Breakdown by metadata (e.g., difficulty)
        - Per-query results if requested
        
        Return:
        {
            "overall": {
                "recall@1": float,
                "recall@3": float,
                "recall@5": float,
                "precision@1": float,
                "mrr": float,
                "map": float,
                "ndcg@5": float,
                "avg_latency_ms": float
            },
            "by_difficulty": {
                "easy": {...},
                "hard": {...}
            },
            "per_query": {...}  # If return_per_query=True
        }
        """
        pass
    
    def evaluate_with_confidence_intervals(
        self,
        retrieval_system,
        k: int = 5,
        n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        TODO: Calculate metrics with confidence intervals using bootstrap.
        
        Bootstrap procedure:
        1. For each iteration:
           - Sample test cases with replacement
           - Calculate metrics on sample
        2. After n_bootstrap iterations:
           - Calculate mean (point estimate)
           - Calculate 95% confidence interval (2.5th and 97.5th percentiles)
        
        Return: {metric_name: (mean, ci_lower, ci_upper)}
        
        Useful for:
        - Assessing statistical significance of improvements
        - Understanding measurement uncertainty
        - Comparing systems rigorously
        """
        pass
    
    def compare_systems(
        self,
        systems: Dict[str, Any],
        k: int = 5
    ) -> pd.DataFrame:
        """
        TODO: Compare multiple retrieval systems.
        
        Args:
        - systems: Dict of {system_name: retrieval_system}
        - k: Evaluation at top-k
        
        Return DataFrame:
        - Rows: Systems
        - Columns: Metrics
        - Cells: Metric values
        
        Add significance testing:
        - Indicate which systems are significantly better (p < 0.05)
        """
        pass
    
    def analyze_failure_cases(
        self,
        retrieval_system,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        TODO: Identify and analyze queries with poor retrieval.
        
        Find queries where recall@5 < threshold.
        
        For each failure:
        - Show query
        - Show what was retrieved vs what should have been
        - Analyze why retrieval failed (using embeddings, keyword analysis)
        - Suggest improvements
        
        Return list of failure analyses.
        """
        pass

# TODO: Run comprehensive evaluation
# TODO: Compare multiple systems (baseline, hybrid, optimized)
# TODO: Analyze failure cases and document patterns
```

## Part 2: Generation Quality Metrics

### Task 2.1: Implement Faithfulness Checking

```python
class GenerationMetrics:
    """Evaluate generation quality."""
    
    def __init__(self, client: openai.OpenAI):
        """TODO: Initialize with OpenAI client for LLM-as-judge."""
        self.client = client
    
    def check_faithfulness(
        self,
        context: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        TODO: Check if answer is faithful to context.
        
        Faithfulness = Answer only contains information from context
        
        Use LLM as judge with prompt:
        "Given the context and answer below, determine if the answer
         contains any information not present in the context.
         
         Context: {context}
         Answer: {answer}
         
         Respond with JSON:
         {
             "is_faithful": true/false,
             "hallucinated_claims": ["claim1", "claim2"],
             "explanation": "..."
         }"
        
        Return parsed response.
        
        Critical metric for RAG - prevents hallucination.
        """
        pass
    
    def check_relevance(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        TODO: Check if answer is relevant to query.
        
        Relevance = Answer addresses the query
        
        Use LLM to score 0-10:
        - 10: Directly answers question
        - 5: Partially relevant
        - 0: Completely unrelated
        
        Also identify:
        - What aspects of query were addressed
        - What aspects were missed
        
        Return:
        {
            "relevance_score": float,  # 0-10
            "addressed_aspects": List[str],
            "missed_aspects": List[str]
        }
        """
        pass
    
    def check_completeness(
        self,
        query: str,
        answer: str,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        TODO: Check if answer is complete.
        
        Completeness = Answer covers all important aspects
        
        If expected_answer provided:
        - Compare coverage with expected answer
        - Identify missing key points
        
        Otherwise:
        - Assess completeness based on query type
        - Factual queries: Is fact stated clearly?
        - Conceptual queries: Are all aspects explained?
        - Procedural queries: Are all steps included?
        
        Return:
        {
            "completeness_score": float,  # 0-10
            "coverage_percentage": float,  # % of expected points covered
            "missing_points": List[str]
        }
        """
        pass
    
    def check_correctness(
        self,
        answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        TODO: Check factual correctness against ground truth.
        
        Use LLM to compare answer with ground truth:
        - Identify factual errors
        - Identify correct facts
        - Calculate accuracy
        
        Return:
        {
            "correctness_score": float,  # 0-10
            "correct_facts": List[str],
            "incorrect_facts": List[str],
            "explanation": str
        }
        """
        pass
    
    def compute_citation_quality(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        TODO: Evaluate quality of source citations in answer.
        
        Check:
        1. Citation coverage: What % of claims have citations?
        2. Citation accuracy: Do citations support claims?
        3. Citation completeness: Are all relevant sources cited?
        
        Use LLM to:
        - Extract claims from answer
        - Verify each claim against cited sources
        - Identify uncited claims
        
        Return:
        {
            "citation_coverage": float,  # % of claims cited
            "citation_accuracy": float,  # % of citations that are correct
            "uncited_claims": List[str],
            "incorrect_citations": List[Dict]  # claim + why citation is wrong
        }
        """
        pass

# TODO: Test each metric on sample answers
# TODO: Verify LLM judge consistency (run same eval multiple times)
# TODO: Measure correlation between automated and human judgments
```

### Task 2.2: Build Generation Evaluator

```python
@dataclass
class GenerationTestCase:
    """Test case for generation evaluation."""
    query: str
    context: str
    expected_answer: Optional[str] = None
    evaluation_aspects: List[str] = field(default_factory=list)  # What to evaluate
    metadata: Dict[str, Any] = field(default_factory=dict)

class GenerationEvaluator:
    """Comprehensive generation evaluation."""
    
    def __init__(
        self,
        test_cases: List[GenerationTestCase],
        client: openai.OpenAI
    ):
        """TODO: Initialize evaluator."""
        self.test_cases = test_cases
        self.metrics = GenerationMetrics(client)
    
    def evaluate(
        self,
        rag_system,
        aspects: List[str] = ["faithfulness", "relevance", "completeness"]
    ) -> Dict[str, Any]:
        """
        TODO: Evaluate RAG system's generation quality.
        
        For each test case:
        1. Generate answer using RAG system
        2. Evaluate specified aspects
        3. Aggregate scores
        
        Return:
        {
            "overall_scores": {
                "faithfulness": float,
                "relevance": float,
                "completeness": float,
                ...
            },
            "per_case_results": [...],
            "failure_cases": [...]  # Cases with scores < 5
        }
        """
        pass
    
    def evaluate_consistency(
        self,
        rag_system,
        n_runs: int = 3,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        TODO: Evaluate consistency of generation.
        
        Run same query n_runs times and measure:
        1. Answer variation (how much do answers differ?)
        2. Consistency of key facts (are same facts stated?)
        3. Stability of scores (do quality scores vary?)
        
        High temperature should show more variation.
        Low temperature should be more consistent.
        
        Return:
        {
            "answer_similarity": float,  # Avg cosine sim between runs
            "fact_consistency": float,  # % of facts appearing in all runs
            "score_stability": float  # Std dev of quality scores
        }
        """
        pass
    
    def evaluate_with_human_baseline(
        self,
        human_annotations: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        TODO: Compare automated metrics with human judgments.
        
        Args:
        - human_annotations: {query_id: {metric: human_score}}
        
        Calculate correlation between automated and human scores.
        
        Metrics to compare:
        - Pearson correlation
        - Spearman correlation (rank-based)
        - Mean absolute error
        
        Return correlations for each metric.
        
        High correlation = automated metrics are reliable.
        """
        pass

# TODO: Build comprehensive generation test suite
# TODO: Run evaluation on RAG system
# TODO: Compare with human annotations if available
```

## Part 3: End-to-End RAG Evaluation

### Task 3.1: Build Complete Evaluation Pipeline

```python
@dataclass
class RAGTestCase:
    """Complete test case for RAG system."""
    test_id: str
    query: str
    relevant_doc_ids: List[str]  # For retrieval eval
    expected_answer: Optional[str] = None  # For generation eval
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class RAGEvaluator:
    """Complete end-to-end RAG evaluation."""
    
    def __init__(
        self,
        test_cases: List[RAGTestCase],
        client: openai.OpenAI
    ):
        """TODO: Initialize complete evaluator."""
        self.test_cases = test_cases
        self.retrieval_evaluator = None  # Initialize with retrieval test cases
        self.generation_evaluator = None  # Initialize with generation test cases
    
    def evaluate_end_to_end(
        self,
        rag_system,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        TODO: Complete end-to-end evaluation.
        
        For each test case:
        1. Query RAG system
        2. Evaluate retrieval quality
        3. Evaluate generation quality
        4. Measure latency
        5. Calculate cost (tokens used)
        
        Return comprehensive report:
        {
            "retrieval_metrics": {...},
            "generation_metrics": {...},
            "performance_metrics": {
                "avg_latency_ms": float,
                "p95_latency_ms": float,
                "avg_cost_per_query": float,
                "tokens_per_query": float
            },
            "quality_score": float,  # Combined score 0-100
            "per_case_results": [...]
        }
        """
        pass
    
    def calculate_quality_score(
        self,
        retrieval_metrics: Dict[str, float],
        generation_metrics: Dict[str, float]
    ) -> float:
        """
        TODO: Calculate overall quality score.
        
        Combine metrics into single score:
        - Retrieval quality (40%): recall@5, ndcg@5
        - Generation quality (60%): faithfulness, relevance, completeness
        
        Formula:
        quality = 0.4 * (0.5*recall@5 + 0.5*ndcg@5) + 
                  0.6 * (0.4*faithfulness + 0.3*relevance + 0.3*completeness)
        
        Scale to 0-100.
        """
        pass
    
    def run_ablation_study(
        self,
        rag_system_variants: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        TODO: Run ablation study on RAG components.
        
        Test variants:
        - Baseline (simple vector search + generation)
        - + Hybrid search
        - + Query expansion
        - + Reranking
        - + Context optimization
        - Full system (all optimizations)
        
        For each variant:
        - Run full evaluation
        - Compare to baseline
        - Calculate improvement
        
        Return comparison table showing impact of each component.
        """
        pass
    
    def track_over_time(
        self,
        rag_system,
        results_history: List[Dict[str, Any]],
        alert_on_degradation: bool = True
    ) -> Dict[str, Any]:
        """
        TODO: Track quality metrics over time.
        
        Use for:
        - Continuous monitoring
        - Detecting quality degradation
        - Validating improvements
        
        Compare current results with historical baseline:
        - Calculate trend (improving/degrading)
        - Detect significant changes
        - Alert if metrics drop > 5%
        
        Return:
        {
            "current_metrics": {...},
            "baseline_metrics": {...},
            "changes": {metric: {"absolute": float, "relative_pct": float}},
            "alerts": [...]  # If degradation detected
        }
        """
        pass

# TODO: Build complete test suite
# TODO: Run end-to-end evaluation
# TODO: Conduct ablation study
```

## Part 4: Test Dataset Creation

### Task 4.1: Generate Synthetic Test Data

```python
class TestDatasetGenerator:
    """Generate test datasets for RAG evaluation."""
    
    def __init__(self, client: openai.OpenAI):
        """TODO: Initialize generator."""
        self.client = client
    
    def generate_questions_from_documents(
        self,
        documents: List[Dict[str, Any]],
        n_questions_per_doc: int = 5
    ) -> List[RAGTestCase]:
        """
        TODO: Generate questions from documents using LLM.
        
        For each document:
        1. Use LLM to generate n questions answerable from document
        2. Vary question types:
           - Factual: "What is...?"
           - Conceptual: "Why does...?"
           - Procedural: "How to...?"
           - Comparative: "What's the difference between...?"
        3. Create test case with document as relevant source
        4. Optionally generate expected answer
        
        Prompt:
        "Given this document, generate {n} diverse questions that can be
         answered using information in the document. Include factual,
         conceptual, and procedural questions.
         
         Document: {document_content}
         
         Return JSON array of:
         [
             {
                 "question": "...",
                 "question_type": "factual/conceptual/procedural",
                 "expected_answer": "...",
                 "difficulty": "easy/medium/hard"
             },
             ...
         ]"
        
        Return list of test cases.
        """
        pass
    
    def generate_adversarial_questions(
        self,
        documents: List[Dict[str, Any]],
        n_questions: int = 10
    ) -> List[RAGTestCase]:
        """
        TODO: Generate challenging questions to stress-test system.
        
        Types of adversarial questions:
        1. Ambiguous: "How does it work?" (unclear referent)
        2. Multi-hop: Requires combining info from multiple documents
        3. Counterfactual: "What if X were different?"
        4. Unanswerable: No information in documents
        5. Contradictory: Documents contain conflicting info
        
        Use LLM to generate these challenging cases.
        
        Return test cases with metadata indicating adversarial type.
        """
        pass
    
    def create_golden_dataset(
        self,
        n_cases: int = 100,
        include_answers: bool = True
    ) -> List[RAGTestCase]:
        """
        TODO: Create curated golden test set.
        
        Combine:
        - Auto-generated questions (70%)
        - Manually crafted edge cases (20%)
        - Adversarial questions (10%)
        
        If include_answers:
        - Use LLM to generate expected answers
        - Manually review for quality
        
        Save to file for reuse.
        """
        pass
    
    def validate_test_cases(
        self,
        test_cases: List[RAGTestCase],
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        TODO: Validate test case quality.
        
        Check:
        1. Answerability: Can question be answered from documents?
        2. Clarity: Is question clear and unambiguous?
        3. Relevance: Do specified relevant_doc_ids actually contain answer?
        4. Diversity: Are questions sufficiently varied?
        5. Coverage: Do questions cover all documents?
        
        Return validation report with quality scores.
        """
        pass

# TODO: Generate test dataset from your documents
# TODO: Validate test case quality
# TODO: Save golden dataset for reuse
```

## Part 5: A/B Testing Framework

### Task 5.1: Implement A/B Testing

```python
class ABTestRunner:
    """Run A/B tests for RAG system improvements."""
    
    def __init__(
        self,
        test_cases: List[RAGTestCase],
        evaluator: RAGEvaluator
    ):
        """TODO: Initialize A/B test runner."""
        self.test_cases = test_cases
        self.evaluator = evaluator
    
    def run_ab_test(
        self,
        system_a: Any,
        system_b: Any,
        system_a_name: str = "Control",
        system_b_name: str = "Variant",
        metric: str = "quality_score"
    ) -> Dict[str, Any]:
        """
        TODO: Run A/B test comparing two systems.
        
        Steps:
        1. Run both systems on all test cases
        2. Calculate metrics for each
        3. Perform statistical significance testing:
           - Use t-test for comparing means
           - Calculate effect size (Cohen's d)
           - Determine if difference is significant (p < 0.05)
        4. Generate report
        
        Return:
        {
            "system_a": {
                "name": str,
                "metrics": {...},
                "avg_score": float
            },
            "system_b": {
                "name": str,
                "metrics": {...},
                "avg_score": float
            },
            "comparison": {
                "metric": str,
                "difference": float,
                "pct_improvement": float,
                "p_value": float,
                "is_significant": bool,
                "effect_size": float,
                "conclusion": str
            }
        }
        """
        pass
    
    def run_multivariate_test(
        self,
        systems: Dict[str, Any],
        metrics: List[str] = ["quality_score", "latency_ms"]
    ) -> pd.DataFrame:
        """
        TODO: Compare multiple systems across multiple metrics.
        
        Args:
        - systems: Dict of {name: rag_system}
        - metrics: List of metrics to compare
        
        Return DataFrame:
        - Rows: Systems
        - Columns: Metrics
        - Cells: Scores (with significance markers)
        
        Add:
        - Statistical significance indicators (* p<0.05, ** p<0.01)
        - Ranking for each metric
        - Overall winner (best average rank)
        """
        pass
    
    def calculate_minimum_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        TODO: Calculate required test cases for statistical power.
        
        Use power analysis to determine:
        - How many test cases needed to detect improvement of size X
        - With confidence level alpha
        - With statistical power (1 - beta)
        
        Formula uses t-distribution for sample size calculation.
        
        Return minimum number of test cases needed.
        
        Useful for:
        - Planning test dataset size
        - Determining if test results are meaningful
        """
        pass

# TODO: Run A/B test comparing baseline vs optimized system
# TODO: Calculate statistical significance
# TODO: Generate comparison report
```

## Part 6: Continuous Monitoring

### Task 6.1: Build Production Monitoring Dashboard

```python
class RAGMonitor:
    """Monitor RAG system quality in production."""
    
    def __init__(
        self,
        evaluator: RAGEvaluator,
        sample_rate: float = 0.1  # Evaluate 10% of queries
    ):
        """
        TODO: Initialize monitor.
        
        Args:
        - evaluator: RAG evaluator
        - sample_rate: % of queries to evaluate (for cost control)
        """
        self.evaluator = evaluator
        self.sample_rate = sample_rate
        self.metrics_history = []
    
    def log_query(
        self,
        query: str,
        retrieved_docs: List[str],
        answer: str,
        latency_ms: float,
        timestamp: datetime
    ):
        """
        TODO: Log production query for monitoring.
        
        Store:
        - Query metadata
        - Retrieved documents
        - Generated answer
        - Performance metrics
        - Timestamp
        
        Optionally run evaluation on sampled queries.
        """
        pass
    
    def calculate_running_metrics(
        self,
        window_size: int = 1000  # Last N queries
    ) -> Dict[str, Any]:
        """
        TODO: Calculate metrics on recent queries.
        
        Track:
        - Average latency (p50, p95, p99)
        - Token usage and costs
        - Error rates
        - Answer lengths
        - Source diversity
        
        If evaluation is enabled:
        - Quality scores on sampled queries
        
        Return current metrics.
        """
        pass
    
    def detect_anomalies(
        self,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        TODO: Detect metric anomalies.
        
        Compare current metrics to baseline:
        - Flag if metric differs by > threshold_std standard deviations
        - Detect sudden changes (e.g., latency spike)
        - Identify degrading trends
        
        Return list of anomalies detected.
        """
        pass
    
    def generate_quality_report(
        self,
        time_period: str = "7d"  # Last 7 days
    ) -> Dict[str, Any]:
        """
        TODO: Generate periodic quality report.
        
        Include:
        - Summary metrics
        - Trends over time (charts if possible)
        - Comparison with previous period
        - Anomalies detected
        - Sample failure cases
        - Recommendations for improvement
        
        Return comprehensive report.
        """
        pass
    
    def alert_on_degradation(
        self,
        metric: str,
        threshold: float,
        notification_callback: callable
    ):
        """
        TODO: Set up alerts for quality degradation.
        
        Monitor specified metric and trigger alert if:
        - Metric drops below threshold
        - Metric shows significant downward trend
        - Anomaly detected
        
        Call notification_callback with alert details.
        """
        pass

# TODO: Implement production monitoring
# TODO: Simulate production queries and log them
# TODO: Generate quality report
# TODO: Test anomaly detection
```

## Part 7: Reflection Questions

### Conceptual Understanding

1. **Metrics Selection**: Which metrics are most important for RAG systems? Why is faithfulness critical?

2. **Trade-offs**: How do retrieval and generation quality trade off against latency? What compromises would you make?

3. **Statistical Significance**: Why is statistical testing important when comparing RAG systems? When might improvements not be significant?

4. **LLM-as-Judge**: What are the limitations of using LLMs to evaluate LLM outputs? How can you validate automated metrics?

### Implementation Insights

5. **Metric Correlations**: Did you find correlations between retrieval and generation metrics? Does better retrieval always lead to better generation?

6. **Test Dataset Quality**: How did you ensure your test dataset was high-quality? What makes a good RAG test case?

7. **Failure Patterns**: What patterns did you observe in failure cases? What types of queries were hardest?

### Production Considerations

8. **Monitoring Strategy**: How would you monitor RAG quality in production with millions of queries? What would you measure and how often?

9. **Cost vs Quality**: Evaluation is expensive (especially LLM-as-judge). How would you balance evaluation thoroughness with cost?

10. **Continuous Improvement**: How would you use evaluation results to continuously improve your RAG system? What feedback loops would you implement?

## Deliverables

1. **Evaluation Framework**: Complete implementation of all metrics
2. **Test Dataset**: 50+ high-quality test cases covering diverse queries
3. **Evaluation Report**: 5-6 pages covering:
   - Retrieval evaluation results
   - Generation evaluation results
   - End-to-end system quality
   - A/B test results (if comparing systems)
   - Recommendations for improvement
4. **Interactive Dashboard**: Notebook visualizing metrics and trends

## Evaluation Rubric

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|-------------------|------------------|
| **Retrieval Metrics** | All metrics correct, well-tested | Most metrics working | Basic metrics only | Incorrect or missing |
| **Generation Metrics** | LLM-as-judge working, validated | Basic metrics working | Simple checks only | Not implemented |
| **Test Dataset** | 50+ diverse, high-quality cases | 30+ reasonable cases | 10+ basic cases | Poor quality |
| **Analysis** | Deep insights, actionable recommendations | Good analysis | Surface-level | Minimal |
| **Code Quality** | Production-ready, well-documented | Clean, some docs | Functional | Poor quality |

## Additional Resources

- [RAGAS: RAG Assessment Framework](https://github.com/explodinggradients/ragas)
- [TruLens for LLM Evaluation](https://www.trulens.org/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [OpenAI Evals Framework](https://github.com/openai/evals)
- Research: "BLEU: a Method for Automatic Evaluation" (Papineni et al., 2002)
- Research: "Large Language Models are not Fair Evaluators" (Wang et al., 2023)

## Hints

1. Start with simple retrieval metrics before tackling generation
2. LLM-as-judge is powerful but can be biased - validate against human judgments
3. Cache LLM evaluations - they're expensive to recompute
4. Use temperature=0 for LLM judges for consistency
5. Bootstrap confidence intervals require 1000+ iterations for stability
6. Create varied test cases - don't just test easy queries
7. Monitor trends over time, not just point-in-time metrics
8. Balance automated metrics with human review of sample cases
9. Document your evaluation methodology for reproducibility
10. Build evaluation into your development workflow - run it frequently

---

**Estimated Time:** 90 minutes  
**Difficulty:** Advanced  
**Topics:** Evaluation metrics, statistical testing, LLM-as-judge, continuous monitoring

Excellent work! Remember: robust evaluation is what separates research prototypes from production systems. Measure rigorously, improve continuously, and never trust a system you haven't thoroughly evaluated.
