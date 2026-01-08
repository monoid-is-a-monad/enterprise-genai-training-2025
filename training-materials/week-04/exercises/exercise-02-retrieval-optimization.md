# Exercise 2: Retrieval Optimization

**Week 4 - RAG Fundamentals**

## Overview

Retrieval is the heart of any RAG system. Even with perfect generation, if you retrieve irrelevant or incomplete context, the system will fail. This exercise focuses on optimizing retrieval through hybrid search, reranking, query expansion, and metadata filtering to build production-quality retrieval systems.

**Time:** 90 minutes  
**Difficulty:** Advanced

## Learning Objectives

By completing this exercise, you will:
- Implement hybrid search combining semantic and keyword methods
- Build reranking systems for improved relevance
- Apply query expansion techniques
- Use metadata filtering effectively
- Measure and optimize retrieval quality
- Handle edge cases and difficult queries

## Prerequisites

- Completed Week 4 Labs 1-2 and Exercise 1
- Understanding of embeddings and vector search
- Familiarity with information retrieval concepts
- OpenAI API access

## Setup

```python
import openai
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import re

client = openai.OpenAI()
```

## Part 1: Baseline Vector Search

### Task 1.1: Implement Basic Vector Search

Start with a baseline semantic search to compare against optimizations.

```python
@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int

class BaselineVectorSearch:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        TODO: Initialize baseline search system.
        """
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        TODO: Index chunks for search.
        
        Args:
        - chunks: List of {chunk_id, content, metadata}
        
        Steps:
        1. Store chunks
        2. Generate embeddings for all chunks (batch for efficiency)
        3. Store embeddings as numpy array
        """
        pass
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Perform semantic search.
        
        Steps:
        1. Generate query embedding
        2. Calculate cosine similarity with all chunk embeddings
        3. Get top-k highest scores
        4. Return SearchResult objects with ranks
        """
        pass
    
    def evaluate_recall(
        self,
        test_queries: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, float]:
        """
        TODO: Evaluate retrieval quality.
        
        Args:
        - test_queries: List of {query, relevant_chunk_ids}
        - k: Number of results to retrieve
        
        Calculate:
        - recall@k: % of relevant docs retrieved
        - precision@k: % of retrieved docs that are relevant
        - MRR: Mean Reciprocal Rank
        - F1@k: Harmonic mean of precision and recall
        
        Return dictionary of metrics.
        """
        pass

# Create test dataset
test_chunks = [
    {
        "chunk_id": "ml_001",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "metadata": {"topic": "ml_basics", "difficulty": "beginner"}
    },
    {
        "chunk_id": "ml_002",
        "content": "Supervised learning uses labeled training data to learn input-output mappings.",
        "metadata": {"topic": "supervised", "difficulty": "beginner"}
    },
    {
        "chunk_id": "ml_003",
        "content": "Neural networks consist of interconnected layers of nodes that process information.",
        "metadata": {"topic": "neural_nets", "difficulty": "intermediate"}
    },
    {
        "chunk_id": "dl_001",
        "content": "Deep learning uses neural networks with many layers to learn hierarchical representations.",
        "metadata": {"topic": "deep_learning", "difficulty": "intermediate"}
    },
    {
        "chunk_id": "dl_002",
        "content": "Convolutional neural networks (CNNs) are specialized for processing grid-like data such as images.",
        "metadata": {"topic": "computer_vision", "difficulty": "advanced"}
    },
    {
        "chunk_id": "dl_003",
        "content": "Recurrent neural networks (RNNs) process sequential data by maintaining hidden state across time steps.",
        "metadata": {"topic": "sequence_modeling", "difficulty": "advanced"}
    },
    {
        "chunk_id": "nlp_001",
        "content": "Transformers use self-attention mechanisms to process sequences in parallel, revolutionizing NLP.",
        "metadata": {"topic": "transformers", "difficulty": "advanced"}
    },
    {
        "chunk_id": "nlp_002",
        "content": "Word embeddings like Word2Vec and GloVe represent words as dense vectors capturing semantic similarity.",
        "metadata": {"topic": "embeddings", "difficulty": "intermediate"}
    },
]

test_queries = [
    {
        "query": "What is machine learning?",
        "relevant_chunk_ids": ["ml_001", "ml_002"]
    },
    {
        "query": "How do neural networks work?",
        "relevant_chunk_ids": ["ml_003", "dl_001"]
    },
    {
        "query": "Tell me about transformers",
        "relevant_chunk_ids": ["nlp_001"]
    },
]

# TODO: Implement baseline search
# TODO: Run evaluation on test queries
# TODO: Record baseline metrics for comparison
```

## Part 2: Hybrid Search

### Task 2.1: Implement Keyword Search (BM25)

Add keyword-based search to complement semantic search.

```python
class KeywordSearch:
    def __init__(self):
        """
        TODO: Initialize keyword search with TF-IDF.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # unigrams and bigrams
            max_features=5000
        )
        self.tfidf_matrix = None
        self.chunks = []
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        TODO: Build TF-IDF index for chunks.
        
        Steps:
        1. Extract content from all chunks
        2. Fit TF-IDF vectorizer
        3. Transform chunks to TF-IDF vectors
        4. Store matrix and chunks
        """
        pass
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        TODO: Perform keyword search using TF-IDF similarity.
        
        Steps:
        1. Transform query to TF-IDF vector
        2. Calculate cosine similarity with all documents
        3. Return top-k results as (chunk_id, score) tuples
        """
        pass

### Task 2.2: Combine Semantic and Keyword Search

class HybridSearch:
    def __init__(
        self,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        TODO: Initialize hybrid search combining both methods.
        
        Args:
        - semantic_weight: Weight for semantic similarity (0-1)
        - keyword_weight: Weight for keyword similarity (0-1)
        
        Note: Weights should sum to 1.0
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        self.semantic_search = BaselineVectorSearch(embedding_model)
        self.keyword_search = KeywordSearch()
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        TODO: Index chunks in both search systems.
        """
        pass
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        return_explanations: bool = False
    ) -> List[SearchResult]:
        """
        TODO: Perform hybrid search.
        
        Algorithm:
        1. Get top-2k results from semantic search
        2. Get top-2k results from keyword search
        3. Normalize scores to [0, 1] range (min-max scaling)
        4. For each unique chunk_id appearing in either result set:
           - Calculate hybrid_score = semantic_weight * semantic_score + 
                                     keyword_weight * keyword_score
           - If chunk only in one result set, use weight * score
        5. Sort by hybrid_score
        6. Return top-k results
        
        If return_explanations=True:
        - Include semantic_score, keyword_score in metadata
        - Include explanation of why chunk was retrieved
        """
        pass
    
    def explain_retrieval(self, query: str, chunk_id: str) -> Dict[str, Any]:
        """
        TODO: Explain why a specific chunk was retrieved.
        
        Return:
        - semantic_score: How semantically similar
        - keyword_score: Keyword match strength
        - matched_keywords: List of overlapping keywords
        - semantic_factors: Key semantic similarities
        - combined_score: Final hybrid score
        """
        pass

# TODO: Implement hybrid search
# TODO: Compare with baseline on test queries
# TODO: Experiment with different weight combinations (0.5/0.5, 0.7/0.3, 0.9/0.1)
# TODO: Identify which types of queries benefit from keyword vs semantic
```

## Part 3: Query Expansion

### Task 3.1: Implement Query Rewriting

Expand queries to improve retrieval coverage.

```python
class QueryExpander:
    def __init__(self, client: openai.OpenAI):
        """
        TODO: Initialize query expander.
        """
        self.client = client
    
    def generate_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        TODO: Generate query variations using LLM.
        
        Prompt the LLM to:
        - Rephrase the query in different ways
        - Add synonyms
        - Expand abbreviations
        - Make implicit information explicit
        
        Example:
        Query: "What is ML?"
        Variations:
        - "What is machine learning?"
        - "Explain machine learning concepts"
        - "Define ML and its applications"
        
        Return list of variations (including original query).
        """
        pass
    
    def expand_with_context(self, query: str, context: str) -> str:
        """
        TODO: Expand query using conversation context.
        
        For follow-up questions like "How does it work?", use context
        to make the query standalone.
        
        Example:
        Context: "Previous query about transformers"
        Query: "How do they work?"
        Expanded: "How do transformers work in NLP?"
        """
        pass
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        TODO: Extract important keywords from query.
        
        Steps:
        1. Use LLM to identify key terms
        2. Remove stop words
        3. Identify technical terms and acronyms
        4. Return ranked list of keywords
        """
        pass

class MultiQueryRetrieval:
    def __init__(self, retrieval_system: HybridSearch, expander: QueryExpander):
        """
        TODO: Initialize multi-query retrieval.
        """
        self.retrieval_system = retrieval_system
        self.expander = expander
    
    def retrieve_with_expansion(
        self,
        query: str,
        top_k: int = 5,
        num_variations: int = 3
    ) -> List[SearchResult]:
        """
        TODO: Retrieve using multiple query variations.
        
        Algorithm:
        1. Generate query variations
        2. Retrieve top-k results for each variation
        3. Aggregate results using Reciprocal Rank Fusion (RRF):
           - For each result, calculate RRF score = sum(1 / (rank + 60))
           - Rank is position in each query's results (1-indexed)
           - Constant 60 is typical RRF parameter
        4. Sort by RRF score
        5. Return top-k unique results
        
        Benefits:
        - Improves recall (finds more relevant docs)
        - Reduces sensitivity to query phrasing
        - More robust to ambiguous queries
        """
        pass
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[SearchResult]],
        k: int = 60
    ) -> List[SearchResult]:
        """
        TODO: Implement Reciprocal Rank Fusion.
        
        For each unique chunk_id across all result lists:
        - RRF_score = sum over all lists: 1 / (rank_in_list + k)
        - Higher score = more consensus across queries
        """
        pass

# TODO: Test query expansion on ambiguous queries
# TODO: Compare recall with and without expansion
# TODO: Measure latency impact of multiple retrievals
```

## Part 4: Reranking

### Task 4.1: Implement Cross-Encoder Reranking

Add a reranking layer to improve result quality.

```python
class Reranker:
    def __init__(self, client: openai.OpenAI):
        """
        TODO: Initialize reranker.
        
        Note: In production, use dedicated cross-encoder models
        (sentence-transformers/ms-marco-MiniLM-L-12-v2) for better
        performance. For this exercise, use LLM-based reranking.
        """
        self.client = client
    
    def rerank_with_llm(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        TODO: Rerank results using LLM to assess relevance.
        
        For each result:
        1. Create prompt asking LLM to score relevance (0-10)
        2. Prompt: "Rate how relevant this passage is to the query.
                    Query: {query}
                    Passage: {content}
                    Relevance (0-10):"
        3. Parse score from response
        4. Sort by scores
        5. Return top-k
        
        Challenges:
        - Expensive (1 API call per result)
        - Slower than vector similarity
        - More accurate for complex queries
        
        Production alternatives:
        - Use cross-encoder models (much faster)
        - Batch scoring requests
        - Cache reranking scores
        """
        pass
    
    def rerank_with_features(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        TODO: Rerank using multiple features.
        
        Calculate composite score from:
        1. Original retrieval score (0.4 weight)
        2. Query term coverage: % of query terms in result (0.2 weight)
        3. Result length score: Penalize very short/long (0.1 weight)
        4. Metadata match: Boost if metadata aligns (0.2 weight)
        5. Recency: If timestamps available, boost recent (0.1 weight)
        
        Normalize each feature to [0, 1], apply weights, sum.
        Sort by composite score.
        """
        pass

class EnhancedRetrievalSystem:
    """
    TODO: Production retrieval combining all optimizations.
    """
    
    def __init__(
        self,
        hybrid_search: HybridSearch,
        query_expander: QueryExpander,
        reranker: Reranker
    ):
        self.hybrid_search = hybrid_search
        self.query_expander = query_expander
        self.reranker = reranker
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_expansion: bool = True,
        use_reranking: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        TODO: Full retrieval pipeline.
        
        Steps:
        1. Optionally expand query
        2. Retrieve top-2k candidates with hybrid search
        3. Apply metadata filters if specified
        4. Optionally rerank to top-k
        5. Return results with metadata:
           - results: List[SearchResult]
           - query_used: Original or expanded query
           - expansion_applied: bool
           - reranking_applied: bool
           - total_candidates: Number before filtering
           - execution_time: Pipeline latency
        """
        pass

# TODO: Implement full pipeline
# TODO: Run ablation study (turn features on/off)
# TODO: Measure impact of each component on quality and speed
```

## Part 5: Metadata Filtering

### Task 5.1: Implement Advanced Filtering

```python
class MetadataFilter:
    """
    TODO: Flexible metadata filtering system.
    """
    
    @staticmethod
    def apply_filter(
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        TODO: Apply metadata filters to results.
        
        Support filters:
        - Exact match: {"topic": "ml_basics"}
        - Range: {"difficulty_level": {"$gte": 2, "$lte": 4}}
        - List membership: {"category": {"$in": ["ml", "dl"]}}
        - Exists: {"author": {"$exists": True}}
        - Logical: {"$and": [...], "$or": [...], "$not": {...}}
        
        Examples:
        1. filter = {"topic": "ml_basics"}
           → Only results where metadata['topic'] == 'ml_basics'
        
        2. filter = {"difficulty": {"$in": ["beginner", "intermediate"]}}
           → Only beginner or intermediate results
        
        3. filter = {
               "$and": [
                   {"topic": "deep_learning"},
                   {"year": {"$gte": 2020}}
               ]
           }
           → Results matching both conditions
        """
        pass
    
    @staticmethod
    def validate_filter(filters: Dict[str, Any]) -> bool:
        """
        TODO: Validate filter syntax.
        
        Check:
        - Valid operators
        - Correct structure
        - No conflicting conditions
        
        Return True if valid, False otherwise.
        """
        pass

# Test cases
test_filters = [
    {"topic": "ml_basics"},
    {"difficulty": {"$in": ["beginner", "intermediate"]}},
    {
        "$and": [
            {"topic": "deep_learning"},
            {"difficulty": "advanced"}
        ]
    },
    {
        "$or": [
            {"topic": "transformers"},
            {"topic": "embeddings"}
        ]
    }
]

# TODO: Test each filter on test_chunks
# TODO: Verify filtering preserves ranking order
# TODO: Measure performance impact of complex filters
```

## Part 6: Evaluation & Optimization

### Task 6.1: Comprehensive Retrieval Evaluation

```python
class RetrievalEvaluator:
    def __init__(self, test_dataset: List[Dict[str, Any]]):
        """
        TODO: Initialize evaluator with test dataset.
        
        Test dataset format:
        [
            {
                "query": str,
                "relevant_chunk_ids": List[str],
                "difficulty": str,  # easy/medium/hard
                "query_type": str  # factual/conceptual/procedural
            },
            ...
        ]
        """
        self.test_dataset = test_dataset
    
    def evaluate_system(
        self,
        retrieval_system,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        TODO: Comprehensive evaluation across multiple metrics.
        
        For each k in k_values, calculate:
        
        1. RECALL@K:
           recall = (relevant docs in top-k) / (total relevant docs)
           Average across all queries
        
        2. PRECISION@K:
           precision = (relevant docs in top-k) / k
           Average across all queries
        
        3. MRR (Mean Reciprocal Rank):
           For each query: 1 / rank of first relevant doc
           Average across all queries
        
        4. NDCG@K (Normalized Discounted Cumulative Gain):
           Accounts for ranking quality
           NDCG = DCG / IDCG
           where DCG = sum(relevance_i / log2(rank_i + 1))
        
        5. Success@K:
           % of queries with at least 1 relevant result in top-k
        
        Also calculate by:
        - Query difficulty (easy/medium/hard)
        - Query type (factual/conceptual/procedural)
        
        Return comprehensive metrics dictionary.
        """
        pass
    
    def calculate_dcg(self, relevances: List[int], k: int) -> float:
        """
        TODO: Calculate Discounted Cumulative Gain.
        
        relevances: Binary list [1, 0, 1, 0, ...] indicating relevant docs
        Formula: sum(rel_i / log2(i + 1)) for i=1 to k
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
        - systems: Dict of {name: retrieval_system}
        - k: Number of results to evaluate
        
        Return DataFrame showing metrics for each system side-by-side.
        """
        pass

### Task 6.2: Query Analysis

def analyze_query_patterns(
    retrieval_system,
    queries: List[str],
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    TODO: Analyze which query patterns work well/poorly.
    
    For each query:
    1. Classify query type (factual, conceptual, procedural, etc.)
    2. Measure query complexity (length, technical terms, ambiguity)
    3. Retrieve results
    4. Assess retrieval quality
    
    Group results by query characteristics and find patterns:
    - Do short queries perform worse?
    - Do technical queries need keyword boost?
    - Are conceptual queries helped by expansion?
    
    Return insights dictionary with recommendations.
    """
    pass

# TODO: Run full evaluation
# TODO: Compare baseline vs hybrid vs hybrid+expansion vs full pipeline
# TODO: Generate performance report
```

## Part 7: Advanced Topics

### Task 7.1: Dense-Sparse Fusion (Optional)

For advanced learners, implement fusion of dense (embeddings) and sparse (BM25) retrievals at the index level rather than score level.

```python
class DenseSparseIndexer:
    """
    TODO: Build inverted index for sparse retrieval alongside dense vectors.
    
    Concept: Maintain both representations and fuse at retrieval time
    for better efficiency than post-hoc score combination.
    """
    pass
```

## Part 8: Reflection Questions

### Conceptual Understanding

1. **Hybrid Search**: Why does combining semantic and keyword search often outperform either alone? What are the failure modes of pure semantic search?

2. **Query Expansion**: In what scenarios does query expansion help most? When might it hurt performance?

3. **Reranking**: Why is two-stage retrieval (retrieve many, rerank to few) better than one-stage? What are the cost/latency trade-offs?

4. **Metadata Filtering**: How does filtering interact with relevance ranking? Should you filter before or after retrieval?

### Implementation Insights

5. **Performance**: Which optimization had the biggest impact on retrieval quality? Which had the biggest computational cost?

6. **Weight Tuning**: How did you determine optimal weights for hybrid search? Did different query types need different weights?

7. **Edge Cases**: What queries performed poorly even with optimizations? Why? How could you improve them?

### Production Considerations

8. **Latency**: How would you reduce retrieval latency while maintaining quality? What would you cache?

9. **Scalability**: How would these optimizations scale to millions of documents? What bottlenecks would emerge?

10. **Monitoring**: What metrics would you track in production to detect retrieval degradation? How would you A/B test improvements?

## Deliverables

1. **Implementation**: All retrieval components fully functional
2. **Evaluation Report**: Comprehensive metrics comparing approaches
3. **Analysis Document**: 3-4 pages covering:
   - Experimental results
   - Query pattern analysis
   - Optimization recommendations
   - Answers to reflection questions
4. **Demo**: Interactive notebook showing optimizations in action

## Evaluation Rubric

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|-------------------|------------------|
| **Hybrid Search** | Fully working with flexible weighting | Works with fixed weights | Basic implementation | Incomplete |
| **Query Expansion** | Multiple strategies, well-integrated | Single strategy working | Basic expansion | Not functional |
| **Reranking** | Multiple methods, efficient | One method working | Basic reranking | Missing |
| **Evaluation** | Comprehensive metrics, deep insights | Good metrics, analysis | Basic evaluation | Superficial |
| **Code Quality** | Production-ready, well-documented | Clean, some docs | Functional but messy | Poor quality |

## Additional Resources

- [Pinecone: Hybrid Search Explained](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)
- [LangChain Multi-Query Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
- [Sentence-Transformers Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- Research: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE, Gao et al., 2022)

## Hints

1. Start with small test dataset to iterate quickly
2. Measure baseline performance first - gives you target to beat
3. BM25 handles exact matches well; use for keyword-heavy queries
4. Query expansion helps with ambiguous queries but adds latency
5. Cross-encoders are best rerankers but slow; use for top-20 → top-5
6. Cache everything you can: embeddings, query expansions, reranking scores
7. Don't over-optimize on test set - validate on held-out queries
8. Monitor latency closely - each optimization adds overhead

---

**Estimated Time:** 90 minutes  
**Difficulty:** Advanced  
**Topics:** Information retrieval, hybrid search, reranking, query processing

Happy optimizing! Remember: retrieval quality is often more important than generation quality in RAG systems - you can't generate good answers from bad context.
