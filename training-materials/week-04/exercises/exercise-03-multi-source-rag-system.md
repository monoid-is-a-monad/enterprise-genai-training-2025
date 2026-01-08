# Exercise 3: Multi-Source RAG System

**Week 4 - RAG Fundamentals**

## Overview

Real-world RAG systems rarely work with a single data source. This exercise focuses on building a RAG system that intelligently retrieves from multiple heterogeneous sources (documents, databases, APIs, knowledge graphs) and combines information coherently. You'll learn to handle source prioritization, conflict resolution, and multi-source attribution.

**Time:** 120 minutes  
**Difficulty:** Advanced

## Learning Objectives

By completing this exercise, you will:
- Integrate multiple data sources into a unified RAG system
- Implement source-aware retrieval and ranking
- Handle conflicting information from different sources
- Build proper attribution and citation systems
- Optimize retrieval across heterogeneous sources
- Design scalable multi-source architectures

## Prerequisites

- Completed Week 4 Labs 1-3
- Completed Exercises 1-2
- Understanding of database queries and APIs
- OpenAI API access

## Setup

```python
import openai
import tiktoken
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

client = openai.OpenAI()
```

## Part 1: Multi-Source Data Model

### Task 1.1: Design Source-Aware Data Structures

```python
class SourceType(Enum):
    """Types of data sources."""
    DOCUMENT = "document"
    DATABASE = "database"
    API = "api"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WEB = "web"

@dataclass
class Source:
    """Metadata about a data source."""
    source_id: str
    source_type: SourceType
    name: str
    authority_score: float  # 0-1, how trustworthy
    recency_weight: float   # 0-1, how much to prioritize recent content
    access_latency_ms: float  # Expected retrieval latency
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """TODO: Convert source to dictionary."""
        pass

@dataclass
class SourcedChunk:
    """Chunk with source attribution."""
    chunk_id: str
    content: str
    source: Source
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def get_authority_score(self) -> float:
        """
        TODO: Calculate chunk authority based on source and metadata.
        
        Factors:
        - Source authority_score (60% weight)
        - Recency if timestamp available (20% weight)
        - Metadata quality indicators (20% weight)
        
        Return combined score 0-1.
        """
        pass
    
    def get_citation(self) -> str:
        """
        TODO: Generate human-readable citation.
        
        Examples:
        - Document: "From ML Guide (page 15)"
        - Database: "From products_db.inventory (updated 2024-01-15)"
        - API: "From WeatherAPI (retrieved 2024-01-20)"
        """
        pass

# TODO: Create sample sources
sources = [
    Source(
        source_id="doc_ml_guide",
        source_type=SourceType.DOCUMENT,
        name="Machine Learning Textbook",
        authority_score=0.9,
        recency_weight=0.3,
        access_latency_ms=10,
        metadata={"author": "Expert Author", "year": 2023}
    ),
    Source(
        source_id="web_wiki",
        source_type=SourceType.WEB,
        name="Wikipedia",
        authority_score=0.7,
        recency_weight=0.6,
        access_latency_ms=200,
        metadata={"domain": "wikipedia.org"}
    ),
    Source(
        source_id="db_research",
        source_type=SourceType.DATABASE,
        name="Research Papers DB",
        authority_score=0.95,
        recency_weight=0.8,
        access_latency_ms=50,
        metadata={"database": "arxiv"}
    ),
]

# TODO: Create sample chunks from each source
# TODO: Verify authority scores and citations work correctly
```

### Task 1.2: Source Connectors

Implement connectors for different source types.

```python
from abc import ABC, abstractmethod

class SourceConnector(ABC):
    """Base class for source connectors."""
    
    def __init__(self, source: Source):
        self.source = source
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SourcedChunk]:
        """Retrieve chunks from this source."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if source is accessible."""
        pass

class DocumentConnector(SourceConnector):
    """Connector for document-based sources."""
    
    def __init__(self, source: Source, chunks: List[SourcedChunk]):
        """
        TODO: Initialize document connector.
        
        Args:
        - source: Source metadata
        - chunks: Pre-indexed chunks from documents
        """
        super().__init__(source)
        self.chunks = chunks
        self.embeddings = None
    
    def index_chunks(self):
        """
        TODO: Build embedding index for chunks.
        
        Use batch embedding for efficiency.
        Store embeddings in numpy array.
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SourcedChunk]:
        """
        TODO: Retrieve from document source using semantic search.
        """
        pass
    
    def is_available(self) -> bool:
        """TODO: Check if chunks are loaded."""
        pass

class DatabaseConnector(SourceConnector):
    """Connector for structured database sources."""
    
    def __init__(self, source: Source, connection_string: str = None):
        """
        TODO: Initialize database connector.
        
        In a real implementation, this would connect to actual DB.
        For exercise, simulate with in-memory data.
        """
        super().__init__(source)
        # Simulate database records
        self.records = []
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SourcedChunk]:
        """
        TODO: Retrieve from database.
        
        Steps:
        1. Parse query to extract database-relevant terms
        2. Convert query to SQL-like filter (simplified)
        3. Retrieve matching records
        4. Convert records to SourcedChunk objects
        5. Rank by relevance
        
        For exercise: Use keyword matching on records.
        In production: Generate SQL queries or use hybrid search.
        """
        pass
    
    def is_available(self) -> bool:
        """TODO: Check database connection."""
        pass

class APIConnector(SourceConnector):
    """Connector for API-based sources."""
    
    def __init__(self, source: Source, api_endpoint: str = None):
        """
        TODO: Initialize API connector.
        
        For exercise: Simulate API responses.
        In production: Make actual API calls.
        """
        super().__init__(source)
        self.api_endpoint = api_endpoint
        self.cache = {}  # Response cache
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SourcedChunk]:
        """
        TODO: Retrieve from API source.
        
        Steps:
        1. Check cache for recent responses
        2. If not cached, make API call (simulated)
        3. Parse API response
        4. Convert to SourcedChunk objects
        5. Cache results
        6. Return top-k
        
        Handle:
        - API rate limits
        - Timeout errors
        - Parse errors
        """
        pass
    
    def is_available(self) -> bool:
        """TODO: Check API health with lightweight request."""
        pass

# TODO: Implement all connectors
# TODO: Test each connector independently
# TODO: Measure retrieval latency for each source type
```

## Part 2: Multi-Source Retrieval

### Task 2.1: Unified Retrieval Interface

```python
class MultiSourceRetriever:
    """Retrieve from multiple sources with intelligent routing."""
    
    def __init__(self):
        """TODO: Initialize multi-source retriever."""
        self.connectors: Dict[str, SourceConnector] = {}
        self.source_priorities: Dict[str, float] = {}
    
    def register_connector(
        self,
        connector: SourceConnector,
        priority: float = 1.0
    ):
        """
        TODO: Register a source connector.
        
        Args:
        - connector: SourceConnector instance
        - priority: Base priority for this source (0-1)
        """
        pass
    
    def retrieve_parallel(
        self,
        query: str,
        top_k_per_source: int = 10,
        total_top_k: int = 5,
        timeout_seconds: float = 2.0
    ) -> List[Tuple[SourcedChunk, float]]:
        """
        TODO: Retrieve from all sources in parallel.
        
        Algorithm:
        1. Query all connectors concurrently (ThreadPoolExecutor)
        2. Apply timeout to prevent slow sources from blocking
        3. Collect results from available sources
        4. Score each result combining:
           - Retrieval relevance score
           - Source authority
           - Source priority
           - Recency (if applicable)
        5. Deduplicate similar results
        6. Sort by combined score
        7. Return top-k overall
        
        Scoring formula:
        combined_score = (
            0.4 * relevance_score +
            0.3 * source.authority_score +
            0.2 * source_priority +
            0.1 * recency_score
        )
        
        Handle:
        - Sources that timeout
        - Sources that error
        - Empty result sets
        """
        pass
    
    def retrieve_sequential(
        self,
        query: str,
        source_order: List[str],
        top_k: int = 5,
        early_stopping: bool = True,
        confidence_threshold: float = 0.8
    ) -> List[Tuple[SourcedChunk, float]]:
        """
        TODO: Retrieve from sources sequentially.
        
        Use when:
        - Sources have strong priority hierarchy
        - Want to minimize latency (stop when confident)
        - Budget for API calls is limited
        
        Algorithm:
        1. Query sources in specified order
        2. Collect results
        3. If early_stopping enabled and top result score > threshold:
           - Stop querying remaining sources
        4. Otherwise continue through all sources
        5. Combine and rank results
        
        Return results with source attribution.
        """
        pass
    
    def retrieve_adaptive(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[SourcedChunk, float]]:
        """
        TODO: Adaptively select sources based on query.
        
        Intelligence:
        1. Analyze query to determine type:
           - Factual: Prioritize databases, authoritative docs
           - Conceptual: Prioritize documents, knowledge bases
           - Current events: Prioritize APIs, web sources
           - Technical: Prioritize documentation, research papers
        
        2. Select subset of sources most likely to have relevant info
        
        3. Retrieve only from selected sources
        
        Benefits:
        - Reduced latency (fewer sources queried)
        - Lower costs (fewer API calls)
        - Better relevance (right source for query type)
        """
        pass
    
    def classify_query_type(self, query: str) -> Dict[str, float]:
        """
        TODO: Classify query to determine source priorities.
        
        Use LLM or heuristics to classify as:
        - factual: Needs precise facts/numbers
        - conceptual: Needs explanations/understanding
        - procedural: Needs how-to/steps
        - current: Needs recent/real-time data
        
        Return confidence scores for each type.
        """
        pass

# TODO: Implement multi-source retriever
# TODO: Test all retrieval modes (parallel, sequential, adaptive)
# TODO: Compare latency and quality across modes
```

### Task 2.2: Deduplication Across Sources

```python
class CrossSourceDeduplicator:
    """Deduplicate similar content from different sources."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        TODO: Initialize deduplicator.
        
        Args:
        - similarity_threshold: Cosine similarity above which
          chunks are considered duplicates
        """
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(
        self,
        chunks: List[SourcedChunk]
    ) -> List[SourcedChunk]:
        """
        TODO: Remove near-duplicate chunks, keeping best sources.
        
        Algorithm:
        1. Calculate embeddings for all chunks (if not cached)
        2. Compute pairwise cosine similarities
        3. Build similarity graph:
           - Nodes = chunks
           - Edges = similarity > threshold
        4. For each connected component (duplicate group):
           - Keep chunk from source with highest authority
           - If tied, keep most recent
        5. Return deduplicated list
        
        Preserve:
        - Original ranking order
        - At least one chunk from each truly unique piece of info
        
        Edge cases:
        - Multiple chunks from same source (keep all)
        - Partial overlaps (may need to keep both)
        """
        pass
    
    def find_conflicting_information(
        self,
        chunks: List[SourcedChunk]
    ) -> List[Dict[str, Any]]:
        """
        TODO: Identify chunks with conflicting information.
        
        Use LLM to detect:
        - Same topic, different facts
        - Contradictory statements
        - Different values for same quantity
        
        Return list of conflict groups:
        [
            {
                "topic": "identified topic",
                "chunks": [chunk1, chunk2],
                "conflict_type": "contradictory_facts",
                "resolution": "prefer_higher_authority"  # or "needs_human_review"
            },
            ...
        ]
        
        This enables:
        - Warning users about conflicts
        - Presenting multiple viewpoints
        - Requiring human review for critical decisions
        """
        pass

# TODO: Test deduplication on overlapping content
# TODO: Test conflict detection on contradictory sources
# TODO: Measure precision/recall of duplicate detection
```

## Part 3: Source Fusion & Attribution

### Task 3.1: Multi-Source Context Assembly

```python
class MultiSourceContextAssembler:
    """Assemble context from multiple sources intelligently."""
    
    def __init__(
        self,
        max_tokens: int = 2000,
        citation_style: str = "inline"  # inline, footnote, or endnote
    ):
        """
        TODO: Initialize context assembler.
        """
        self.max_tokens = max_tokens
        self.citation_style = citation_style
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def assemble_with_citations(
        self,
        chunks: List[Tuple[SourcedChunk, float]],
        query: str
    ) -> Dict[str, Any]:
        """
        TODO: Assemble context with proper citations.
        
        Steps:
        1. Sort chunks by combined score
        2. Add chunks until token limit reached
        3. Insert citations based on citation_style
        4. Group chunks by source if beneficial
        5. Add source metadata section
        
        Citation styles:
        
        INLINE:
        "Machine learning is... [1]
         Deep learning uses... [2]"
        
        FOOTNOTE:
        "Machine learning is...*
         Deep learning uses...**
         
         *From ML Textbook, page 5
         **From Research Papers DB, updated 2024-01-15"
        
        ENDNOTE:
        "Machine learning is...
         Deep learning uses...
         
         Sources:
         [1] ML Textbook (Chapter 2)
         [2] Research Papers DB (arxiv:2024.12345)"
        
        Return:
        {
            "context": str,  # Assembled context with citations
            "sources": List[Source],  # Unique sources used
            "citations": Dict[str, str],  # Citation ID to full reference
            "token_count": int
        }
        """
        pass
    
    def assemble_by_perspective(
        self,
        chunks: List[Tuple[SourcedChunk, float]],
        query: str
    ) -> Dict[str, Any]:
        """
        TODO: Assemble context showing multiple perspectives.
        
        When chunks contain different viewpoints:
        1. Cluster chunks by perspective/viewpoint
        2. Present each perspective with its sources
        3. Note where perspectives agree/disagree
        
        Example output:
        "According to academic sources [1,2]:
         - Machine learning requires large datasets...
         
         Industry practitioners note [3]:
         - In practice, small datasets can work with...
         
         These perspectives differ on data requirements but
         agree on the importance of..."
        
        Return structure showing multiple perspectives clearly.
        """
        pass
    
    def assemble_with_confidence(
        self,
        chunks: List[Tuple[SourcedChunk, float]],
        query: str
    ) -> Dict[str, Any]:
        """
        TODO: Assemble context with confidence indicators.
        
        Annotate information with:
        - HIGH confidence: Multiple authoritative sources agree
        - MEDIUM confidence: One authoritative source or multiple agree
        - LOW confidence: Single low-authority source or sources conflict
        
        Example:
        "Machine learning requires labeled data [HIGH - 3 sources agree].
         The optimal dataset size is debatable [LOW - sources conflict]."
        
        Return context with confidence markers.
        """
        pass

# TODO: Test all assembly methods
# TODO: Compare readability and usefulness
# TODO: Measure token efficiency
```

### Task 3.2: Answer Generation with Attribution

```python
class AttributedAnswerGenerator:
    """Generate answers with proper source attribution."""
    
    def __init__(self, client: openai.OpenAI):
        """TODO: Initialize generator."""
        self.client = client
    
    def generate_with_citations(
        self,
        query: str,
        context_with_citations: str,
        sources: List[Source],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        TODO: Generate answer that properly cites sources.
        
        System prompt should instruct model to:
        - Use information from context
        - Cite sources using provided citation markers
        - Acknowledge when sources disagree
        - Note confidence level based on source agreement
        - Admit when context doesn't fully answer question
        
        Return:
        {
            "answer": str,  # Answer with inline citations
            "sources_cited": List[str],  # Which sources were actually used
            "confidence": float,  # Overall answer confidence 0-1
            "coverage": float,  # How fully query was answered 0-1
            "usage": Dict  # Token usage
        }
        """
        pass
    
    def generate_comparative_answer(
        self,
        query: str,
        source_perspectives: Dict[str, List[SourcedChunk]],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        TODO: Generate answer comparing multiple source perspectives.
        
        When sources disagree:
        1. Present each perspective fairly
        2. Note key differences
        3. Explain possible reasons for disagreement
        4. Provide balanced conclusion if possible
        5. Recommend further investigation if needed
        
        Example answer:
        "Regarding {query}, sources present different views:
         
         Academic research [1,2] emphasizes...
         Industry experience [3] suggests...
         
         These differences likely stem from...
         
         For your use case, consider..."
        """
        pass
    
    def extract_cited_sources(self, answer: str, sources: List[Source]) -> List[str]:
        """
        TODO: Extract which sources were actually cited in answer.
        
        Parse answer for citation markers and map back to source IDs.
        Useful for:
        - Tracking source utilization
        - Validating all claims are supported
        - Analytics on source usage patterns
        """
        pass

# TODO: Test answer generation with various query types
# TODO: Verify citations are accurate and complete
# TODO: Check handling of conflicting sources
```

## Part 4: Complete Multi-Source RAG System

### Task 4.1: End-to-End System

```python
class MultiSourceRAGSystem:
    """Complete RAG system with multi-source support."""
    
    def __init__(
        self,
        client: openai.OpenAI,
        max_context_tokens: int = 2000
    ):
        """
        TODO: Initialize complete system.
        
        Components:
        - Multi-source retriever
        - Deduplicator
        - Context assembler
        - Answer generator
        """
        self.client = client
        self.retriever = MultiSourceRetriever()
        self.deduplicator = CrossSourceDeduplicator()
        self.assembler = MultiSourceContextAssembler(max_context_tokens)
        self.generator = AttributedAnswerGenerator(client)
    
    def add_source(
        self,
        connector: SourceConnector,
        priority: float = 1.0
    ):
        """TODO: Add a data source to the system."""
        pass
    
    def query(
        self,
        question: str,
        retrieval_mode: str = "parallel",  # parallel, sequential, adaptive
        top_k: int = 5,
        enable_deduplication: bool = True,
        citation_style: str = "inline",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        TODO: Complete multi-source RAG query.
        
        Pipeline:
        1. Retrieve from multiple sources
        2. Deduplicate if enabled
        3. Assemble context with citations
        4. Generate answer
        5. Extract performance metrics
        
        Return:
        {
            "answer": str,
            "sources_used": List[Source],
            "sources_cited": List[str],
            "context": str,
            "confidence": float,
            "retrieval_stats": {
                "sources_queried": int,
                "sources_returned_results": int,
                "total_chunks_retrieved": int,
                "chunks_after_dedup": int,
                "retrieval_latency_ms": float
            },
            "generation_stats": {
                "generation_latency_ms": float,
                "tokens_used": Dict
            }
        }
        """
        pass
    
    def query_with_source_selection(
        self,
        question: str,
        allowed_sources: Optional[List[str]] = None,
        required_sources: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        TODO: Query with user-specified source constraints.
        
        Args:
        - allowed_sources: Only use these sources
        - required_sources: Must include these sources
        
        Use case: User wants answer specifically from documentation
        vs. from all available sources.
        """
        pass
    
    def explain_retrieval(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        TODO: Explain retrieval decision-making process.
        
        Return detailed breakdown:
        - Which sources were queried and why
        - How results were scored and ranked
        - Why certain sources were prioritized
        - What was deduplicated
        - Which chunks were selected for context
        
        Useful for:
        - Debugging poor retrieval
        - Understanding system behavior
        - Building user trust through transparency
        """
        pass

# TODO: Build complete system with multiple sources
# TODO: Test end-to-end on complex queries
# TODO: Measure quality vs single-source baseline
```

### Task 4.2: System Evaluation

```python
class MultiSourceEvaluator:
    """Evaluate multi-source RAG system."""
    
    def __init__(self, rag_system: MultiSourceRAGSystem):
        """TODO: Initialize evaluator."""
        self.rag_system = rag_system
    
    def evaluate_source_contribution(
        self,
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        TODO: Measure each source's contribution to answers.
        
        For each source:
        - % of queries where it was queried
        - % of queries where it provided results
        - % of queries where it was cited in answer
        - Average relevance score when used
        - Average authority score
        
        Insights:
        - Which sources are most valuable
        - Which are underutilized
        - Which have low relevance despite high authority
        
        Return analysis by source.
        """
        pass
    
    def evaluate_multi_source_benefit(
        self,
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        TODO: Compare multi-source vs single-source performance.
        
        For each query:
        1. Run with all sources (multi-source)
        2. Run with each source individually
        3. Compare answer quality
        
        Measure:
        - Improvement in answer completeness
        - Improvement in answer accuracy
        - Cases where multi-source resolved ambiguity
        - Cases where it introduced confusion
        
        Return comparison metrics.
        """
        pass
    
    def analyze_source_conflicts(
        self,
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        TODO: Analyze how system handles conflicting sources.
        
        Find queries where:
        - Retrieved chunks contain contradictions
        - Sources provide different answers
        
        Evaluate:
        - Was conflict detected?
        - Was it handled appropriately?
        - Did answer acknowledge disagreement?
        - Was resolution reasonable?
        
        Return conflict analysis.
        """
        pass

# TODO: Run comprehensive evaluation
# TODO: Generate evaluation report
# TODO: Identify optimization opportunities
```

## Part 5: Reflection Questions

### Conceptual Understanding

1. **Source Authority**: How should you weight source authority vs. relevance? When might a less authoritative source be preferred?

2. **Deduplication**: What are the risks of aggressive deduplication? When should you keep near-duplicates?

3. **Conflicting Information**: How should RAG systems handle contradictory information from equally authoritative sources?

4. **Source Selection**: When is adaptive source selection better than querying all sources? What are the trade-offs?

### Implementation Insights

5. **Performance**: What was the latency impact of querying multiple sources? How did parallel vs sequential retrieval compare?

6. **Quality**: Did multi-source retrieval produce better answers than single-source? In what scenarios?

7. **Attribution**: How did you ensure citations were accurate? What challenges did you encounter?

### Production Considerations

8. **Scalability**: How would you scale to 100+ data sources? What architectural changes would be needed?

9. **Monitoring**: What metrics would you track to detect when a source becomes unreliable or outdated?

10. **Cost Management**: How would you optimize costs when some sources charge per query (APIs)?

## Deliverables

1. **Implementation**: Full multi-source RAG system
2. **Demo**: Jupyter notebook with examples showing:
   - Single vs multi-source comparison
   - Conflict resolution
   - Source attribution
3. **Analysis Report**: 4-5 pages covering:
   - Source integration approach
   - Performance evaluation
   - Design decisions and trade-offs
   - Answers to reflection questions
4. **System Documentation**: Architecture diagram and user guide

## Evaluation Rubric

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|-------------------|------------------|
| **Multi-Source Integration** | 3+ source types, seamless integration | 2-3 sources working well | Basic multi-source | Single source only |
| **Deduplication & Conflicts** | Robust handling, conflict detection | Basic dedup working | Minimal handling | Not implemented |
| **Attribution System** | Accurate, multiple citation styles | Single style, mostly accurate | Basic citations | Poor or missing |
| **Performance** | Optimized, parallel retrieval | Decent performance | Functional but slow | Inefficient |
| **Code Quality** | Production-ready, well-documented | Clean, some docs | Functional | Poor quality |

## Additional Resources

- [LangChain Multi-Vector Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
- [Weaviate Multi-Tenancy](https://weaviate.io/developers/weaviate/concepts/multi-tenancy)
- [RAG Fusion Paper](https://github.com/Raudaschl/rag-fusion)
- [Pinecone Namespaces for Multi-Source](https://docs.pinecone.io/docs/namespaces)
- Research: "Attributed Question Answering: Evaluation and Modeling" (Bohnet et al., 2022)

## Hints

1. Start with 2-3 sources before scaling up
2. Use source IDs consistently across all components
3. Cache source metadata to avoid repeated lookups
4. Implement circuit breakers for unreliable sources
5. Test deduplication threshold carefully - too high misses dups, too low over-deduplicates
6. Citation formatting is harder than it looks - test thoroughly
7. Consider using asyncio for truly parallel source queries
8. Monitor source query latencies - identify slow sources early
9. Build admin interface to adjust source priorities without code changes
10. Consider implementing source health checks and automatic disabling

---

**Estimated Time:** 120 minutes  
**Difficulty:** Advanced  
**Topics:** Multi-source integration, information fusion, attribution, conflict resolution

Good luck building your multi-source RAG system! Remember: in the real world, information comes from many places - your RAG system needs to handle that complexity gracefully.
