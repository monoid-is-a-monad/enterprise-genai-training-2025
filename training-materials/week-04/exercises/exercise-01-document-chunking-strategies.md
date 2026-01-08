# Exercise 1: Document Chunking Strategies

**Week 4 - RAG Fundamentals**

## Overview

Document chunking is one of the most critical decisions in RAG system design. The chunking strategy directly impacts retrieval quality, context coherence, and generation accuracy. In this exercise, you'll implement and compare multiple chunking strategies to understand their trade-offs and when to apply each.

**Time:** 90 minutes  
**Difficulty:** Intermediate

## Learning Objectives

By completing this exercise, you will:
- Implement multiple document chunking strategies
- Understand the trade-offs between different approaches
- Learn to choose appropriate chunk sizes and overlap
- Handle edge cases in text splitting
- Evaluate chunking quality metrics
- Apply chunking strategies to different document types

## Prerequisites

- Completed Week 4 Labs 1-2
- Understanding of tokenization
- Familiarity with regex for text processing
- OpenAI API access

## Setup

```python
import openai
import tiktoken
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

client = openai.OpenAI()
```

## Part 1: Implement Chunking Strategies

### Task 1.1: Fixed-Size Token Chunking

Implement a chunking strategy that splits text into fixed-size chunks based on token count with configurable overlap.

```python
@dataclass
class Chunk:
    content: str
    chunk_id: str
    metadata: Dict[str, Any]
    start_position: int
    end_position: int
    token_count: int

class TokenBasedChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50, model: str = "gpt-3.5-turbo"):
        """
        TODO: Initialize the token-based chunker.
        
        Parameters:
        - chunk_size: Target tokens per chunk
        - overlap: Overlapping tokens between chunks
        - model: Model for tokenizer
        """
        pass
    
    def chunk_text(self, text: str, document_id: str) -> List[Chunk]:
        """
        TODO: Chunk text into fixed-size token chunks with overlap.
        
        Steps:
        1. Tokenize the full text
        2. Create chunks of chunk_size tokens
        3. Apply overlap by moving forward by (chunk_size - overlap) tokens
        4. Decode tokens back to text for each chunk
        5. Track start/end positions
        6. Return list of Chunk objects with metadata
        
        Handle edge cases:
        - Last chunk may be smaller than chunk_size
        - Overlap should not exceed chunk_size
        - Empty text should return empty list
        """
        pass

# Test your implementation
sample_text = """
Machine learning is a subset of artificial intelligence that focuses on the development
of algorithms and statistical models that enable computer systems to improve their performance
on a specific task through experience. The three main types of machine learning are supervised
learning, unsupervised learning, and reinforcement learning.

Deep learning is a specialized branch of machine learning that uses neural networks with
multiple layers to progressively extract higher-level features from raw input. Deep learning
has revolutionized fields such as computer vision, natural language processing, and speech
recognition.

Transformers are a type of deep learning architecture that has become the foundation for
modern large language models. They use self-attention mechanisms to process sequential data
in parallel, making them highly efficient and effective for tasks like translation and
text generation.
"""

# TODO: Test with different chunk sizes (200, 500, 1000 tokens)
# TODO: Test with different overlaps (0, 25, 50, 100 tokens)
# TODO: Count total chunks and check for proper overlap
```

### Task 1.2: Semantic Boundary Chunking

Implement chunking that respects semantic boundaries (sentences, paragraphs) while staying within token limits.

```python
class SemanticChunker:
    def __init__(self, max_chunk_size: int = 500, model: str = "gpt-3.5-turbo"):
        """
        TODO: Initialize semantic chunker.
        """
        pass
    
    def chunk_by_sentences(self, text: str, document_id: str) -> List[Chunk]:
        """
        TODO: Chunk text by sentences while respecting token limits.
        
        Steps:
        1. Split text into sentences (use regex: r'(?<=[.!?])\s+')
        2. Accumulate sentences until reaching max_chunk_size tokens
        3. Start new chunk when adding next sentence would exceed limit
        4. Handle long sentences that exceed max_chunk_size alone
        5. Apply minimal overlap (1-2 sentences) for context continuity
        6. Create Chunk objects with metadata
        
        Benefits:
        - Preserves semantic coherence
        - Avoids mid-sentence cuts
        - More natural context for generation
        
        Challenges:
        - Variable chunk sizes
        - Handling very long sentences
        - Determining optimal overlap
        """
        pass
    
    def chunk_by_paragraphs(self, text: str, document_id: str) -> List[Chunk]:
        """
        TODO: Chunk text by paragraphs with fallback to sentences.
        
        Steps:
        1. Split text by paragraphs (double newlines: r'\n\n+')
        2. If paragraph fits in max_chunk_size, use as chunk
        3. If paragraph too large, split by sentences
        4. If sentence too large, fall back to token-based splitting
        5. Apply paragraph-level overlap when beneficial
        
        When to use:
        - Structured documents with clear paragraphs
        - Documents where paragraph = semantic unit
        - Technical documentation
        """
        pass

# TODO: Test semantic chunking on sample_text
# TODO: Compare chunk boundaries with token-based approach
# TODO: Measure semantic coherence (how often sentences are split)
```

### Task 1.3: Hierarchical Document Chunking

Implement chunking for structured documents (markdown, HTML) that preserves hierarchy.

```python
class HierarchicalChunker:
    def __init__(self, max_chunk_size: int = 600, model: str = "gpt-3.5-turbo"):
        """
        TODO: Initialize hierarchical chunker.
        """
        pass
    
    def chunk_markdown(self, text: str, document_id: str) -> List[Chunk]:
        """
        TODO: Chunk markdown while preserving document structure.
        
        Steps:
        1. Identify headers (# ## ### etc.) to determine sections
        2. Each section becomes candidate chunk
        3. Include header in chunk for context
        4. If section too large, split by subsections
        5. If no subsections, fall back to sentence splitting
        6. Store header hierarchy in metadata (e.g., "h1:Introduction > h2:Overview")
        7. Apply section-aware overlap
        
        Metadata to include:
        - section_title: The header text
        - header_level: 1-6 for h1-h6
        - parent_section: Previous header in hierarchy
        - section_index: Position in document structure
        
        Benefits:
        - Preserves document structure
        - Chunks are topically coherent
        - Easy to generate citations (Section 2.1: ...)
        """
        pass
    
    def extract_section_hierarchy(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        TODO: Extract section hierarchy from markdown.
        
        Return list of:
        {
            'level': 1-6,
            'title': 'Section Title',
            'start_pos': char index,
            'end_pos': char index,
            'parent': parent section title or None
        }
        """
        pass

# Test markdown chunking
markdown_doc = """
# Machine Learning Guide

## Introduction

Machine learning is a method of data analysis that automates analytical model building.

## Types of Machine Learning

### Supervised Learning

Supervised learning uses labeled training data to learn the mapping from input to output.

### Unsupervised Learning

Unsupervised learning finds hidden patterns in data without labeled examples.

## Deep Learning

### Neural Networks

Neural networks are computing systems inspired by biological neural networks.

### Transformers

Transformers use self-attention mechanisms for processing sequential data.
"""

# TODO: Implement and test hierarchical chunking
# TODO: Verify metadata includes section hierarchy
# TODO: Check that chunks respect document structure
```

## Part 2: Evaluate Chunking Quality

### Task 2.1: Implement Chunking Metrics

Create metrics to evaluate and compare chunking strategies.

```python
class ChunkingEvaluator:
    def __init__(self, chunker, text: str, document_id: str):
        """
        TODO: Initialize evaluator with chunker and text.
        """
        self.chunker = chunker
        self.text = text
        self.document_id = document_id
        self.chunks = None
    
    def evaluate(self) -> Dict[str, Any]:
        """
        TODO: Compute comprehensive chunking metrics.
        
        Metrics to calculate:
        
        1. BASIC STATS:
           - total_chunks: Number of chunks created
           - avg_chunk_size: Mean tokens per chunk
           - std_chunk_size: Standard deviation of chunk sizes
           - min_chunk_size: Smallest chunk
           - max_chunk_size: Largest chunk
        
        2. COVERAGE:
           - text_coverage: % of original text preserved in chunks
           - overlap_ratio: Average % of overlap between consecutive chunks
        
        3. SEMANTIC COHERENCE:
           - sentence_split_rate: % of sentences that are split across chunks
           - paragraph_preservation: % of paragraphs kept intact
        
        4. EFFICIENCY:
           - tokens_per_chunk_target: How close to target size
           - chunk_size_variance: Consistency of chunk sizes
        
        Return all metrics in a dictionary.
        """
        pass
    
    def calculate_sentence_split_rate(self) -> float:
        """
        TODO: Calculate how often sentences are split mid-way.
        
        Steps:
        1. Extract all sentences from original text
        2. For each sentence, check if it appears complete in any chunk
        3. Calculate % of sentences that are split
        
        Lower is better (means fewer broken sentences).
        """
        pass
    
    def calculate_embedding_coherence(self) -> float:
        """
        TODO: Measure semantic coherence using embeddings.
        
        Steps:
        1. Get embeddings for each chunk
        2. Calculate cosine similarity between consecutive chunks
        3. Return average similarity score
        
        Higher similarity = better semantic continuity between chunks.
        
        Interpretation:
        - 0.9-1.0: Excellent continuity (might be too much overlap)
        - 0.7-0.9: Good continuity
        - 0.5-0.7: Moderate continuity
        - <0.5: Poor continuity (chunks too different)
        """
        pass

# TODO: Evaluate all three chunking strategies on the same text
# TODO: Create comparison table of metrics
# TODO: Identify which strategy works best for which metrics
```

### Task 2.2: Compare Strategies

```python
def compare_chunking_strategies(text: str, document_id: str = "test_doc"):
    """
    TODO: Compare all chunking strategies systematically.
    
    Test configurations:
    1. Token-based (200, 300, 500, 1000 tokens) with 10% overlap
    2. Sentence-based (max 500 tokens)
    3. Paragraph-based (max 500 tokens)
    4. Hierarchical/markdown (max 600 tokens)
    
    For each strategy:
    - Run ChunkingEvaluator
    - Collect metrics
    - Time the chunking operation
    
    Create comparison table showing:
    - Strategy name
    - Avg chunk size
    - Total chunks
    - Sentence split rate
    - Embedding coherence
    - Processing time
    
    Return DataFrame with results.
    """
    pass

# TODO: Run comparison on sample_text and markdown_doc
# TODO: Identify best strategy for each document type
# TODO: Document trade-offs (speed vs quality, consistency vs semantics)
```

## Part 3: Advanced Chunking Patterns

### Task 3.1: Smart Chunking with Auto-Selection

Implement an intelligent chunker that automatically selects the best strategy based on document characteristics.

```python
class SmartChunker:
    def __init__(self, max_chunk_size: int = 500, model: str = "gpt-3.5-turbo"):
        """
        TODO: Initialize smart chunker with all strategies.
        """
        self.max_chunk_size = max_chunk_size
        self.token_chunker = TokenBasedChunker(max_chunk_size, overlap=50, model=model)
        self.semantic_chunker = SemanticChunker(max_chunk_size, model=model)
        self.hierarchical_chunker = HierarchicalChunker(max_chunk_size, model=model)
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        TODO: Analyze document to determine characteristics.
        
        Detect:
        - has_markdown_headers: bool (contains # headers)
        - has_clear_paragraphs: bool (>3 instances of double newlines)
        - avg_sentence_length: float (tokens per sentence)
        - has_structure: bool (markdown/HTML structure)
        - text_length: int (total tokens)
        
        Return analysis dictionary.
        """
        pass
    
    def select_strategy(self, analysis: Dict[str, Any]) -> str:
        """
        TODO: Select best chunking strategy based on analysis.
        
        Decision logic:
        - If has_markdown_headers: use 'hierarchical'
        - Elif has_clear_paragraphs: use 'paragraph'
        - Elif avg_sentence_length < max_chunk_size * 0.8: use 'sentence'
        - Else: use 'token'
        
        Return strategy name and reasoning.
        """
        pass
    
    def chunk(self, text: str, document_id: str) -> Tuple[List[Chunk], Dict[str, Any]]:
        """
        TODO: Automatically chunk using best strategy.
        
        Steps:
        1. Analyze document
        2. Select strategy
        3. Apply selected chunker
        4. Return chunks and metadata about decision
        
        Metadata:
        - strategy_used: str
        - reason: str (why this strategy)
        - analysis: Dict (document characteristics)
        - metrics: Dict (chunking results)
        """
        pass

# TODO: Test SmartChunker on various document types
# TODO: Verify it selects appropriate strategies
# TODO: Compare results with manual strategy selection
```

### Task 3.2: Sliding Window with Dynamic Overlap

Implement a sliding window chunker with adaptive overlap based on content similarity.

```python
class SlidingWindowChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        min_overlap: int = 25,
        max_overlap: int = 100,
        model: str = "gpt-3.5-turbo"
    ):
        """
        TODO: Initialize sliding window chunker.
        
        Concept: Adjust overlap dynamically based on semantic similarity
        between chunk boundaries. Higher overlap for more coherent content.
        """
        pass
    
    def chunk_with_adaptive_overlap(self, text: str, document_id: str) -> List[Chunk]:
        """
        TODO: Implement adaptive overlap chunking.
        
        Algorithm:
        1. Create initial chunk of chunk_size tokens
        2. Get embedding for end portion of chunk (last 50 tokens)
        3. Get embedding for start portion of next window (next 50 tokens)
        4. Calculate similarity between end and start
        5. If similarity > 0.8: use min_overlap (low context break)
        6. If similarity < 0.5: use max_overlap (high context break)
        7. Else: interpolate between min and max based on similarity
        8. Move window forward by (chunk_size - calculated_overlap)
        9. Repeat until text exhausted
        
        Benefits:
        - Maintains context where needed
        - Reduces redundancy where content is distinct
        - Balances efficiency and quality
        """
        pass
    
    def calculate_boundary_similarity(self, chunk1_end: str, chunk2_start: str) -> float:
        """
        TODO: Calculate semantic similarity at chunk boundaries.
        
        Use embeddings to measure how related the content is at boundaries.
        """
        pass

# TODO: Test adaptive overlap on sample text
# TODO: Compare with fixed overlap strategies
# TODO: Measure if it improves retrieval quality
```

## Part 4: Real-World Application

### Task 4.1: Multi-Document Chunking Pipeline

Build a production-ready chunking pipeline for multiple document types.

```python
from typing import List, Literal
from pathlib import Path

class DocumentChunkingPipeline:
    """
    TODO: Production pipeline for chunking multiple documents.
    """
    
    def __init__(self, output_dir: str = "./chunks"):
        """
        TODO: Initialize pipeline with all chunkers.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.smart_chunker = SmartChunker()
        self.processed_documents = []
    
    def process_document(
        self,
        content: str,
        document_id: str,
        doc_type: Literal["markdown", "text", "technical", "narrative"],
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """
        TODO: Process a single document through the pipeline.
        
        Steps:
        1. Validate input
        2. Apply appropriate preprocessing (clean text, normalize)
        3. Use SmartChunker to chunk
        4. Enrich chunks with document metadata
        5. Save chunks to disk (JSON)
        6. Return chunks
        
        Metadata to add:
        - document_id
        - document_type
        - processed_timestamp
        - chunking_strategy_used
        - source_file (if provided)
        - custom metadata
        """
        pass
    
    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        parallel: bool = False
    ) -> Dict[str, List[Chunk]]:
        """
        TODO: Process multiple documents.
        
        Args:
        - documents: List of {content, document_id, doc_type, metadata}
        - parallel: Whether to process concurrently
        
        Returns:
        - Dictionary mapping document_id to chunks
        
        If parallel=True, use ThreadPoolExecutor for concurrent processing.
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        TODO: Get pipeline statistics.
        
        Return:
        - total_documents_processed
        - total_chunks_created
        - avg_chunks_per_document
        - strategies_used (counter)
        - total_processing_time
        """
        pass

# TODO: Create sample documents of different types
# TODO: Process them through the pipeline
# TODO: Verify chunks are saved correctly
# TODO: Check statistics are accurate
```

### Task 4.2: Chunking Quality Report

```python
def generate_chunking_report(chunks: List[Chunk]) -> str:
    """
    TODO: Generate a comprehensive HTML/Markdown report on chunking quality.
    
    Include:
    1. Summary Statistics:
       - Total chunks
       - Size distribution (histogram)
       - Strategy used
    
    2. Quality Metrics:
       - Sentence preservation rate
       - Semantic coherence score
       - Coverage percentage
    
    3. Sample Chunks:
       - Show first 3 chunks with metadata
       - Highlight overlap regions
    
    4. Recommendations:
       - Suggested improvements
       - Optimal parameters for this document type
    
    Return formatted markdown report.
    """
    pass

# TODO: Generate report for processed documents
# TODO: Include visualizations (if possible)
```

## Part 5: Reflection Questions

Answer the following questions based on your implementation and experiments:

### Conceptual Understanding

1. **Trade-offs**: What are the main trade-offs between fixed-size token chunking and semantic boundary chunking? When would you use each?

2. **Overlap Strategy**: How does chunk overlap affect RAG performance? What are the downsides of too much or too little overlap?

3. **Document Structure**: Why is preserving document structure (headers, sections) important for RAG systems? Give examples where it matters most.

4. **Chunk Size**: How would you determine the optimal chunk size for a specific use case? What factors should you consider?

5. **Edge Cases**: What edge cases did you encounter in chunking (e.g., very long sentences, missing paragraphs, tables)? How did you handle them?

### Implementation Insights

6. **Performance**: Which chunking strategy was fastest? Which produced the highest quality chunks? Did you need to make speed vs. quality trade-offs?

7. **Semantic Coherence**: How did you measure whether chunks maintained semantic coherence? What metrics were most useful?

8. **Auto-Selection**: Did the SmartChunker reliably choose appropriate strategies? Were there cases where it made suboptimal choices?

### Production Considerations

9. **Scalability**: How would you scale your chunking pipeline to process millions of documents? What bottlenecks might you encounter?

10. **Maintenance**: If the chunking strategy needs to change after documents are already processed, how would you handle re-chunking? What about version control?

11. **Monitoring**: What metrics would you monitor in production to ensure chunking quality remains high? How would you detect when chunking degrades?

## Deliverables

Submit the following:

1. **Code**: Complete implementations of all chunking strategies
2. **Evaluation Results**: Comparison table showing metrics for each strategy
3. **Report**: 2-3 page analysis document covering:
   - Strategy comparison and recommendations
   - Metrics interpretation
   - Answers to reflection questions
   - Best practices guide
4. **Example Output**: Sample chunks from different strategies with annotations

## Evaluation Rubric

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|-------------------|------------------|
| **Implementation** | All strategies implemented correctly with edge case handling | Most strategies work, minor bugs | Basic implementation, missing features | Incomplete or non-functional |
| **Evaluation** | Comprehensive metrics, insightful analysis | Good metrics, reasonable analysis | Basic metrics, limited analysis | Metrics missing or incorrect |
| **Code Quality** | Clean, documented, production-ready | Generally clean, some documentation | Functional but messy | Poor structure, hard to follow |
| **Reflection** | Deep insights, shows experimentation | Good understanding, thoughtful | Basic answers, limited depth | Superficial or missing |

## Additional Resources

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Pinecone Chunking Strategies Guide](https://www.pinecone.io/learn/chunking-strategies/)
- [OpenAI Tokenizer Playground](https://platform.openai.com/tokenizer)
- Research: "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)

## Hints

1. Start with token-based chunking - it's the simplest and helps understand the basics
2. Test each strategy on the same text to make fair comparisons
3. Embeddings-based coherence is expensive - cache them during evaluation
4. Consider using sentence-transformers for faster embeddings during development
5. Visualize chunk boundaries on actual text to debug issues
6. Small chunk sizes (200-300 tokens) work well for precise retrieval
7. Larger chunks (800-1000 tokens) work better when you need more context
8. Always include source metadata for traceability

---

**Estimated Time:** 90 minutes  
**Difficulty:** Intermediate  
**Topics:** Document processing, tokenization, semantic analysis, evaluation metrics

Good luck! Remember, chunking is an art as much as a science - there's rarely one "correct" answer, only trade-offs to understand and optimize for your specific use case.
