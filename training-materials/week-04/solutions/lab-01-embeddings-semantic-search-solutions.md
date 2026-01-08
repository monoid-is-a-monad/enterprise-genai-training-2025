# Lab 1 Solutions: Embeddings & Semantic Search

**Week 4 - RAG Fundamentals**

## Table of Contents
1. [Part 1: Understanding Text Embeddings](#part-1-understanding-text-embeddings)
2. [Part 2: Similarity Metrics](#part-2-similarity-metrics)
3. [Part 3: Building a Vector Database](#part-3-building-a-vector-database)
4. [Part 4: Semantic Search Engine](#part-4-semantic-search-engine)
5. [Part 5: Optimization Techniques](#part-5-optimization-techniques)
6. [Part 6: Production Considerations](#part-6-production-considerations)

---

## Part 1: Understanding Text Embeddings

### Exercise 1.1: Explore Embedding Properties

```python
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding for text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# 1. Generate embeddings for sentence pairs
sentence_pairs = [
    ("The weather is sunny", "It's a bright day"),
    ("I am happy", "I am sad"),
    ("Machine learning is fascinating", "AI is interesting"),
    ("The car is red", "The vehicle is crimson"),
    ("Hello world", "Goodbye universe")
]

print("=" * 80)
print("SENTENCE PAIR SIMILARITIES")
print("=" * 80)

for pair in sentence_pairs:
    emb1 = np.array(get_embedding(pair[0]))
    emb2 = np.array(get_embedding(pair[1]))
    
    # Calculate cosine similarity
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    print(f"\n{pair[0]}")
    print(f"  <-> {pair[1]}")
    print(f"  Similarity: {similarity:.4f}")

# Expected results:
# - "sunny" / "bright day": ~0.85-0.90 (high similarity)
# - "happy" / "sad": ~0.75-0.80 (moderately high, both emotions)
# - "machine learning" / "AI": ~0.88-0.92 (very high similarity)
# - "red car" / "crimson vehicle": ~0.90-0.93 (synonyms)
# - "hello" / "goodbye": ~0.70-0.75 (both greetings but opposite)
```

**Analysis:**
- Most similar: "red car" / "crimson vehicle" and "machine learning" / "AI"
- Least similar: "hello world" / "goodbye universe"
- Synonyms have very high similarity (~0.90+)
- Opposite concepts still show moderate similarity if in same domain

### Testing Variations

```python
# 2. Test punctuation and case sensitivity
variations = [
    "Hello",
    "Hello!",
    "hello",
    "HELLO",
    "Hello...",
    "hello world"
]

print("\n" + "=" * 80)
print("PUNCTUATION AND CASE SENSITIVITY")
print("=" * 80)

base_embedding = np.array(get_embedding("Hello"))

for variant in variations[1:]:
    variant_embedding = np.array(get_embedding(variant))
    similarity = cosine_similarity([base_embedding], [variant_embedding])[0][0]
    print(f"'Hello' vs '{variant}': {similarity:.6f}")

# Observations:
# - Case differences: minimal impact (~0.998-0.999 similarity)
# - Punctuation: very minimal impact (~0.997-0.999 similarity)
# - Additional words: more significant impact (~0.85-0.90)
# - Embeddings are relatively robust to formatting
```

---

## Part 2: Similarity Metrics

### Exercise 2.1: Compare Distance Metrics

```python
from typing import List, Dict, Tuple

# Create diverse corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn vulpine leaps above the idle canine",  # Similar to #1
    "Python is a powerful programming language",
    "JavaScript is used for web development",
    "The Eiffel Tower is located in Paris, France",
    "Mount Everest is the highest mountain on Earth",
    "Artificial intelligence is transforming technology",
    "Machine learning algorithms can recognize patterns",  # Similar to #7
    "The sun rises in the east and sets in the west",
    "Climate change affects global weather patterns",
    "Shakespeare wrote many famous plays including Hamlet",
    "Mozart composed beautiful classical music",
    "The Pacific Ocean is the largest body of water",
    "The Great Wall of China is visible from space",  # Debatable fact
    "Dogs are loyal companions and popular pets"
]

# Generate embeddings
print("Generating embeddings for corpus...")
corpus_embeddings = np.array([get_embedding(doc) for doc in corpus])

# Test query
query = "Machine learning and AI are related fields"
query_embedding = np.array(get_embedding(query))

# Compare metrics
def compare_metrics(query_emb: np.ndarray, corpus_embs: np.ndarray, top_k: int = 5):
    """Compare different distance metrics for ranking."""
    
    results = {}
    
    # Cosine similarity
    cosine_scores = cosine_similarity([query_emb], corpus_embs)[0]
    cosine_top = np.argsort(cosine_scores)[::-1][:top_k]
    results['cosine'] = [(idx, cosine_scores[idx]) for idx in cosine_top]
    
    # Euclidean distance
    euclidean_dists = np.linalg.norm(corpus_embs - query_emb, axis=1)
    euclidean_top = np.argsort(euclidean_dists)[:top_k]
    results['euclidean'] = [(idx, -euclidean_dists[idx]) for idx in euclidean_top]
    
    # Dot product
    dot_products = np.dot(corpus_embs, query_emb)
    dot_top = np.argsort(dot_products)[::-1][:top_k]
    results['dot_product'] = [(idx, dot_products[idx]) for idx in dot_top]
    
    # Manhattan distance
    manhattan_dists = np.sum(np.abs(corpus_embs - query_emb), axis=1)
    manhattan_top = np.argsort(manhattan_dists)[:top_k]
    results['manhattan'] = [(idx, -manhattan_dists[idx]) for idx in manhattan_top]
    
    return results

print("\n" + "=" * 80)
print(f"QUERY: {query}")
print("=" * 80)

results = compare_metrics(query_embedding, corpus_embeddings)

for metric, top_docs in results.items():
    print(f"\n{metric.upper()}:")
    print("-" * 80)
    for idx, score in top_docs:
        print(f"  [{score:>10.4f}] {corpus[idx]}")

# Analysis:
# - Cosine similarity: Best for semantic similarity (ignores magnitude)
# - Euclidean: Similar to cosine for normalized vectors
# - Dot product: Considers magnitude (can bias toward longer texts)
# - Manhattan: Similar to Euclidean but less sensitive to outliers
# 
# Recommendation: Cosine similarity for most semantic search tasks
```

**Key Findings:**
- **Cosine similarity** is best for semantic search (most commonly used)
- All metrics should rank AI/ML-related documents highest
- Expected top results: #7, #8 (AI/ML documents)
- Dot product can be faster but may bias results by document length

---

## Part 3: Building a Vector Database

### Complete Vector Database Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import pickle
from pathlib import Path

@dataclass
class Document:
    """Represents a document in the vector database."""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            embedding=np.array(data['embedding']),
            metadata=data['metadata'],
            created_at=datetime.fromisoformat(data['created_at'])
        )


class VectorDatabase:
    """
    Simple in-memory vector database for semantic search.
    """
    
    def __init__(self, client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize vector database.
        
        Args:
            client: OpenAI client for generating embeddings
            embedding_model: Embedding model to use
        """
        self.client = client
        self.embedding_model = embedding_model
        self.documents: Dict[str, Document] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.document_ids: List[str] = []
        
        # Statistics
        self.total_searches = 0
        self.total_insertions = 0
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding)
    
    def _rebuild_matrix(self):
        """Rebuild embeddings matrix from documents."""
        if not self.documents:
            self.embeddings_matrix = None
            self.document_ids = []
            return
        
        self.document_ids = list(self.documents.keys())
        embeddings = [self.documents[doc_id].embedding for doc_id in self.document_ids]
        self.embeddings_matrix = np.array(embeddings)
    
    def insert(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Insert a document into the database.
        
        Args:
            content: Document text
            metadata: Optional metadata
            doc_id: Optional document ID (auto-generated if not provided)
        
        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.documents)}_{datetime.now().timestamp()}"
        
        # Generate embedding
        embedding = self._get_embedding(content)
        
        # Create document
        doc = Document(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Store document
        self.documents[doc_id] = doc
        self.total_insertions += 1
        
        # Rebuild matrix
        self._rebuild_matrix()
        
        return doc_id
    
    def insert_batch(
        self,
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Insert multiple documents efficiently.
        
        Args:
            contents: List of document texts
            metadatas: Optional list of metadata dicts
        
        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(contents)
        
        # Generate embeddings in batch
        texts = [text.replace("\n", " ") for text in contents]
        response = self.client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        
        doc_ids = []
        for i, (content, metadata) in enumerate(zip(contents, metadatas)):
            doc_id = f"doc_{len(self.documents)}_{i}_{datetime.now().timestamp()}"
            embedding = np.array(response.data[i].embedding)
            
            doc = Document(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            
            self.documents[doc_id] = doc
            doc_ids.append(doc_id)
            self.total_insertions += 1
        
        # Rebuild matrix once
        self._rebuild_matrix()
        
        return doc_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of (Document, similarity_score) tuples
        """
        if not self.documents:
            return []
        
        self.total_searches += 1
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Filter documents by metadata if needed
        if filter_metadata:
            filtered_docs = [
                doc for doc in self.documents.values()
                if all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
            if not filtered_docs:
                return []
            
            filtered_ids = [doc.id for doc in filtered_docs]
            filtered_embeddings = np.array([doc.embedding for doc in filtered_docs])
        else:
            filtered_ids = self.document_ids
            filtered_embeddings = self.embeddings_matrix
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return documents with scores
        results = [
            (self.documents[filtered_ids[idx]], similarities[idx])
            for idx in top_indices
        ]
        
        return results
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """Delete document by ID."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._rebuild_matrix()
            return True
        return False
    
    def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update document content or metadata.
        
        Args:
            doc_id: Document ID
            content: New content (will regenerate embedding)
            metadata: New or updated metadata
        
        Returns:
            True if successful, False if document not found
        """
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        
        if content is not None:
            # Regenerate embedding
            embedding = self._get_embedding(content)
            doc.content = content
            doc.embedding = embedding
            self._rebuild_matrix()
        
        if metadata is not None:
            doc.metadata.update(metadata)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_documents': len(self.documents),
            'total_searches': self.total_searches,
            'total_insertions': self.total_insertions,
            'embedding_model': self.embedding_model,
            'embedding_dimension': len(self.embeddings_matrix[0]) if self.embeddings_matrix is not None else 0
        }
    
    def save(self, filepath: str):
        """Save database to file."""
        data = {
            'documents': [doc.to_dict() for doc in self.documents.values()],
            'embedding_model': self.embedding_model,
            'stats': {
                'total_searches': self.total_searches,
                'total_insertions': self.total_insertions
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load database from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.embedding_model = data['embedding_model']
        self.documents = {
            doc['id']: Document.from_dict(doc)
            for doc in data['documents']
        }
        
        self.total_searches = data['stats']['total_searches']
        self.total_insertions = data['stats']['total_insertions']
        
        self._rebuild_matrix()


# Test the vector database
print("\n" + "=" * 80)
print("VECTOR DATABASE DEMO")
print("=" * 80)

db = VectorDatabase(client)

# Insert documents
documents = [
    ("Python is a versatile programming language", {"category": "programming", "language": "python"}),
    ("Machine learning models can predict outcomes", {"category": "ai", "subtopic": "ml"}),
    ("The Great Wall of China is a historic landmark", {"category": "history", "location": "china"}),
    ("Photosynthesis is how plants convert sunlight", {"category": "science", "field": "biology"}),
    ("JavaScript enables interactive web pages", {"category": "programming", "language": "javascript"}),
    ("Neural networks are inspired by the brain", {"category": "ai", "subtopic": "deep learning"}),
    ("The Colosseum was built in ancient Rome", {"category": "history", "location": "italy"}),
    ("DNA contains genetic information", {"category": "science", "field": "biology"}),
]

print("\nInserting documents...")
for content, metadata in documents:
    db.insert(content, metadata)

print(f"✓ Inserted {len(documents)} documents")

# Perform searches
queries = [
    ("What is machine learning?", None),
    ("Tell me about programming languages", {"category": "programming"}),
    ("Ancient buildings and monuments", {"category": "history"}),
]

for query, filters in queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    if filters:
        print(f"Filters: {filters}")
    print('-' * 80)
    
    results = db.search(query, top_k=3, filter_metadata=filters)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. [{score:.4f}] {doc.content}")
        print(f"   Metadata: {doc.metadata}")

# Show statistics
print(f"\n{'='*80}")
print("DATABASE STATISTICS")
print('=' * 80)
stats = db.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")
```

---

## Part 4: Semantic Search Engine

### Complete Semantic Search Implementation

```python
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

@dataclass
class SearchResult:
    """Search result with relevance information."""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str] = None


class SemanticSearchEngine:
    """
    Production-ready semantic search engine.
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        enable_reranking: bool = True,
        enable_highlighting: bool = True
    ):
        """
        Initialize search engine.
        
        Args:
            vector_db: Vector database instance
            enable_reranking: Enable hybrid search with reranking
            enable_highlighting: Enable result highlighting
        """
        self.db = vector_db
        self.enable_reranking = enable_reranking
        self.enable_highlighting = enable_highlighting
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction (remove stop words, etc.)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for'}
        words = re.findall(r'\w+', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword matching score."""
        content_lower = content.lower()
        matches = sum(1 for kw in keywords if kw in content_lower)
        return matches / len(keywords) if keywords else 0.0
    
    def _rerank(
        self,
        results: List[Tuple[Document, float]],
        query: str,
        alpha: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """
        Rerank results using hybrid scoring (semantic + keyword).
        
        Args:
            results: Initial search results
            query: Original query
            alpha: Weight for semantic score (1-alpha for keyword score)
        
        Returns:
            Reranked results
        """
        keywords = self._extract_keywords(query)
        
        reranked = []
        for doc, semantic_score in results:
            keyword_score = self._keyword_score(doc.content, keywords)
            hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score
            reranked.append((doc, hybrid_score))
        
        # Sort by hybrid score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def _highlight(self, content: str, query: str, context_chars: int = 50) -> List[str]:
        """
        Extract highlighted snippets from content.
        
        Args:
            content: Document content
            query: Search query
            context_chars: Characters of context around match
        
        Returns:
            List of highlighted snippets
        """
        keywords = self._extract_keywords(query)
        highlights = []
        
        content_lower = content.lower()
        
        for keyword in keywords:
            start = 0
            while True:
                idx = content_lower.find(keyword, start)
                if idx == -1:
                    break
                
                # Extract context
                snippet_start = max(0, idx - context_chars)
                snippet_end = min(len(content), idx + len(keyword) + context_chars)
                snippet = content[snippet_start:snippet_end]
                
                # Add ellipsis if needed
                if snippet_start > 0:
                    snippet = "..." + snippet
                if snippet_end < len(content):
                    snippet = snippet + "..."
                
                # Bold the keyword
                snippet = snippet.replace(
                    content[idx:idx+len(keyword)],
                    f"**{content[idx:idx+len(keyword)]}**"
                )
                
                highlights.append(snippet)
                start = idx + 1
        
        return highlights[:3]  # Limit to 3 highlights
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        rerank: Optional[bool] = None,
        highlight: Optional[bool] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            rerank: Override reranking setting
            highlight: Override highlighting setting
        
        Returns:
            List of search results
        """
        # Get initial results
        raw_results = self.db.search(query, top_k=top_k * 2, filter_metadata=filters)
        
        if not raw_results:
            return []
        
        # Rerank if enabled
        if (rerank if rerank is not None else self.enable_reranking):
            raw_results = self._rerank(raw_results, query)
        
        # Limit to top_k
        raw_results = raw_results[:top_k]
        
        # Create search results
        results = []
        for doc, score in raw_results:
            highlights = None
            if (highlight if highlight is not None else self.enable_highlighting):
                highlights = self._highlight(doc.content, query)
            
            result = SearchResult(
                document_id=doc.id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                highlights=highlights
            )
            results.append(result)
        
        return results
    
    def search_with_explanation(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search with detailed explanation of results.
        
        Returns:
            Dictionary with results and explanations
        """
        results = self.search(query, top_k=top_k)
        
        keywords = self._extract_keywords(query)
        
        explanation = {
            'query': query,
            'extracted_keywords': keywords,
            'total_results': len(results),
            'results': []
        }
        
        for result in results:
            keyword_matches = [
                kw for kw in keywords
                if kw in result.content.lower()
            ]
            
            result_info = {
                'content': result.content,
                'semantic_score': result.score,
                'keyword_matches': keyword_matches,
                'metadata': result.metadata
            }
            explanation['results'].append(result_info)
        
        return explanation


# Test semantic search engine
print("\n" + "=" * 80)
print("SEMANTIC SEARCH ENGINE")
print("=" * 80)

search_engine = SemanticSearchEngine(db, enable_reranking=True, enable_highlighting=True)

test_queries = [
    "How do computers learn from data?",
    "Famous historical buildings",
    "Biology and life sciences",
]

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    
    results = search_engine.search(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result.score:.4f}] {result.content}")
        if result.highlights:
            print(f"   Highlights:")
            for highlight in result.highlights:
                print(f"     • {highlight}")
        print(f"   Metadata: {result.metadata}")

# Test with explanation
print(f"\n{'='*80}")
print("SEARCH WITH EXPLANATION")
print('='*80)

explanation = search_engine.search_with_explanation("machine learning algorithms", top_k=3)
print(json.dumps(explanation, indent=2))
```

---

## Part 5: Optimization Techniques

### Multi-Index Support

```python
class MultiIndexVectorDB:
    """
    Vector database with multiple indexes for different embedding models or domains.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.indexes: Dict[str, VectorDatabase] = {}
    
    def create_index(self, name: str, embedding_model: str = "text-embedding-3-small"):
        """Create a new index."""
        self.indexes[name] = VectorDatabase(self.client, embedding_model)
    
    def get_index(self, name: str) -> Optional[VectorDatabase]:
        """Get an index by name."""
        return self.indexes.get(name)
    
    def search_all(
        self,
        query: str,
        top_k_per_index: int = 5
    ) -> Dict[str, List[Tuple[Document, float]]]:
        """Search across all indexes."""
        results = {}
        for name, index in self.indexes.items():
            results[name] = index.search(query, top_k=top_k_per_index)
        return results


# Test multi-index
multi_db = MultiIndexVectorDB(client)
multi_db.create_index("general", "text-embedding-3-small")
multi_db.create_index("code", "text-embedding-3-large")  # Use larger model for code

print("✓ Created multi-index database")
```

### Approximate Nearest Neighbor (ANN) Optimization

```python
class ANNVectorDatabase(VectorDatabase):
    """
    Vector database with approximate nearest neighbor search for scale.
    """
    
    def __init__(self, client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        super().__init__(client, embedding_model)
        self.use_ann = False
        self.ann_index = None
    
    def build_ann_index(self):
        """Build ANN index for faster search (simple clustering-based approach)."""
        if len(self.documents) < 1000:
            # Don't use ANN for small collections
            self.use_ann = False
            return
        
        # For demonstration: simple k-means clustering
        from sklearn.cluster import KMeans
        
        n_clusters = min(100, len(self.documents) // 10)
        self.ann_index = KMeans(n_clusters=n_clusters, random_state=42)
        self.ann_index.fit(self.embeddings_matrix)
        self.use_ann = True
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search with ANN optimization.
        """
        if not self.use_ann or filter_metadata is not None:
            # Fall back to exact search
            return super().search(query, top_k, filter_metadata)
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Find closest cluster
        cluster_id = self.ann_index.predict([query_embedding])[0]
        
        # Get documents in that cluster and nearby clusters
        cluster_labels = self.ann_index.labels_
        nearby_indices = np.where(
            (cluster_labels == cluster_id) |
            (cluster_labels == (cluster_id + 1) % self.ann_index.n_clusters) |
            (cluster_labels == (cluster_id - 1) % self.ann_index.n_clusters)
        )[0]
        
        # Search only within these documents
        candidate_embeddings = self.embeddings_matrix[nearby_indices]
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.documents[self.document_ids[nearby_indices[idx]]], similarities[idx])
            for idx in top_indices
        ]
        
        return results

# Note: For production, use specialized ANN libraries like FAISS, Annoy, or HNSW
```

---

## Part 6: Production Considerations

### Performance Benchmarking

```python
import time
from typing import Callable

def benchmark_search(
    search_func: Callable,
    queries: List[str],
    name: str = "Search"
) -> Dict[str, float]:
    """
    Benchmark search performance.
    
    Args:
        search_func: Search function to test
        queries: List of test queries
        name: Name for the benchmark
    
    Returns:
        Performance metrics
    """
    print(f"\nBenchmarking {name}...")
    
    latencies = []
    
    for query in queries:
        start = time.time()
        results = search_func(query)
        latency = time.time() - start
        latencies.append(latency)
    
    metrics = {
        'avg_latency': np.mean(latencies),
        'p50_latency': np.percentile(latencies, 50),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
    }
    
    print(f"Results for {name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value*1000:.2f}ms")
    
    return metrics


# Benchmark the search
test_queries = [
    "machine learning algorithms",
    "historical landmarks",
    "programming languages",
    "biological processes",
    "artificial intelligence"
]

def search_func(query: str):
    return db.search(query, top_k=10)

metrics = benchmark_search(search_func, test_queries, "Vector Search")
```

### Best Practices Summary

```python
"""
PRODUCTION BEST PRACTICES FOR EMBEDDINGS & SEMANTIC SEARCH:

1. EMBEDDING MODEL SELECTION:
   - text-embedding-3-small: Fast, cost-effective (1536 dimensions)
   - text-embedding-3-large: Higher quality (3072 dimensions)
   - Consider domain-specific fine-tuned models

2. PERFORMANCE OPTIMIZATION:
   - Batch embedding generation (up to 2048 texts per request)
   - Cache embeddings to avoid regeneration
   - Use ANN algorithms for large collections (>10K documents)
   - Consider dimensionality reduction for very large scale

3. SEARCH QUALITY:
   - Use cosine similarity for most cases
   - Implement hybrid search (semantic + keyword)
   - Add reranking for better relevance
   - Use metadata filtering to narrow results

4. SCALABILITY:
   - Use dedicated vector databases (Pinecone, Weaviate, Qdrant)
   - Implement sharding for multi-billion document scale
   - Use approximate nearest neighbor (ANN) search
   - Cache frequent queries

5. MONITORING:
   - Track search latency (p50, p95, p99)
   - Monitor embedding API costs
   - Measure search relevance (clicks, user feedback)
   - Alert on quality degradation

6. COST MANAGEMENT:
   - Cache embeddings aggressively
   - Use smaller models where appropriate
   - Batch operations when possible
   - Monitor API usage

7. EVALUATION:
   - Create test query sets with expected results
   - Measure recall@k and precision@k
   - Collect user feedback on relevance
   - A/B test search algorithm changes

8. SECURITY:
   - Implement access control per document
   - Filter results based on user permissions
   - Sanitize query inputs
   - Audit sensitive searches
"""

print("""
✓ Lab 1 Complete!

Key Takeaways:
- Embeddings capture semantic meaning in vector form
- Cosine similarity is the standard metric for semantic search
- Vector databases enable efficient similarity search
- Production systems need caching, monitoring, and optimization
- Hybrid search (semantic + keyword) improves relevance

Next Steps:
- Proceed to Lab 2: Basic RAG Implementation
- Experiment with different embedding models
- Try implementing with a real vector database (Pinecone, Weaviate)
- Build evaluation metrics for your search quality
""")
```

---

## Additional Resources

### Recommended Vector Databases
- **Pinecone**: Managed, scalable vector database
- **Weaviate**: Open-source with GraphQL interface
- **Qdrant**: High-performance with filtering
- **Milvus**: Scalable for billions of vectors
- **FAISS**: Facebook's similarity search library

### Further Reading
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)
- [Semantic Search Best Practices](https://www.pinecone.io/learn/semantic-search/)

