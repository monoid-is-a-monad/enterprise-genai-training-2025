# Lab 2 Solutions: Basic RAG Implementation

**Week 4 - RAG Fundamentals**

## Table of Contents
1. [Part 1: Document Chunking Strategies](#part-1-document-chunking-strategies)
2. [Part 2: Building the RAG Pipeline](#part-2-building-the-rag-pipeline)
3. [Part 3: Context Assembly](#part-3-context-assembly)
4. [Part 4: Complete RAG System](#part-4-complete-rag-system)
5. [Part 5: RAG Evaluation](#part-5-rag-evaluation)
6. [Part 6: Optimization & Best Practices](#part-6-optimization--best-practices)

---

## Part 1: Document Chunking Strategies

### Complete Document Chunker Implementation

```python
import re
import tiktoken
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI()

@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[np.ndarray] = None


class DocumentChunker:
    """
    Advanced document chunking with multiple strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        tokenizer_name: str = "cl100k_base"
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            tokenizer_name: Tokenizer to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    def chunk_by_tokens(
        self,
        text: str,
        document_id: str = "doc"
    ) -> List[Chunk]:
        """
        Chunk text by token count with overlap.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
        
        Returns:
            List of chunks
        """
        # Tokenize entire text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Find actual character positions in original text
            # This is approximate but works well
            chars_per_token = len(text) / len(tokens)
            start_char = int(start_idx * chars_per_token)
            end_char = int(end_idx * chars_per_token)
            
            # Create chunk
            chunk = Chunk(
                content=chunk_text.strip(),
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                chunk_index=chunk_index,
                metadata={
                    "token_count": len(chunk_tokens),
                    "char_count": len(chunk_text),
                    "chunking_method": "tokens"
                },
                start_char=start_char,
                end_char=end_char
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def chunk_by_sentences(
        self,
        text: str,
        document_id: str = "doc"
    ) -> List[Chunk]:
        """
        Chunk text by sentences while respecting token limits.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
        
        Returns:
            List of chunks
        """
        # Split into sentences (improved regex)
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        char_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If single sentence exceeds limit, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    self._save_sentence_chunk(
                        chunks, current_chunk, document_id,
                        chunk_index, char_position
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by tokens
                sub_chunks = self.chunk_by_tokens(sentence, f"{document_id}_long_sentence")
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{document_id}_chunk_{chunk_index}"
                    sub_chunk.chunk_index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
                
                char_position += len(sentence)
                continue
            
            # Check if adding sentence exceeds limit
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                self._save_sentence_chunk(
                    chunks, current_chunk, document_id,
                    chunk_index, char_position
                )
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Keep last sentence for context
                    overlap_sentence = current_chunk[-1]
                    current_chunk = [overlap_sentence]
                    current_tokens = len(self.tokenizer.encode(overlap_sentence))
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            char_position += len(sentence)
        
        # Save final chunk
        if current_chunk:
            self._save_sentence_chunk(
                chunks, current_chunk, document_id,
                chunk_index, char_position
            )
        
        return chunks
    
    def _save_sentence_chunk(
        self,
        chunks: List[Chunk],
        sentences: List[str],
        document_id: str,
        chunk_index: int,
        char_position: int
    ):
        """Helper to save a sentence-based chunk."""
        chunk_text = " ".join(sentences)
        tokens = self.tokenizer.encode(chunk_text)
        
        chunk = Chunk(
            content=chunk_text,
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            chunk_index=chunk_index,
            metadata={
                "sentence_count": len(sentences),
                "token_count": len(tokens),
                "char_count": len(chunk_text),
                "chunking_method": "sentences"
            },
            start_char=char_position - len(chunk_text),
            end_char=char_position
        )
        
        chunks.append(chunk)
    
    def chunk_by_paragraphs(
        self,
        text: str,
        document_id: str = "doc"
    ) -> List[Chunk]:
        """
        Chunk text by paragraphs while respecting token limits.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
        
        Returns:
            List of chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = len(self.tokenizer.encode(para))
            
            # If paragraph is too long, split by sentences
            if para_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    self._save_paragraph_chunk(
                        chunks, chunk_text, document_id, chunk_index
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split paragraph by sentences
                para_chunks = self.chunk_by_sentences(para, document_id)
                for para_chunk in para_chunks:
                    para_chunk.chunk_id = f"{document_id}_chunk_{chunk_index}"
                    para_chunk.chunk_index = chunk_index
                    chunks.append(para_chunk)
                    chunk_index += 1
                
                continue
            
            # Check if adding paragraph exceeds limit
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                self._save_paragraph_chunk(
                    chunks, chunk_text, document_id, chunk_index
                )
                chunk_index += 1
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Save final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            self._save_paragraph_chunk(
                chunks, chunk_text, document_id, chunk_index
            )
        
        return chunks
    
    def _save_paragraph_chunk(
        self,
        chunks: List[Chunk],
        chunk_text: str,
        document_id: str,
        chunk_index: int
    ):
        """Helper to save a paragraph-based chunk."""
        tokens = self.tokenizer.encode(chunk_text)
        paragraph_count = chunk_text.count('\n\n') + 1
        
        chunk = Chunk(
            content=chunk_text,
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            chunk_index=chunk_index,
            metadata={
                "paragraph_count": paragraph_count,
                "token_count": len(tokens),
                "char_count": len(chunk_text),
                "chunking_method": "paragraphs"
            }
        )
        
        chunks.append(chunk)
    
    def chunk_markdown(
        self,
        text: str,
        document_id: str = "doc"
    ) -> List[Chunk]:
        """
        Chunk markdown text by headers while respecting token limits.
        
        Args:
            text: Markdown text to chunk
            document_id: Document identifier
        
        Returns:
            List of chunks
        """
        # Split by headers
        header_pattern = r'^#{1,6}\s+.+$'
        lines = text.split('\n')
        
        sections = []
        current_section = []
        current_header = None
        
        for line in lines:
            if re.match(header_pattern, line):
                # Save previous section
                if current_section:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_section)
                    })
                
                # Start new section
                current_header = line
                current_section = [line]
            else:
                current_section.append(line)
        
        # Save final section
        if current_section:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_section)
            })
        
        # Convert sections to chunks
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = section['content']
            section_tokens = len(self.tokenizer.encode(section_text))
            
            if section_tokens <= self.chunk_size:
                # Section fits in one chunk
                chunk = Chunk(
                    content=section_text,
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    metadata={
                        "header": section['header'],
                        "token_count": section_tokens,
                        "chunking_method": "markdown"
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Section too large, split further
                sub_chunks = self.chunk_by_paragraphs(section_text, document_id)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{document_id}_chunk_{chunk_index}"
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.metadata['header'] = section['header']
                    sub_chunk.metadata['chunking_method'] = "markdown"
                    chunks.append(sub_chunk)
                    chunk_index += 1
        
        return chunks
    
    def smart_chunk(
        self,
        text: str,
        document_id: str = "doc",
        document_type: str = "text"
    ) -> List[Chunk]:
        """
        Intelligently choose chunking strategy based on document type.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            document_type: Type of document (text, markdown, code)
        
        Returns:
            List of chunks
        """
        if document_type == "markdown":
            return self.chunk_markdown(text, document_id)
        elif document_type == "code":
            # For code, preserve structure better
            return self.chunk_by_paragraphs(text, document_id)
        else:
            # For general text, prefer sentences
            return self.chunk_by_sentences(text, document_id)


# Test chunking strategies
print("=" * 80)
print("DOCUMENT CHUNKING DEMO")
print("=" * 80)

sample_text = """
Artificial Intelligence (AI) is revolutionizing technology. Machine learning, a subset of AI, 
enables computers to learn from data without explicit programming.

Deep learning uses neural networks with multiple layers. These networks can process vast amounts 
of data and identify complex patterns. Applications include image recognition, natural language 
processing, and autonomous vehicles.

The field continues to evolve rapidly. Researchers are developing more efficient algorithms and 
architectures. Ethical considerations around AI deployment are increasingly important.
"""

chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

strategies = [
    ("Tokens", chunker.chunk_by_tokens),
    ("Sentences", chunker.chunk_by_sentences),
    ("Paragraphs", chunker.chunk_by_paragraphs),
]

for strategy_name, strategy_func in strategies:
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy_name}")
    print('='*80)
    
    chunks = strategy_func(sample_text, "demo_doc")
    
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Tokens: {chunk.metadata['token_count']}")
        print(f"  Content: {chunk.content[:100]}...")
```

---

## Part 2: Building the RAG Pipeline

### Complete RAG Pipeline Implementation

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RAGDocument:
    """Document in the RAG system."""
    id: str
    content: str
    chunks: List[Chunk]
    metadata: Dict[str, Any]


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.
    """
    
    def __init__(
        self,
        client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-3.5-turbo",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            client: OpenAI client
            embedding_model: Model for embeddings
            generation_model: Model for generation
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.client = client
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.documents: Dict[str, RAGDocument] = {}
        self.chunks: List[Chunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding)
    
    def _embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks in batch."""
        texts = [chunk.content.replace("\n", " ") for chunk in chunks]
        
        # Batch embedding (max 2048 texts per request)
        embeddings = []
        batch_size = 2048
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def ingest_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "sentences"
    ) -> RAGDocument:
        """
        Ingest a document into the RAG system.
        
        Args:
            content: Document content
            document_id: Unique document ID
            metadata: Optional metadata
            chunking_strategy: Strategy for chunking
        
        Returns:
            RAGDocument object
        """
        print(f"Ingesting document: {document_id}")
        
        # Chunk document
        if chunking_strategy == "tokens":
            chunks = self.chunker.chunk_by_tokens(content, document_id)
        elif chunking_strategy == "sentences":
            chunks = self.chunker.chunk_by_sentences(content, document_id)
        elif chunking_strategy == "paragraphs":
            chunks = self.chunker.chunk_by_paragraphs(content, document_id)
        elif chunking_strategy == "markdown":
            chunks = self.chunker.chunk_markdown(content, document_id)
        else:
            chunks = self.chunker.smart_chunk(content, document_id)
        
        print(f"  Created {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"  Generating embeddings...")
        embeddings = self._embed_chunks(chunks)
        
        # Store embeddings with chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.metadata.update(metadata or {})
        
        # Create document
        doc = RAGDocument(
            id=document_id,
            content=content,
            chunks=chunks,
            metadata=metadata or {}
        )
        
        # Store document
        self.documents[document_id] = doc
        self.chunks.extend(chunks)
        
        # Rebuild embeddings matrix
        self._rebuild_embeddings()
        
        print(f"✓ Document ingested successfully")
        return doc
    
    def ingest_documents(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        chunking_strategy: str = "sentences"
    ) -> List[RAGDocument]:
        """
        Ingest multiple documents.
        
        Args:
            documents: List of (content, doc_id, metadata) tuples
            chunking_strategy: Strategy for chunking
        
        Returns:
            List of RAGDocument objects
        """
        rag_docs = []
        for content, doc_id, metadata in documents:
            doc = self.ingest_document(
                content, doc_id, metadata, chunking_strategy
            )
            rag_docs.append(doc)
        return rag_docs
    
    def _rebuild_embeddings(self):
        """Rebuild embeddings matrix from all chunks."""
        if not self.chunks:
            self.chunk_embeddings = None
            return
        
        self.chunk_embeddings = np.array([chunk.embedding for chunk in self.chunks])
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
        
        Returns:
            List of (Chunk, similarity_score) tuples
        """
        if not self.chunks:
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Filter chunks by metadata if needed
        if filter_metadata:
            filtered_chunks = [
                chunk for chunk in self.chunks
                if all(chunk.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
            if not filtered_chunks:
                return []
            
            filtered_embeddings = np.array([chunk.embedding for chunk in filtered_chunks])
        else:
            filtered_chunks = self.chunks
            filtered_embeddings = self.chunk_embeddings
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with scores
        results = [
            (filtered_chunks[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return results
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
        max_context_tokens: int = 2000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks with scores
            max_context_tokens: Maximum tokens for context
            temperature: Generation temperature
            system_prompt: Optional system prompt
        
        Returns:
            Dictionary with answer and metadata
        """
        # Assemble context from chunks
        context_parts = []
        total_tokens = 0
        used_chunks = []
        
        for chunk, score in retrieved_chunks:
            chunk_tokens = chunk.metadata.get('token_count', 0)
            
            if total_tokens + chunk_tokens > max_context_tokens:
                break
            
            context_parts.append(chunk.content)
            total_tokens += chunk_tokens
            used_chunks.append((chunk, score))
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer the question, say so.
Always cite specific information from the context when possible."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'context': context,
            'chunks_used': len(used_chunks),
            'context_tokens': total_tokens,
            'sources': [
                {
                    'chunk_id': chunk.chunk_id,
                    'document_id': chunk.document_id,
                    'score': score,
                    'content': chunk.content[:200] + "..."
                }
                for chunk, score in used_chunks
            ],
            'model': self.generation_model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve + generate.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            temperature: Generation temperature
        
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve
        retrieved = self.retrieve(question, top_k, filter_metadata)
        
        if not retrieved:
            return {
                'answer': "I don't have any relevant information to answer this question.",
                'context': '',
                'chunks_used': 0,
                'sources': []
            }
        
        # Generate
        result = self.generate(question, retrieved, temperature=temperature)
        
        return result


# Test the RAG pipeline
print("\n" + "=" * 80)
print("RAG PIPELINE DEMO")
print("=" * 80)

rag = RAGPipeline(client)

# Sample documents
documents = [
    ("""Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve 
from experience without being explicitly programmed. The core idea is to develop algorithms that 
can access data and use it to learn for themselves.

There are three main types of machine learning:
1. Supervised Learning: The algorithm learns from labeled training data
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
3. Reinforcement Learning: The algorithm learns through interaction with an environment

Common applications include image recognition, natural language processing, and recommendation systems.""",
     "ml_fundamentals",
     {"category": "ai", "topic": "machine_learning"}),
    
    ("""Deep Learning and Neural Networks

Deep learning is a subset of machine learning based on artificial neural networks. These networks 
are inspired by the structure and function of the human brain, consisting of layers of interconnected 
nodes (neurons).

Key architectures include:
- Convolutional Neural Networks (CNNs): Excellent for image processing
- Recurrent Neural Networks (RNNs): Good for sequential data
- Transformers: State-of-the-art for natural language processing

Deep learning has achieved remarkable results in computer vision, speech recognition, and language 
translation tasks.""",
     "deep_learning",
     {"category": "ai", "topic": "deep_learning"}),
    
    ("""Natural Language Processing

Natural Language Processing (NLP) is a field of AI focused on the interaction between computers 
and human language. It combines computational linguistics with machine learning and deep learning.

Key NLP tasks include:
- Text classification and sentiment analysis
- Named entity recognition
- Machine translation
- Question answering
- Text summarization

Modern NLP relies heavily on transformer models like BERT and GPT, which have achieved human-level 
performance on many benchmarks.""",
     "nlp_overview",
     {"category": "ai", "topic": "nlp"})
]

# Ingest documents
print("\nIngesting documents...")
for content, doc_id, metadata in documents:
    rag.ingest_document(content, doc_id, metadata)

# Test queries
queries = [
    "What are the three types of machine learning?",
    "How do neural networks work?",
    "What is NLP used for?",
]

for query in queries:
    print(f"\n{'='*80}")
    print(f"Question: {query}")
    print('='*80)
    
    result = rag.query(query, top_k=3)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources used: {result['chunks_used']} chunks")
    print(f"Context tokens: {result['context_tokens']}")
    
    print("\nTop sources:")
    for i, source in enumerate(result['sources'][:2], 1):
        print(f"{i}. [{source['score']:.3f}] {source['document_id']}")
        print(f"   {source['content']}")
```

---

## Part 3: Context Assembly

### Advanced Context Assembly Techniques

```python
class ContextAssembler:
    """
    Advanced context assembly for RAG systems.
    """
    
    def __init__(self, tokenizer: tiktoken.Encoding):
        self.tokenizer = tokenizer
    
    def assemble_with_deduplication(
        self,
        chunks: List[Tuple[Chunk, float]],
        max_tokens: int = 2000
    ) -> str:
        """
        Assemble context while removing duplicate information.
        
        Args:
            chunks: Retrieved chunks with scores
            max_tokens: Maximum tokens for context
        
        Returns:
            Assembled context string
        """
        seen_content = set()
        context_parts = []
        total_tokens = 0
        
        for chunk, score in chunks:
            # Simple deduplication: check if substantial overlap
            chunk_sentences = set(re.split(r'[.!?]+', chunk.content.lower()))
            
            # Check overlap with existing content
            overlap = len(chunk_sentences & seen_content)
            overlap_ratio = overlap / len(chunk_sentences) if chunk_sentences else 0
            
            # Skip if more than 50% overlap
            if overlap_ratio > 0.5:
                continue
            
            chunk_tokens = len(self.tokenizer.encode(chunk.content))
            
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk.content)
            total_tokens += chunk_tokens
            seen_content.update(chunk_sentences)
        
        return "\n\n".join(context_parts)
    
    def assemble_with_reranking(
        self,
        chunks: List[Tuple[Chunk, float]],
        query: str,
        max_tokens: int = 2000
    ) -> Tuple[str, List[Tuple[Chunk, float]]]:
        """
        Assemble context with reranking based on query relevance.
        
        Args:
            chunks: Retrieved chunks with scores
            query: Original query
            max_tokens: Maximum tokens for context
        
        Returns:
            Tuple of (assembled context, reranked chunks)
        """
        # Extract query keywords
        query_keywords = set(re.findall(r'\w+', query.lower()))
        
        # Rerank chunks
        reranked = []
        for chunk, semantic_score in chunks:
            # Calculate keyword overlap score
            chunk_words = set(re.findall(r'\w+', chunk.content.lower()))
            keyword_overlap = len(query_keywords & chunk_words) / len(query_keywords)
            
            # Combine scores (70% semantic, 30% keyword)
            combined_score = 0.7 * semantic_score + 0.3 * keyword_overlap
            
            reranked.append((chunk, combined_score))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Assemble context
        context_parts = []
        total_tokens = 0
        used_chunks = []
        
        for chunk, score in reranked:
            chunk_tokens = len(self.tokenizer.encode(chunk.content))
            
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk.content)
            total_tokens += chunk_tokens
            used_chunks.append((chunk, score))
        
        context = "\n\n".join(context_parts)
        return context, used_chunks
    
    def assemble_with_document_structure(
        self,
        chunks: List[Tuple[Chunk, float]],
        max_tokens: int = 2000
    ) -> str:
        """
        Assemble context preserving document structure.
        
        Args:
            chunks: Retrieved chunks with scores
            max_tokens: Maximum tokens for context
        
        Returns:
            Assembled context string with structure
        """
        # Group chunks by document
        doc_chunks: Dict[str, List[Tuple[Chunk, float]]] = {}
        for chunk, score in chunks:
            doc_id = chunk.document_id
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append((chunk, score))
        
        # Sort chunks within each document by position
        for doc_id in doc_chunks:
            doc_chunks[doc_id].sort(key=lambda x: x[0].chunk_index)
        
        # Assemble context document by document
        context_parts = []
        total_tokens = 0
        
        for doc_id, doc_chunk_list in doc_chunks.items():
            doc_parts = []
            doc_tokens = 0
            
            # Add document header
            header = f"--- Document: {doc_id} ---"
            header_tokens = len(self.tokenizer.encode(header))
            
            if total_tokens + header_tokens > max_tokens:
                break
            
            doc_parts.append(header)
            doc_tokens += header_tokens
            
            # Add chunks
            for chunk, score in doc_chunk_list:
                chunk_tokens = len(self.tokenizer.encode(chunk.content))
                
                if total_tokens + doc_tokens + chunk_tokens > max_tokens:
                    break
                
                doc_parts.append(chunk.content)
                doc_tokens += chunk_tokens
            
            context_parts.extend(doc_parts)
            total_tokens += doc_tokens
        
        return "\n\n".join(context_parts)


# Test context assembly
assembler = ContextAssembler(tiktoken.get_encoding("cl100k_base"))

# Use chunks from previous retrieval
retrieved = rag.retrieve("How does deep learning work?", top_k=5)

print("\n" + "=" * 80)
print("CONTEXT ASSEMBLY TECHNIQUES")
print("=" * 80)

# Standard assembly
standard_context = "\n\n".join([chunk.content for chunk, _ in retrieved[:3]])
print(f"\nStandard assembly: {len(standard_context)} characters")

# With deduplication
dedup_context = assembler.assemble_with_deduplication(retrieved)
print(f"With deduplication: {len(dedup_context)} characters")

# With reranking
reranked_context, reranked_chunks = assembler.assemble_with_reranking(
    retrieved, "How does deep learning work?"
)
print(f"With reranking: {len(reranked_context)} characters")
print(f"Reranked order: {[c.document_id for c, _ in reranked_chunks]}")

# With document structure
structured_context = assembler.assemble_with_document_structure(retrieved)
print(f"With document structure: {len(structured_context)} characters")
```

---

## Part 4: Complete RAG System

### Enhanced RAG System with All Features

```python
class EnhancedRAGSystem:
    """
    Production-ready RAG system with all enhancements.
    """
    
    def __init__(
        self,
        client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-3.5-turbo",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.pipeline = RAGPipeline(
            client,
            embedding_model,
            generation_model,
            chunk_size,
            chunk_overlap
        )
        self.assembler = ContextAssembler(tiktoken.get_encoding("cl100k_base"))
        
        # Query history for refinement
        self.query_history: List[Dict[str, Any]] = []
    
    def ingest(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "smart"
    ):
        """Ingest a document."""
        return self.pipeline.ingest_document(
            content, document_id, metadata, chunking_strategy
        )
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        use_deduplication: bool = True,
        temperature: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced query with all features.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            use_reranking: Enable reranking
            use_deduplication: Enable deduplication
            temperature: Generation temperature
            filters: Metadata filters
        
        Returns:
            Enhanced query result
        """
        # Retrieve chunks
        retrieved = self.pipeline.retrieve(question, top_k=top_k * 2, filter_metadata=filters)
        
        if not retrieved:
            return {
                'answer': "I don't have any relevant information to answer this question.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Apply enhancements
        if use_reranking:
            context, used_chunks = self.assembler.assemble_with_reranking(
                retrieved, question
            )
        elif use_deduplication:
            context = self.assembler.assemble_with_deduplication(retrieved)
            used_chunks = retrieved[:top_k]
        else:
            context = "\n\n".join([c.content for c, _ in retrieved[:top_k]])
            used_chunks = retrieved[:top_k]
        
        # Generate answer
        result = self.pipeline.generate(question, used_chunks, temperature=temperature)
        
        # Calculate confidence based on retrieval scores
        avg_score = np.mean([score for _, score in used_chunks])
        result['confidence'] = float(avg_score)
        
        # Store query
        self.query_history.append({
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'chunks_used': len(used_chunks)
        })
        
        return result
    
    def conversational_query(
        self,
        question: str,
        conversation_history: List[Dict[str, str]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query with conversation context.
        
        Args:
            question: Current question
            conversation_history: Previous conversation
            top_k: Number of chunks to retrieve
        
        Returns:
            Query result
        """
        # Retrieve chunks
        retrieved = self.pipeline.retrieve(question, top_k=top_k)
        
        if not retrieved:
            return {
                'answer': "I don't have any relevant information to answer this question.",
                'sources': []
            }
        
        # Assemble context
        context = "\n\n".join([chunk.content for chunk, _ in retrieved])
        
        # Build messages with conversation history
        messages = [
            {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided context.
Use the conversation history to maintain continuity in your responses."""}
        ]
        
        # Add conversation history
        for msg in conversation_history[-5:]:  # Last 5 turns
            messages.append(msg)
        
        # Add current question with context
        messages.append({
            "role": "user",
            "content": f"""Context:
{context}

Question: {question}"""
        })
        
        # Generate response
        response = self.pipeline.client.chat.completions.create(
            model=self.pipeline.generation_model,
            messages=messages,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'context': context,
            'chunks_used': len(retrieved),
            'sources': [
                {'document_id': chunk.document_id, 'score': score}
                for chunk, score in retrieved
            ]
        }


# Test enhanced RAG system
print("\n" + "=" * 80)
print("ENHANCED RAG SYSTEM")
print("=" * 80)

enhanced_rag = EnhancedRAGSystem(client)

# Ingest documents
for content, doc_id, metadata in documents:
    enhanced_rag.ingest(content, doc_id, metadata)

# Test enhanced query
question = "What are the main types of machine learning and neural network architectures?"

result = enhanced_rag.query(
    question,
    top_k=5,
    use_reranking=True,
    use_deduplication=True
)

print(f"\nQuestion: {question}")
print(f"\nAnswer: {result['answer']}")
print(f"\nConfidence: {result['confidence']:.3f}")
print(f"Chunks used: {result['chunks_used']}")
print(f"Total tokens: {result['usage']['total_tokens']}")

# Test conversational query
print(f"\n{'='*80}")
print("CONVERSATIONAL QUERY")
print('='*80)

conversation = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."}
]

follow_up = "Can you give me examples of its applications?"

result = enhanced_rag.conversational_query(follow_up, conversation)
print(f"\nFollow-up: {follow_up}")
print(f"Answer: {result['answer']}")
```

---

## Part 5: RAG Evaluation

### Evaluation Metrics for RAG Systems

```python
from typing import List, Dict, Tuple
import numpy as np

class RAGEvaluator:
    """
    Evaluate RAG system performance.
    """
    
    def __init__(self, rag_system: EnhancedRAGSystem):
        self.rag = rag_system
    
    def evaluate_retrieval(
        self,
        test_cases: List[Tuple[str, List[str]]],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            test_cases: List of (query, relevant_doc_ids) tuples
            top_k: Number of chunks to retrieve
        
        Returns:
            Evaluation metrics
        """
        recall_at_k = []
        precision_at_k = []
        mrr_scores = []  # Mean Reciprocal Rank
        
        for query, relevant_doc_ids in test_cases:
            # Retrieve chunks
            retrieved = self.rag.pipeline.retrieve(query, top_k=top_k)
            retrieved_doc_ids = [chunk.document_id for chunk, _ in retrieved]
            
            # Calculate recall@k
            relevant_set = set(relevant_doc_ids)
            retrieved_set = set(retrieved_doc_ids)
            recall = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0
            recall_at_k.append(recall)
            
            # Calculate precision@k
            precision = len(relevant_set & retrieved_set) / len(retrieved_set) if retrieved_set else 0
            precision_at_k.append(precision)
            
            # Calculate MRR
            reciprocal_rank = 0
            for i, doc_id in enumerate(retrieved_doc_ids, 1):
                if doc_id in relevant_set:
                    reciprocal_rank = 1 / i
                    break
            mrr_scores.append(reciprocal_rank)
        
        return {
            f'recall@{top_k}': np.mean(recall_at_k),
            f'precision@{top_k}': np.mean(precision_at_k),
            f'mrr@{top_k}': np.mean(mrr_scores),
            'f1_score': 2 * np.mean(precision_at_k) * np.mean(recall_at_k) / 
                       (np.mean(precision_at_k) + np.mean(recall_at_k))
                       if (np.mean(precision_at_k) + np.mean(recall_at_k)) > 0 else 0
        }
    
    def evaluate_generation(
        self,
        test_cases: List[Tuple[str, str]],
        client: OpenAI
    ) -> Dict[str, float]:
        """
        Evaluate generation quality using GPT-4.
        
        Args:
            test_cases: List of (query, expected_answer) tuples
            client: OpenAI client
        
        Returns:
            Evaluation metrics
        """
        relevance_scores = []
        factual_scores = []
        completeness_scores = []
        
        for query, expected_answer in test_cases:
            # Get RAG answer
            result = self.rag.query(query)
            generated_answer = result['answer']
            
            # Evaluate with GPT-4
            eval_prompt = f"""Evaluate this question-answer pair on a scale of 1-10:

Question: {query}

Expected Answer: {expected_answer}

Generated Answer: {generated_answer}

Rate the following:
1. Relevance (1-10): How relevant is the answer to the question?
2. Factual Accuracy (1-10): How factually correct is the answer?
3. Completeness (1-10): How complete is the answer?

Respond in JSON format: {{"relevance": X, "factual": Y, "completeness": Z}}"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0
            )
            
            try:
                scores = eval(response.choices[0].message.content)
                relevance_scores.append(scores['relevance'])
                factual_scores.append(scores['factual'])
                completeness_scores.append(scores['completeness'])
            except:
                continue
        
        return {
            'avg_relevance': np.mean(relevance_scores),
            'avg_factual_accuracy': np.mean(factual_scores),
            'avg_completeness': np.mean(completeness_scores),
            'overall_score': np.mean([
                np.mean(relevance_scores),
                np.mean(factual_scores),
                np.mean(completeness_scores)
            ])
        }


# Test evaluation
print("\n" + "=" * 80)
print("RAG EVALUATION")
print("=" * 80)

evaluator = RAGEvaluator(enhanced_rag)

# Test retrieval evaluation
retrieval_test_cases = [
    ("What is machine learning?", ["ml_fundamentals"]),
    ("Explain neural networks", ["deep_learning"]),
    ("What is NLP?", ["nlp_overview"]),
]

retrieval_metrics = evaluator.evaluate_retrieval(retrieval_test_cases, top_k=3)
print("\nRetrieval Metrics:")
for metric, value in retrieval_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

---

## Part 6: Optimization & Best Practices

### Best Practices Summary

```python
"""
RAG SYSTEM BEST PRACTICES:

1. CHUNKING STRATEGY:
   - Choose based on document structure
   - Sentences: General text (preserves semantic units)
   - Paragraphs: Well-structured documents
   - Tokens: Code, unstructured text
   - Markdown: Documentation
   - Chunk size: 300-800 tokens (balance context vs precision)
   - Overlap: 10-20% of chunk size

2. RETRIEVAL OPTIMIZATION:
   - Use top_k = 5-10 for initial retrieval
   - Apply reranking for better relevance
   - Implement hybrid search (semantic + keyword)
   - Add metadata filters for scoped search
   - Cache embeddings aggressively

3. CONTEXT ASSEMBLY:
   - Deduplicate similar chunks
   - Preserve document structure when relevant
   - Limit context to model's optimal window
   - Prioritize by relevance score
   - Include source citations

4. GENERATION QUALITY:
   - Use clear system prompts
   - Set temperature based on task:
     * 0.0-0.3: Factual, precise answers
     * 0.5-0.7: Balanced creativity/accuracy
     * 0.8-1.0: Creative, varied responses
   - Instruct model to cite sources
   - Handle "I don't know" gracefully

5. EVALUATION:
   - Track retrieval metrics (recall, precision, MRR)
   - Measure generation quality (relevance, accuracy, completeness)
   - Monitor latency and costs
   - Collect user feedback
   - A/B test improvements

6. SCALABILITY:
   - Use vector databases (Pinecone, Weaviate) for production
   - Batch embed documents
   - Implement caching at multiple levels
   - Use async operations
   - Shard large document collections

7. COST OPTIMIZATION:
   - Cache embeddings (biggest cost)
   - Use smaller models where appropriate
   - Optimize chunk size (fewer chunks = fewer embeddings)
   - Batch operations
   - Monitor token usage

8. ERROR HANDLING:
   - Handle empty retrievals
   - Validate chunk quality
   - Retry failed embeddings
   - Graceful degradation
   - Log errors for debugging

9. MONITORING:
   - Query latency (p50, p95, p99)
   - Retrieval quality over time
   - Generation quality
   - Cost per query
   - User satisfaction metrics

10. SECURITY:
    - Implement access control
    - Filter results by user permissions
    - Sanitize queries
    - Audit sensitive queries
    - Protect PII in context
"""

print("""
✓ Lab 2 Complete!

Key Takeaways:
- Document chunking is critical for RAG quality
- Retrieval-augmented generation combines search with LLM generation
- Context assembly affects answer quality and costs
- Evaluation is essential for continuous improvement
- Production systems need caching, monitoring, and optimization

Next Steps:
- Proceed to Lab 3: Enterprise RAG System
- Experiment with different chunking strategies
- Try hybrid search approaches
- Build comprehensive evaluation pipelines
- Deploy with a production vector database
""")
```

