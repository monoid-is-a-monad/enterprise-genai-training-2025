# RAG Cheatsheet

A quick reference for building and tuning Retrieval-Augmented Generation systems.

## Pipeline Blocks
- Ingestion: loaders (PDF/MD/TXT), metadata, normalization
- Preprocessing: cleaning, boilerplate removal, language detection
- Chunking: fixed-length, token-aware, markdown/semantic chunks
- Embeddings: model choice, dimensionality, batching
- Indexing: HNSW/IVF, MMR configuration, filters
- Retrieval: top-k, MMR, hybrid (BM25 + dense), filters
- Post-retrieval: re-ranking (LLM or cross-encoder), deduping, compression
- Augmentation: prompt construction, context windows, citation markers
- Generation: system prompts, output schemas, guardrails
- Evaluation: recall@n, faithfulness, answer quality, latency, cost

## Quick Defaults
- Chunk size: 500–1,000 tokens, 10–20% overlap
- Retrieval: k=8–12; MMR lambda=0.3–0.5; diversity enabled
- Hybrid: 60% dense + 40% BM25; normalize scores; reciprocal rank fusion
- Reranking: LLM or cross-encoder on top 20 → keep 5–8
- Citing: reference with [docId:chunkId] and include attribution block

## Prompt Template (Answering)
System: You are a concise, factual assistant. Only answer from the provided context. If missing, say you don’t know.

User: <question>

Context:
- {snippet_1}
- {snippet_2}
- {snippet_3}

Requirements:
- Cite sources with [docId:chunkId].
- If context conflicts, note uncertainty.
- Max 150 words.

## Compression Options
- Summarize chunks with an LLM to fit token budget (retain citations)
- Extract key sentences (TextRank) + named entities
- Remove boilerplate (navigation, headers, footers)

## Troubleshooting
- Low recall: increase k, improve chunking, add BM25, better query rewriting
- Hallucinations: stricter system prompt, require citations, reduce temperature
- Duplicate/near-duplicate context: enable deduping + MMR
- High latency: reduce k, enable caching, stream reranking, pre-compute features
- Cost spikes: batch embeddings, cache results, compress long context
