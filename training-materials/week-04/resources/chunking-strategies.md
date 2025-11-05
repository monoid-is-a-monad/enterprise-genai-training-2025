# Chunking Strategies for RAG

Well-chosen chunks drive recall, precision, and low latency. Use this as a guide.

## Goals
- Maximize semantic coherence per chunk
- Preserve entity/section boundaries when possible
- Balance retrieval coverage vs. context length

## Common Strategies
1) Fixed-length tokens
- Size: 500–1,000 tokens, 10–20% overlap
- Pros: simple, predictable; Cons: may split concepts

2) Paragraph or heading-aware
- Split on headings (Markdown H1–H3) and paragraphs
- Pros: preserves structure; Cons: uneven sizes

3) Semantic splitter
- Embed sentences, group by similarity until token limit
- Pros: coherence; Cons: preprocessing cost

4) Hybrid
- Start with heading/paragraph, then token-pack to target size

## Overlap Guidance
- Short, dense topics: 10–15%
- Long narratives or references: 15–25%
- Code or formulas: include preceding definitions/examples

## Metadata to Preserve
- doc_id, chunk_id, source path/URL
- section title, hierarchy (h1/h2/h3)
- page number (PDF) or timestamp (transcripts)
- language, version, last-updated

## Quality Checks
- Token histogram; cap >95th percentile via split
- No empty/boilerplate-only chunks
- Validate citations map correctly to original source

## MMR and Diversity
- Use MMR to select top-k diverse chunks
- Lambda 0.3–0.5 often works well

## Pitfalls
- Overlapping too much causes redundancy
- Over-chunking bloats index and cost
- Ignoring structure reduces answer quality
