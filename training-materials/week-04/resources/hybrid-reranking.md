# Hybrid Retrieval and Re-ranking

Combine sparse (BM25) and dense vectors, then re-rank to maximize relevance and faithfulness.

## Retrieval Fusion
- Weighted score fusion: normalize scores per retriever, then `score = α * dense + (1-α) * bm25` (α≈0.6)
- Reciprocal Rank Fusion (RRF): `1 / (k + rank)` summed over retrievers; robust default
- Rank-based voting: count occurrences across retrievers; keep high-agreement docs

## Filter & Diversity
- Apply metadata filters first (doc type, date, language)
- Use MMR to promote diversity among top candidates

## Re-ranking Options
- Cross-encoders (e.g., MS MARCO trained) for passage-level scoring
- LLM re-ranking: ask LLM to label top 20 → select best 5–8
- Heuristics: demote near-duplicates, boost fresher docs

## LLM Re-ranking Prompt (Pattern)
System: You are ranking candidate passages for relevance to a query.

User:
Query: {q}
Candidates:
1) {c1}
2) {c2}
...
N) {cN}

Return JSON:
[
  {"index": 7, "score": 0.92, "reason": "..."},
  ...
]

## Guardrails
- Never re-rank with the same LLM instance used for answering (leakage risk)
- Cache query embeddings and retrieval results to stabilize latency
- Log features for offline evaluation (scores, ranks, chosen ids)
