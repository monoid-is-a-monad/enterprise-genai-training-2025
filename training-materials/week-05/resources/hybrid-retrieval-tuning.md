# Hybrid Retrieval Tuning (Dense + Lexical + Re-ranking)

Dial in retrieval quality with principled fusion, diversity, and rewriting.

## Fusion Strategies
- Weighted score fusion: normalize per retriever → `score = α·dense + (1-α)·bm25` (α≈0.6)
- Reciprocal Rank Fusion (RRF): `Σ 1/(k + rank)`; robust across systems
- Rank voting: count cross-retriever agreement; boost consensus

## Normalization
- Always normalize dense and BM25 scores separately before fusion (min/max or z-score) to avoid one dominating.

## Diversity (MMR)
- Use MMR to reduce redundancy among top-k; λ=0.3–0.5 often strong.
- Apply after fusion, before re-ranking to widen candidate variety.

## Re-ranking Options
- Cross-encoder: highest precision on top-N (e.g., 50 → keep 8–12)
- LLM re-ranking: cheaper LLM than answer model; return JSON {index, score, reason}
- Heuristics: dedupe near-duplicates; boost freshness/authority; demote boilerplate

## Query Rewriting Interplay
- HyDE: generate hypothetical answer → embed → retrieve; improves underspecified queries
- Multi-query: produce 3–5 paraphrases; union and fuse results
- Step-back: retrieve with higher-level abstraction first; then refine

## BM25 & Lexical Tips
- Ensure tokenizer and stopword lists fit your domain (code, legal, medical)
- Tune k1 and b (if exposed) using validation set; default often ok
- Keyword synonym expansion for domain-specific terms

## Practical Defaults
- k (initial): 50–100 combined (dense + bm25)
- Rerank to: 8–12 for augmentation
- α (weighted fusion): 0.6 (sweep 0.4–0.7)
- MMR λ: 0.4 (sweep 0.3–0.6)
- RRF k: 60 (robust default)

## Offline Tuning Procedure
1) Build a canary set of (query → relevant doc_ids) and answer keys.
2) Sweep α or use RRF; record recall@k, ndcg@k, MRR and latency p95.
3) Add MMR; ensure dedupe; re-evaluate.
4) Layer re-ranking; pick top-N sizes to fit token budget.
5) Introduce rewriting (HyDE, multi-query); union and fuse; re-evaluate.

## Guardrails
- Use different LLMs for rewriting, re-ranking, and answering to reduce leakage bias.
- Cache embeddings and retrieval results to stabilize latency and cost.
- Log features (scores, ranks, chosen ids) for audits and A/B analysis.
