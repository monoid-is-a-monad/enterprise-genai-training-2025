# Index Tuning Cheatsheet (HNSW / IVF / PQ)

A quick, vendor-agnostic reference to tune vector indexes for RAG.

## HNSW (Graph Index)
- Build-time knobs:
  - M (graph degree): 16–48 typical. Higher → better recall, more RAM.
  - ef_construction: 100–400. Higher → better graph quality.
- Query-time knob:
  - ef_search: 50–400. Higher → higher recall, higher latency.
- Tips:
  - Start with M=32, ef_construction=200, ef_search=100; measure and adjust.
  - Pin hot shards to CPU NUMA to reduce p95 latency.
  - Validate recall with a fixed canary before changing ef_search.

## IVF (Inverted File Index)
- Build-time knob:
  - nlist: number of clusters/buckets (e.g., 1024–16384). Larger → finer partition.
- Query-time knob:
  - nprobe: buckets searched per query (e.g., 8–128). Higher → better recall.
- Tips:
  - Ensure representative training vectors for k-means clustering.
  - Sweep nprobe to find recall/latency knee; log p50/p95.

## PQ / OPQ (Product Quantization)
- Build-time knobs:
  - m (subvectors): e.g., 16–64. More subvectors → less error, more RAM.
  - bits per code: e.g., 8. More bits → higher recall, more RAM.
  - OPQ: learn rotation before PQ to improve code quality.
- Tips:
  - Use OPQ for better accuracy at same code size.
  - Don’t over-compress critical corpora (retain a high-quality tier for fallbacks).

## Filters, Tenancy, Sharding
- Add metadata fields for tenant, doc_type, language, updated_at; index hot fields.
- Isolate tenants via namespaces/collections; shard by tenant/domain; replicate for HA.
- Prefer early filter application to cut candidate sets before re-ranking.

## Practical Defaults
- Corpora ≤ 5M vectors: HNSW first (M=32, ef_c=200, ef_s=100).
- Corpora 5M–100M: IVF+Flat/IVF+PQ with nlist in thousands, nprobe tuned to recall target.
- Memory constrained: IVF+PQ (OPQ) with careful evaluation vs HNSW.

## Vendor Mapping (Indicative)
- FAISS: `IndexHNSWFlat`, `IndexIVFFlat`, `IndexIVFPQ`, `OPQMatrix`
- Qdrant: HNSW params (m, ef_construct), search params (ef)
- Weaviate: HNSW (efConstruction, m), filters (where), vectorIndexConfig
- Pinecone: HNSW/IVF under the hood; control via pod size/replicas/namespaces
- Chroma: HNSW-based by default; tune through client config and environment

## Checklist Before Deploy
- [ ] Canary set covers your common and edge queries
- [ ] Recall@k and nDCG meet baseline at p95 latency budget
- [ ] Delete/TTL flows validated; compaction restores space
- [ ] Freshness SLOs monitored (age of last ingest)
- [ ] Feature logging enabled (scores, ranks, params)
