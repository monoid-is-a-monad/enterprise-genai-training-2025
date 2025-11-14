# Recall vs Latency Evaluation Guide

A practical, repeatable way to measure retrieval quality and speed.

## What to Measure
- Retrieval: recall@k, precision@k, nDCG@k, MRR, coverage
- Latency: p50/p95 per stage (embed, search, rerank, generate)
- Cost: tokens, $ per query, cache hit rates
- Stability: stdev across runs; sensitivity to parameter changes

## Canary Set Design
- 100–300 queries representative of production mix (include hard cases)
- For each: ground truth relevant doc_ids (+ graded relevance if possible)
- Gold answers for answer-quality checks (correctness, faithfulness)

## Harness Structure
```python
from typing import Callable, Dict, List, Tuple
import time, math

# Types
# retrievers: dict name -> search_fn(query_vec, k, **params) -> List[Doc(id, score)]
# ground_truth: Dict[str, set[str]]


def precision_recall_at_k(retrieved: List[str], relevant: set[str], k: int):
    topk = retrieved[:k]
    hits = sum(1 for x in topk if x in relevant)
    return hits/max(1,k), hits/max(1,len(relevant))


def mrr(retrieved: List[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0/i
    return 0.0


def dcg_at_k(rels: List[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(retrieved: List[str], relevant: Dict[str,int], k: int) -> float:
    rels = [relevant.get(doc_id, 0) for doc_id in retrieved]
    dcg = dcg_at_k(rels, k)
    ideal = sorted(relevant.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg/idcg if idcg>0 else 0.0


def evaluate(
    queries: List[Tuple[str, list[float]]],
    ground_truth: Dict[str, set[str]],
    search_fn: Callable,
    k: int,
    params: Dict
):
    pr, re, rr, lat = [], [], [], []
    for qid, qvec in queries:
        t0 = time.perf_counter()
        docs = search_fn(qvec, k=k, **params)
        lat.append((time.perf_counter()-t0)*1000)
        ids = [d.id for d in docs]
        p, r = precision_recall_at_k(ids, ground_truth[qid], k)
        pr.append(p); re.append(r); rr.append(mrr(ids, ground_truth[qid]))
    p50 = sorted(lat)[len(lat)//2]
    p95 = sorted(lat)[int(len(lat)*0.95)]
    return {
        'params': params,
        'precision@k': sum(pr)/len(pr),
        'recall@k': sum(re)/len(re),
        'mrr': sum(rr)/len(rr),
        'latency_p50_ms': p50,
        'latency_p95_ms': p95,
    }
```

## Procedure
1) Fix k and sweep retrieval parameters (e.g., ef_search, nprobe, α, RRF k).
2) Add MMR; choose λ for best recall/latency trade-off.
3) Layer re-ranking; pick top-N to keep within token budget.
4) Introduce rewriting (HyDE, multi-query); union and fuse; reassess.
5) Gate changes with thresholds (e.g., recall@10 ≥ baseline, p95 < SLA).

## Reporting
- Aggregate by query class (FAQ, spec, code, legal) and by tenant
- Plot recall@k vs p95 to find knees in the curve
- Keep a living dashboard (Grafana/Datadog/BI) and attach runs to PRs
