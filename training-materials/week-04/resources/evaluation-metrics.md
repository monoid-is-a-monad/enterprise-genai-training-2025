# Evaluation Metrics for RAG

A practical guide to measuring retrieval and answer quality.

## Retrieval Metrics
- Recall@k: Of all truly relevant chunks, how many are in top-k?
- Precision@k: Of top-k retrieved chunks, how many are truly relevant?
- MRR (Mean Reciprocal Rank): Average of 1/rank of the first relevant item
- nDCG (Normalized Discounted Cumulative Gain): Position-weighted relevance with ideal normalization
- Coverage: % of queries where at least one relevant chunk retrieved

### Quick formulas
- Recall@k = (# relevant in top-k) / (# total relevant)
- Precision@k = (# relevant in top-k) / k
- MRR = mean(1 / rank_first_relevant)
- nDCG@k = DCG@k / IDCG@k, DCG@k = Σ (rel_i / log2(i+1))

## Answer Quality Metrics
- Correctness: Does the answer match the gold answer?
- Faithfulness: Is the answer supported by retrieved context? (no hallucinations)
- Completeness: Did the answer cover key points?
- Conciseness: Was it succinct and within budget?
- Citation accuracy: Do cited sources actually support the answer?

## Latency & Cost
- Retrieval latency (p50/p95)
- Generation latency (p50/p95)
- Total time per query
- Token usage (prompt + completion)
- Cost per query/session

## Minimal Evaluation Harness (Python)
```python
from typing import List, Dict, Tuple
import math

# Ground truth format: {query: {relevant_ids: set([doc_ids]), answer: str}}

def precision_recall_at_k(retrieved: List[str], relevant: set[str], k: int) -> Tuple[float, float]:
    topk = retrieved[:k]
    hits = sum(1 for x in topk if x in relevant)
    precision = hits / max(1, k)
    recall = hits / max(1, len(relevant))
    return precision, recall


def mrr(retrieved: List[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(rels: List[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    # relevant is a dict of {doc_id: graded_relevance}
    rels = [relevant.get(doc_id, 0) for doc_id in retrieved]
    dcg = dcg_at_k(rels, k)
    ideal = sorted(relevant.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# Example usage
retrieved = ["d3", "d8", "d1", "d2"]
relevant_set = {"d1", "d2"}
relevant_graded = {"d1": 3, "d2": 2}

p, r = precision_recall_at_k(retrieved, relevant_set, k=3)
print({"precision@3": p, "recall@3": r})
print({"mrr": mrr(retrieved, relevant_set)})
print({"ndcg@3": ndcg_at_k(retrieved, relevant_graded, 3)})
```

## LLM-based Judging (Faithfulness)
- Prompt an LLM to label answers as Supported/Contradicted/Insufficient
- Provide retrieved context and require JSON output with justification
- Use a separate, cheaper model than the one for generation

```text
System: You are evaluating if an answer is supported by the context.

User:
Question: {q}
Answer: {a}
Context:
1) {c1}
2) {c2}

Return JSON: {"faithfulness": "supported|insufficient|contradicted", "evidence": [1,2], "notes": "..."}
```

## Reporting
- Aggregate by query type, source corpus, user segment
- Track trends over time (drift)
- Create dashboards for p50/p95 latency, recall@k, correctness, faithfulness
- Gate deploys with a baseline A/B harness
