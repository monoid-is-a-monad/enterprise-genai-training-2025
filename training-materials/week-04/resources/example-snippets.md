# Example Snippets: RAG Building Blocks

Reusable patterns for core RAG operations.

## 1) MMR Selection
```python
from typing import List, Tuple
import numpy as np


def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int = 8,
    lambda_diversity: float = 0.4
) -> List[int]:
    """Maximal Marginal Relevance selection indices."""
    selected = []
    candidates = list(range(len(doc_vecs)))
    query_sim = (doc_vecs @ query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    while candidates and len(selected) < k:
        if not selected:
            best = int(np.argmax(query_sim[candidates]))
            selected.append(candidates.pop(best))
            continue
        selected_vecs = doc_vecs[selected]
        sim_to_selected = selected_vecs @ doc_vecs.T
        sim_to_selected /= (
            np.linalg.norm(selected_vecs, axis=1, keepdims=True)
            * np.linalg.norm(doc_vecs, axis=1)
            + 1e-8
        )
        redundancy = sim_to_selected.max(axis=0)
        scores = lambda_diversity * query_sim - (1 - lambda_diversity) * redundancy
        best = int(np.argmax(scores[candidates]))
        selected.append(candidates.pop(best))
    return selected
```

## 2) Hybrid Retrieval (Dense + BM25)
```python
from typing import Dict

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-8:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse_scores(dense: Dict[str, float], sparse: Dict[str, float], alpha: float = 0.6):
    d = normalize_scores(dense)
    s = normalize_scores(sparse)
    ids = set(d) | set(s)
    return {i: alpha * d.get(i, 0.0) + (1 - alpha) * s.get(i, 0.0) for i in ids}
```

## 3) Reciprocal Rank Fusion
```python
def rrf(rankings: Dict[str, int], k: int = 60) -> float:
    # rankings: {retriever_name: rank_position (1-based)}
    return sum(1.0 / (k + r) for r in rankings.values())
```

## 4) Token Budgeting & Compression
```python
from typing import List


def compress_chunks(chunks: List[str], max_tokens: int, summarize) -> List[str]:
    """Summarize chunks to fit token budget. `summarize` is a callable LLM summarizer."""
    compressed = []
    for ch in chunks:
        if estimate_tokens(ch) <= max_tokens:
            compressed.append(ch)
        else:
            summary = summarize(
                f"Summarize the following to <= {max_tokens} tokens, keep citations: {ch}"
            )
            compressed.append(summary)
    return compressed
```

## 5) LLM Re-ranking Prompt
```text
System: Rank candidate passages for relevance to the query.
User:
Query: {q}
Candidates:
1) {c1}
2) {c2}
...
N) {cN}
Return JSON list of {"index": i, "score": 0-1, "reason": "..."} sorted by score desc.
```

## 6) Faithfulness Check Harness
```python
from typing import Literal


def judge_faithfulness(question: str, answer: str, context: list[str], call_llm) -> dict:
    prompt = f"""
You are evaluating if an answer is supported by the provided context.
Question: {question}
Answer: {answer}
Context:\n- " + "\n- ".join(context) + "
Return JSON with keys: faithfulness (supported|insufficient|contradicted), evidence (indices), notes.
"""
    out = call_llm(prompt)
    return out
```
