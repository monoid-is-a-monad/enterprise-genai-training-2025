# Week 5 - Lesson 3: Query Rewriting, Re-ranking, and Evaluation at Scale

**Duration:** 90 minutes  
**Level:** Advanced  
**Prerequisites:** Week 5 Lessons 1–2, Week 4 RAG fundamentals

---

## 🎯 Learning Objectives

By the end of this lesson, you will:
- [ ] Implement query rewriting (HyDE, multi-query, step-back) to maximize recall
- [ ] Apply robust re-ranking with cross-encoders or LLMs, including JSON outputs
- [ ] Build scalable offline evaluation harnesses for retrieval and answer quality
- [ ] Set guardrails (bias, leakage, safety) and deploy with A/B and canary gates

---

## 📚 Table of Contents

1. [Introduction](#1-introduction)
2. [Query Rewriting Patterns](#2-query-rewriting-patterns)
3. [Re-ranking Strategies](#3-re-ranking-strategies)
4. [Evaluation at Scale](#4-evaluation-at-scale)
5. [Operationalizing](#5-operationalizing)
6. [Practical Examples](#6-practical-examples)
7. [Best Practices](#7-best-practices)
8. [Common Pitfalls](#8-common-pitfalls)
9. [Summary](#9-summary)
10. [Further Reading](#10-further-reading)

---

## 1. Introduction

We now combine rewriting and re-ranking to systematically improve retrieval quality and answer fidelity. The key is to measure, iterate, and control latency/cost.

### Pipeline Overview

```mermaid
graph TB
    Q[User Query] --> RW[Rewriting: HyDE / Multi-Query / Step-back]
    RW --> D[Dense Search]
    RW --> Lx[BM25 Search]
    D --> F[Score Fusion]
    Lx --> F
    F --> RR[Re-ranking]
    RR --> S[Selected Context]
    S --> G[LLM Generation]
    G --> E[Evaluation (Offline/Online)]
```

---

## 2. Query Rewriting Patterns

### 2.1 HyDE (Hypothetical Document Embeddings)
- Generate a short factual draft answer. Embed that text to retrieve passages.
- Works well for underspecified or high-level questions.

### 2.2 Multi-Query Paraphrasing
- Create 3–5 diverse paraphrases; union and fuse retrieval results.
- Improves robustness to phrasing and vocabulary mismatch.

### 2.3 Step-back Prompting
- Ask: “What higher-level concept would best answer this?” Retrieve on the abstraction; then refine.
- Helps when users ask overly specific or tangential questions.

### 2.4 Safety and Scope Control
- Constrain rewriting prompts to remain faithful to the user’s intent (avoid topic drift).
- Log rewritten forms for audits.

---

## 3. Re-ranking Strategies

### 3.1 Cross-encoder Re-ranking
- Encode [query, passage] jointly for precise relevance scoring. Apply to top 50–100.
- Keep the best 8–12 passages for augmentation.

### 3.2 LLM Re-ranking
- Use an LLM to score passages and return JSON with {index, score, reason}.
- Employ a cheaper/different model than the answer model to reduce leakage bias.

### 3.3 Heuristics and Diversity
- Dedupe near-duplicates, boost freshness/authority, consider domain priors.
- Use MMR pre- or post-fusion to maintain diversity.

---

## 4. Evaluation at Scale

### 4.1 Ground Truth and Datasets
- Build a canary set: 100–300 queries, with relevant doc_ids (graded if possible) and gold answers.
- Cover common intents and edge cases (IDs, tables, code, legal).

### 4.2 Retrieval Metrics
- Recall@k, Precision@k, nDCG@k, MRR, coverage.
- Report p50/p95 latency per stage and total token/cost.

### 4.3 Answer Quality Metrics
- Correctness, faithfulness, completeness, conciseness, citation accuracy.
- LLM-based judging with carefully designed prompts and held-out models.

### 4.4 Experimentation and Gates
- A/B or interleaving; online canary traffic with rollbacks.
- Promotion requires exceeding baseline thresholds (e.g., recall@10, p95 latency).

---

## 5. Operationalizing

- Separate models: rewriting vs re-ranking vs answering.
- Cache embeddings and fused retrieval results.
- Feature logging: ranks, scores, chosen ids, params; attach runs to PRs.
- Guardrails: max tokens, strict citation requirements, PII redaction.

---

## 6. Practical Examples

### 6.1 Rewriting: HyDE, Multi-Query, Step-back

```python
from typing import List
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def hyde(question: str) -> str:
    sys = "Write a concise, factual draft answer to help retrieve evidence."
    out = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Question: {question}\nAnswer in 4-6 sentences."}
        ],
        temperature=0.2,
    )
    return out.choices[0].message.content


def multi_query(question: str, n: int = 3) -> List[str]:
    sys = "Paraphrase into diverse queries that preserve intent."
    out = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Paraphrase into {n} concise queries: {question}"}
        ],
        temperature=0.5,
    )
    text = out.choices[0].message.content
    return [q.strip("- ") for q in text.split("\n") if q.strip()]


def step_back(question: str) -> str:
    sys = "Rewrite the question at a higher level to retrieve better context; preserve intent."
    out = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Original: {question}\nReturn a single higher-level query."}
        ],
        temperature=0.3,
    )
    return out.choices[0].message.content.strip()
```

### 6.2 LLM Re-ranking Pattern (JSON)

```python
from typing import List, Tuple, Dict


def llm_rerank(question: str, candidates: List[str]) -> List[Tuple[int, float]]:
    sys = (
        "Rank passages by relevance to the query. Return JSON list of objects: "
        "{index: <int>, score: <0..1>, reason: <string>} sorted by score desc."
    )
    user = "Query: " + question + "\nCandidates:\n" + "\n".join(
        f"{i}) {c}" for i, c in enumerate(candidates)
    )
    out = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    text = out.choices[0].message.content
    # TODO: robust JSON parsing and validation here
    # return list of (index, score)
    return []
```

### 6.3 Offline Evaluation Harness (Skeleton)

```python
from typing import Callable, Dict, List, Tuple
import time, math

# search_fn: (query_text or vec, k, **params) -> List[Doc(id, score)]
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


def evaluate(
    queries: List[Tuple[str, str]],  # (qid, question)
    search_fn: Callable,
    k: int,
    params: Dict,
    ground_truth: Dict[str, set[str]]
):
    prec, rec, rr, lat = [], [], [], []
    for qid, qtext in queries:
        t0 = time.perf_counter()
        docs = search_fn(qtext, k=k, **params)
        lat.append((time.perf_counter()-t0)*1000)
        ids = [d.id for d in docs]
        p, r = precision_recall_at_k(ids, ground_truth[qid], k)
        prec.append(p); rec.append(r); rr.append(mrr(ids, ground_truth[qid]))
    p50 = sorted(lat)[len(lat)//2]
    p95 = sorted(lat)[int(len(lat)*0.95)]
    return {
        'params': params,
        'precision@k': sum(prec)/len(prec),
        'recall@k': sum(rec)/len(rec),
        'mrr': sum(rr)/len(rr),
        'latency_p50_ms': p50,
        'latency_p95_ms': p95,
    }
```

---

## 7. Best Practices

- Retrieve wide → re-rank narrow → compress for token budget with citation preservation.
- Separate models for rewriting, re-ranking, and answering.
- Normalize scores before fusion; cache and log everything needed to reproduce runs.
- Tune with a stable canary set and promote only when surpassing baselines.

---

## 8. Common Pitfalls

- Over-aggressive rewriting that changes user intent.
- Using the same LLM instance for re-ranking and answering (leakage).
- Skipping normalization before fusion or omitting MMR, causing redundancy.
- No canary gates: silent recall losses ship to production.

---

## 9. Summary

You can now combine rewriting and re-ranking to improve retrieval systematically and measure the impact at scale with robust harnesses and promotion gates.

---

## 10. Further Reading

- HyDE, RRF, MMR literature; cross-encoder re-ranking papers (MS MARCO variants)
- Week 4 resources on hybrid retrieval and evaluation

---

## Repo Resources

- Week 5 Resources Index: [../resources/README.md](../resources/README.md)
- Hybrid Retrieval Tuning: [../resources/hybrid-retrieval-tuning.md](../resources/hybrid-retrieval-tuning.md)
- Recall vs Latency Evaluation: [../resources/recall-vs-latency-evaluation.md](../resources/recall-vs-latency-evaluation.md)
- Index Tuning Cheatsheet: [../resources/index-tuning-cheatsheet.md](../resources/index-tuning-cheatsheet.md)
- Week 4 Hybrid Retrieval & Re-ranking: [../../week-04/resources/hybrid-reranking.md](../../week-04/resources/hybrid-reranking.md)
- Week 4 Evaluation Metrics: [../../week-04/resources/evaluation-metrics.md](../../week-04/resources/evaluation-metrics.md)
