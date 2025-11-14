# Week 5: Advanced RAG & Vector Databases

**Provided by:** ADC ENGINEERING & CONSULTING LTD

**Duration:** 20 hours

## Overview

This week builds on RAG foundations to cover advanced retrieval strategies, vector database indexing and tuning, hybrid search with re-ranking, and query rewriting techniques for high recall and precision at scale.

## Learning Objectives

- [ ] Design advanced RAG architectures for enterprise needs
- [ ] Tune vector indexes (HNSW/IVF/PQ) for quality/latency trade-offs
- [ ] Implement hybrid retrieval (dense + lexical) with principled fusion
- [ ] Apply re-ranking (cross-encoder/LLM) and context compression
- [ ] Use query rewriting (HyDE, multi-query, step-back) to boost recall
- [ ] Plan for multi-tenant, sharded, and filtered deployments
- [ ] Measure performance with robust metrics and an evaluation harness

## Content Structure

### Lessons
1. Advanced RAG Architectures & Retrieval Strategies — [lessons/01-advanced-rag-architectures-and-retrieval.md](./lessons/01-advanced-rag-architectures-and-retrieval.md)
2. Vector Databases Deep Dive & Index Tuning — [lessons/02-vector-databases-deep-dive-and-index-tuning.md](./lessons/02-vector-databases-deep-dive-and-index-tuning.md)
3. Query Rewriting, Re-ranking, and Evaluation at Scale — [lessons/03-query-rewriting-reranking-evaluation-at-scale.md](./lessons/03-query-rewriting-reranking-evaluation-at-scale.md)
4. Scaling & Operations for Enterprise RAG — [lessons/04-scaling-operations-for-enterprise-rag.md](./lessons/04-scaling-operations-for-enterprise-rag.md)

### Labs
1. Hybrid Retrieval + Re-ranking — [labs/lab-01-hybrid-retrieval-and-reranking.ipynb](./labs/lab-01-hybrid-retrieval-and-reranking.ipynb)
2. Index Tuning and Recall Testing — [labs/lab-02-index-tuning-and-recall-testing.ipynb](./labs/lab-02-index-tuning-and-recall-testing.ipynb)
3. Query Rewriting & Production RAG Patterns — [labs/lab-03-query-rewriting-and-production-rag.ipynb](./labs/lab-03-query-rewriting-and-production-rag.ipynb)

### Exercises
> **Note:** The planned exercises are comprehensively covered in the labs:
> - Fusion Weight Tuning → See Lab 1 (Bonus Challenge: Alpha Parameter Sweep)
> - Query Rewriting Comparison → See Lab 3 (Exercise 4: Query Rewriting Comparison)

## Tools & Libraries (indicative)
```python
openai>=1.0.0
chromadb>=0.4.0  # or pinecone-client, weaviate-client, qdrant-client
faiss-cpu  # if using FAISS locally
rank-bm25  # for lexical baseline (optional)
tiktoken
python-dotenv
```

## Repo Resources
- Resources Index: [resources/README.md](./resources/README.md)
- Index Tuning Cheatsheet: [resources/index-tuning-cheatsheet.md](./resources/index-tuning-cheatsheet.md)
- Hybrid Retrieval Tuning: [resources/hybrid-retrieval-tuning.md](./resources/hybrid-retrieval-tuning.md)
- Recall vs Latency Evaluation: [resources/recall-vs-latency-evaluation.md](./resources/recall-vs-latency-evaluation.md)

## Notes
- Refer to Week 4 resources for refresher material on chunking strategies, hybrid retrieval, and evaluation metrics.
- This week emphasizes offline evaluation and metrics-driven tuning. Ensure you log features for reproducibility.

**Week Coordinator:** Training Team  
**Last Updated:** November 11, 2025
