"""
Evaluation Harness for Retrieval Quality and Latency

Measures recall@k, precision@k, MRR, nDCG, and latency percentiles.
"""
import json
import time
import math
from typing import List, Dict, Tuple, Callable, Any
from mock_retriever import create_retriever, Document


def precision_recall_at_k(
    retrieved: List[str],
    relevant: set,
    k: int
) -> Tuple[float, float]:
    """
    Calculate precision and recall at k.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        k: Cutoff position
    
    Returns:
        (precision@k, recall@k)
    """
    topk = retrieved[:k]
    hits = sum(1 for doc_id in topk if doc_id in relevant)
    
    precision = hits / max(1, k)
    recall = hits / max(1, len(relevant))
    
    return precision, recall


def mrr(retrieved: List[str], relevant: set) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
    
    Returns:
        Reciprocal rank (0 if no relevant docs found)
    """
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(
    retrieved: List[str],
    relevant_graded: Dict[str, int],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant_graded: Dict mapping doc_id to graded relevance (0-3)
        k: Cutoff position
    
    Returns:
        nDCG@k score (0-1)
    """
    # Get relevance scores for retrieved docs
    rels = [relevant_graded.get(doc_id, 0) for doc_id in retrieved]
    dcg = dcg_at_k(rels, k)
    
    # Calculate ideal DCG
    ideal = sorted(relevant_graded.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    queries: List[Tuple[str, str]],
    ground_truth: Dict[str, set],
    ground_truth_graded: Dict[str, Dict[str, int]],
    search_fn: Callable,
    k: int,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run evaluation on a set of queries.
    
    Args:
        queries: List of (query_id, query_text) tuples
        ground_truth: Dict mapping query_id to set of relevant doc_ids
        ground_truth_graded: Dict mapping query_id to {doc_id: relevance_score}
        search_fn: Search function (query_text, k, **params) -> List[Document]
        k: Number of results to retrieve
        params: Search parameters to pass through
    
    Returns:
        Dict with aggregated metrics
    """
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []
    latencies = []
    
    for query_id, query_text in queries:
        # Measure latency
        t0 = time.perf_counter()
        results = search_fn(query_text, k=k, **params)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        
        # Extract document IDs
        retrieved_ids = [doc.id for doc in results]
        
        # Calculate metrics
        p, r = precision_recall_at_k(
            retrieved_ids,
            ground_truth.get(query_id, set()),
            k
        )
        precisions.append(p)
        recalls.append(r)
        
        mrrs.append(mrr(retrieved_ids, ground_truth.get(query_id, set())))
        
        if query_id in ground_truth_graded:
            ndcgs.append(ndcg_at_k(
                retrieved_ids,
                ground_truth_graded[query_id],
                k
            ))
    
    # Aggregate results
    latencies_sorted = sorted(latencies)
    return {
        'params': params,
        'precision@k': sum(precisions) / len(precisions),
        'recall@k': sum(recalls) / len(recalls),
        'mrr': sum(mrrs) / len(mrrs),
        'ndcg@k': sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        'latency_p50_ms': latencies_sorted[len(latencies_sorted) // 2],
        'latency_p95_ms': latencies_sorted[int(len(latencies_sorted) * 0.95)],
        'num_queries': len(queries),
    }


def load_sample_data(filepath: str = "sample_data.json") -> Tuple[List, Dict, Dict]:
    """Load sample queries and ground truth from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    queries = [(q['id'], q['text']) for q in data['queries']]
    ground_truth = {q['id']: set(q['relevant']) for q in data['queries']}
    ground_truth_graded = {
        q['id']: q.get('relevant_graded', {})
        for q in data['queries']
    }
    
    return queries, ground_truth, ground_truth_graded


def main():
    """Run evaluation with sample data."""
    print("Loading sample data...")
    queries, ground_truth, ground_truth_graded = load_sample_data()
    
    print(f"Loaded {len(queries)} queries")
    
    # Create retriever (replace with your actual retriever)
    retriever = create_retriever()
    
    # Define parameters to test
    params = {'k': 10, 'method': 'hybrid'}
    
    print(f"\nRunning evaluation with params: {params}")
    results = evaluate(
        queries,
        ground_truth,
        ground_truth_graded,
        retriever.search,
        k=10,
        params=params
    )
    
    # Print results
    print("\nResults:")
    for key, value in results.items():
        if key != 'params':
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
