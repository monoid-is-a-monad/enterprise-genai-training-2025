"""
Mock Retriever for Demonstration

Replace this with your actual vector DB search function.
"""
import time
import random
from typing import List, Dict, Any


class Document:
    """Simple document representation."""
    def __init__(self, id: str, score: float, content: str = ""):
        self.id = id
        self.score = score
        self.content = content


class MockRetriever:
    """
    Stub retriever that returns random documents with scores.
    Replace with your actual vector DB (Chroma, Pinecone, Weaviate, etc.).
    """
    
    def __init__(self, corpus_size: int = 1000):
        self.corpus_size = corpus_size
    
    def search(self, query: str, k: int = 10, **params) -> List[Document]:
        """
        Mock search function.
        
        Args:
            query: Query text (ignored in mock)
            k: Number of results to return
            **params: Additional search parameters (e.g., ef_search, alpha)
        
        Returns:
            List of Document objects
        """
        # Simulate latency
        time.sleep(random.uniform(0.02, 0.08))
        
        # Return random documents with scores
        doc_ids = [f"doc_{random.randint(1, self.corpus_size)}" for _ in range(k)]
        scores = sorted([random.random() for _ in range(k)], reverse=True)
        
        return [
            Document(id=doc_id, score=score, content=f"Content of {doc_id}")
            for doc_id, score in zip(doc_ids, scores)
        ]


# Example: Replace with your actual retriever
def create_retriever() -> MockRetriever:
    """Factory function to create retriever instance."""
    return MockRetriever(corpus_size=1000)


if __name__ == "__main__":
    # Quick test
    retriever = create_retriever()
    results = retriever.search("sample query", k=5)
    
    print("Mock retriever test:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.id} (score: {doc.score:.3f})")
