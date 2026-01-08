"""
Week 3 - Exercise 2: API Optimization Challenge

Learning Objectives:
- Optimize OpenAI API usage for cost and latency
- Implement effective caching strategies
- Use streaming for better user experience
- Manage rate limits and token budgets
- Monitor and measure API performance

Scenario:
You're building a customer service chatbot that handles 10,000 conversations
per day. You need to optimize for cost (target <$50/day) while maintaining
good response quality and latency (target <2 seconds).

Time: 90 minutes
"""

import os
from openai import OpenAI
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import hashlib
from collections import deque
import asyncio
import tiktoken

# TODO: Initialize your OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Part 1: Cost Analysis and Model Selection (20 minutes)
# ============================================================================

@dataclass
class ConversationMetrics:
    """Tracks metrics for a single conversation"""
    conversation_id: str
    model_used: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    response_time: float = 0.0
    messages_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class CostAnalyzer:
    """
    TODO: Implement cost analysis for different model choices
    
    Compare:
    - gpt-4-turbo: $0.01/1K prompt, $0.03/1K completion
    - gpt-4: $0.03/1K prompt, $0.06/1K completion
    - gpt-3.5-turbo: $0.0005/1K prompt, $0.0015/1K completion
    """
    
    PRICING = {
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    }
    
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def estimate_conversation_cost(
        self,
        messages: List[Dict[str, str]],
        model: str,
        expected_completion_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        TODO: Estimate cost for a conversation
        
        Should return:
        {
            "prompt_tokens": int,
            "estimated_completion_tokens": int,
            "total_tokens": int,
            "prompt_cost": float,
            "completion_cost": float,
            "total_cost": float,
            "model": str
        }
        
        Hints:
        - Use tiktoken to count tokens in messages
        - Calculate cost based on PRICING table
        """
        # TODO: Implement this method
        pass
    
    def compare_models(
        self,
        messages: List[Dict[str, str]],
        expected_completion_tokens: int = 150
    ) -> Dict[str, Dict[str, Any]]:
        """
        TODO: Compare costs across all available models
        
        Return comparison showing which model is most cost-effective
        """
        # TODO: Implement this method
        pass
    
    def calculate_daily_cost(
        self,
        conversations_per_day: int,
        avg_messages_per_conversation: int,
        avg_tokens_per_message: int,
        avg_completion_tokens: int,
        model: str
    ) -> Dict[str, Any]:
        """
        TODO: Calculate projected daily costs
        
        Given conversation volume, estimate:
        - Total daily tokens
        - Total daily cost
        - Cost per conversation
        - Whether it fits budget ($50/day target)
        """
        # TODO: Implement this method
        pass


def test_cost_analysis():
    """Test cost analysis implementation"""
    # TODO: Initialize analyzer
    # analyzer = CostAnalyzer()
    
    sample_conversation = [
        {"role": "system", "content": "You are a helpful customer service assistant."},
        {"role": "user", "content": "I need help with my order #12345"},
        {"role": "assistant", "content": "I'll be happy to help you with order #12345. What seems to be the issue?"},
        {"role": "user", "content": "The tracking shows it's stuck in transit for 5 days"}
    ]
    
    # TODO: Uncomment and test
    # comparison = analyzer.compare_models(sample_conversation, expected_completion_tokens=100)
    # print("Model Cost Comparison:")
    # for model, costs in comparison.items():
    #     print(f"\n{model}:")
    #     print(f"  Total cost: ${costs['total_cost']:.4f}")
    #     print(f"  Total tokens: {costs['total_tokens']}")
    
    # TODO: Calculate daily costs for 10,000 conversations
    # daily = analyzer.calculate_daily_cost(
    #     conversations_per_day=10000,
    #     avg_messages_per_conversation=6,
    #     avg_tokens_per_message=50,
    #     avg_completion_tokens=100,
    #     model="gpt-3.5-turbo"
    # )
    # print(f"\nProjected Daily Cost: ${daily['total_daily_cost']:.2f}")
    # print(f"Within budget? {daily['within_budget']}")


# ============================================================================
# Part 2: Intelligent Caching System (30 minutes)
# ============================================================================

class ResponseCache:
    """
    TODO: Implement a smart caching system to reduce API calls
    
    Features:
    1. Cache common queries and their responses
    2. Use semantic similarity for cache hits (not just exact matches)
    3. Implement TTL (time-to-live) for cache entries
    4. Track cache hit rate
    5. Periodic cache cleanup
    """
    
    def __init__(
        self,
        client: OpenAI,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.95
    ):
        """
        Args:
            client: OpenAI client for embeddings
            ttl_seconds: How long cache entries are valid
            similarity_threshold: How similar queries must be for cache hit
        """
        # TODO: Initialize cache data structures
        pass
    
    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """
        TODO: Generate cache key from messages
        
        Hints:
        - Hash the message content
        - Include model in key
        - Consider only last N messages for efficiency
        """
        # TODO: Implement this method
        pass
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        TODO: Get embedding for semantic similarity
        
        Use text-embedding-3-small for cost efficiency
        """
        # TODO: Implement this method
        pass
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        TODO: Calculate cosine similarity between embeddings
        """
        # TODO: Implement this method
        pass
    
    def _find_similar_cache_entry(
        self,
        messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        TODO: Search cache for semantically similar entry
        
        Steps:
        1. Get embedding for current query
        2. Compare with cached query embeddings
        3. Return cache entry if similarity > threshold
        4. Return None if no match
        """
        # TODO: Implement this method
        pass
    
    def get(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        TODO: Retrieve from cache if exists and valid
        
        Should:
        1. Check exact match first (fastest)
        2. Check semantic similarity if no exact match
        3. Verify TTL hasn't expired
        4. Update cache statistics
        """
        # TODO: Implement this method
        pass
    
    def set(
        self,
        messages: List[Dict[str, str]],
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        TODO: Add response to cache
        
        Should store:
        - Response text
        - Timestamp
        - Embedding of query
        - Usage metadata (tokens, cost)
        """
        # TODO: Implement this method
        pass
    
    def cleanup_expired(self):
        """
        TODO: Remove expired cache entries
        """
        # TODO: Implement this method
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        TODO: Return cache performance statistics
        
        Should include:
        - Total requests
        - Cache hits / misses
        - Hit rate
        - Total tokens saved
        - Cost savings
        - Cache size
        """
        # TODO: Implement this method
        pass


class CachedChatClient:
    """
    TODO: Wrapper around OpenAI client that uses caching
    """
    
    def __init__(self, client: OpenAI, cache: ResponseCache):
        self.client = client
        self.cache = cache
        self.metrics = []
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> Dict[str, Any]:
        """
        TODO: Get completion with caching
        
        1. Check cache first
        2. If miss, call API
        3. Store in cache
        4. Track metrics
        """
        # TODO: Implement this method
        pass


def test_caching():
    """Test caching implementation"""
    # TODO: Initialize cache and client
    # cache = ResponseCache(client, ttl_seconds=300)
    # cached_client = CachedChatClient(client, cache)
    
    # Test with repeated queries
    queries = [
        "How do I reset my password?",
        "How can I reset my password?",  # Similar, should hit cache
        "What is your return policy?",
        "How do I reset my password?",  # Exact match, should hit cache
    ]
    
    # TODO: Uncomment and test
    # for query in queries:
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": query}
    #     ]
    #     response = cached_client.chat_completion(messages)
    #     print(f"\nQuery: {query}")
    #     print(f"Cache hit: {response.get('cache_hit', False)}")
    
    # stats = cache.get_statistics()
    # print(f"\nCache Statistics:")
    # print(f"Hit rate: {stats['hit_rate']:.1%}")
    # print(f"Cost savings: ${stats['cost_savings']:.4f}")


# ============================================================================
# Part 3: Streaming with Progress Feedback (20 minutes)
# ============================================================================

class StreamingChatClient:
    """
    TODO: Implement streaming with user feedback
    
    Features:
    - Stream responses token by token
    - Show typing indicators
    - Allow early termination if user isn't interested
    - Track time to first token (TTFT)
    - Handle streaming errors gracefully
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        on_token: Optional[Callable[[str], None]] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ) -> Dict[str, Any]:
        """
        TODO: Stream chat completion with callbacks
        
        Callbacks:
        - on_start: Called when streaming begins
        - on_token: Called for each token (string chunk)
        - on_complete: Called with full response when done
        - on_error: Called if streaming fails
        
        Return metrics:
        - time_to_first_token
        - total_time
        - tokens_streamed
        - full_response
        """
        # TODO: Implement this method
        pass
    
    async def stream_chat_async(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        on_token: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        TODO: Async version for better performance
        """
        # TODO: Implement this method
        pass


def test_streaming():
    """Test streaming implementation"""
    # TODO: Initialize streaming client
    # streaming_client = StreamingChatClient(client)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how caching works in 3 paragraphs"}
    ]
    
    # Callback to print tokens as they arrive
    def print_token(token: str):
        print(token, end="", flush=True)
    
    # TODO: Uncomment and test
    # print("Streaming response:")
    # result = streaming_client.stream_chat(
    #     messages,
    #     on_token=print_token
    # )
    # print(f"\n\nMetrics:")
    # print(f"Time to first token: {result['time_to_first_token']:.3f}s")
    # print(f"Total time: {result['total_time']:.3f}s")


# ============================================================================
# Part 4: Rate Limiting and Token Budgets (20 minutes)
# ============================================================================

class TokenBudgetManager:
    """
    TODO: Manage token budgets to control costs
    
    Features:
    - Set daily/hourly token budgets
    - Track usage across conversations
    - Reject requests when budget exceeded
    - Alert when approaching limits
    - Reset budgets on schedule
    """
    
    def __init__(
        self,
        daily_token_limit: int,
        hourly_token_limit: int,
        alert_threshold: float = 0.8
    ):
        """
        Args:
            daily_token_limit: Max tokens per day
            hourly_token_limit: Max tokens per hour
            alert_threshold: Alert when usage exceeds this fraction
        """
        # TODO: Initialize budget tracking
        pass
    
    def check_budget(self, estimated_tokens: int) -> Dict[str, Any]:
        """
        TODO: Check if request fits within budget
        
        Return:
        {
            "allowed": bool,
            "reason": str (if not allowed),
            "daily_remaining": int,
            "hourly_remaining": int,
            "alert": bool (if near limit)
        }
        """
        # TODO: Implement this method
        pass
    
    def record_usage(self, tokens_used: int):
        """
        TODO: Record token usage
        """
        # TODO: Implement this method
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        TODO: Get current usage statistics
        """
        # TODO: Implement this method
        pass
    
    def reset_if_needed(self):
        """
        TODO: Reset budgets if period has expired
        """
        # TODO: Implement this method
        pass


class OptimizedChatClient:
    """
    TODO: Combine all optimizations into production-ready client
    
    Should include:
    - Cost-based model selection
    - Intelligent caching
    - Streaming support
    - Token budget management
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        client: OpenAI,
        cache: ResponseCache,
        budget_manager: TokenBudgetManager,
        default_model: str = "gpt-3.5-turbo"
    ):
        # TODO: Initialize with all components
        pass
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        TODO: Optimized chat completion
        
        1. Check token budget
        2. Check cache
        3. Select model based on budget/requirements
        4. Make API call (streamed or not)
        5. Update cache and budgets
        6. Return response with metrics
        """
        # TODO: Implement this method
        pass
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        TODO: Generate comprehensive performance report
        
        Should include:
        - Total conversations handled
        - Total cost
        - Average cost per conversation
        - Cache hit rate
        - Average latency
        - Token usage
        - Model distribution
        """
        # TODO: Implement this method
        pass


# ============================================================================
# Reflection Questions
# ============================================================================

"""
After completing the exercises, reflect on these questions:

1. COST OPTIMIZATION:
   - What was your biggest cost saving opportunity?
   - When is gpt-4 worth the extra cost vs gpt-3.5-turbo?
   - How much did caching reduce costs?

2. CACHING STRATEGY:
   - What cache hit rate did you achieve?
   - How did semantic similarity help?
   - What TTL value worked best?
   - When should you NOT cache responses?

3. LATENCY VS COST:
   - How did streaming improve perceived latency?
   - What's the tradeoff between response quality and speed?
   - When would you choose a faster model over a better one?

4. BUDGET MANAGEMENT:
   - How did you handle budget exhaustion?
   - What alert thresholds made sense?
   - How would you allocate budget across users/priorities?

5. PRODUCTION CONSIDERATIONS:
   - What metrics would you monitor in production?
   - How would you detect performance degradation?
   - What would your incident response playbook include?

Write your reflections in: exercise-02-reflections.md
"""


# ============================================================================
# Optional Challenge: Dynamic Model Selection
# ============================================================================

"""
CHALLENGE: Implement dynamic model selection based on:
- Query complexity (use GPT-4 for complex, GPT-3.5 for simple)
- Remaining budget (downgrade model if low on budget)
- User priority (premium users get better models)
- Time of day (use cheaper models during peak hours)

Hints:
- Use a classifier to determine query complexity
- Track budget burn rate
- Implement user tier system
- Monitor usage patterns
"""


if __name__ == "__main__":
    print("Week 3 - Exercise 2: API Optimization Challenge")
    print("="*80)
    
    # Uncomment to run tests as you complete each part
    # test_cost_analysis()
    # test_caching()
    # test_streaming()
