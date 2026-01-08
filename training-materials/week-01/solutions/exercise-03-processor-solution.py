"""
Exercise 3: Custom Text Processor - SOLUTION

This solution demonstrates building a production-ready text processing system
with the OpenAI API, including caching, error handling, and metrics tracking.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import time
from typing import Optional, Dict, List
from datetime import datetime
import json
import hashlib

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# PART 1: Base TextProcessor Class (15 min)
# ============================================================================

class TextProcessor:
    """
    A text processing utility that uses OpenAI API for various NLP tasks.
    
    Features:
    - Generic API request method with error handling
    - Result caching to avoid duplicate API calls
    - Retry logic for transient failures
    - Comprehensive metrics tracking
    """
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=500):
        """Initialize the TextProcessor with default parameters."""
        
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Cache for storing previous results
        self.cache = {}
        
        # Metrics tracking
        self.metrics = {
            "requests_made": 0,
            "cache_hits": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": 0,
            "start_time": datetime.now()
        }
    
    def _make_request(self, system_prompt: str, user_prompt: str, 
                      use_cache: bool = True, **kwargs) -> Optional[str]:
        """
        Make a request to OpenAI API with error handling and caching.
        
        Features:
        - Checks cache before making API calls
        - Handles errors gracefully with retries
        - Updates comprehensive metrics
        - Stores successful results in cache
        """
        
        # Generate cache key
        cache_key = hashlib.md5(
            f"{system_prompt}:{user_prompt}:{self.model}:{kwargs}".encode()
        ).hexdigest()
        
        # Check cache
        if use_cache and cache_key in self.cache:
            self.metrics["cache_hits"] += 1
            return self.cache[cache_key]
        
        # Make API request with retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Merge default parameters with overrides
                params = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    **kwargs
                }
                
                # Make the API call
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **params
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Update metrics
                self.metrics["requests_made"] += 1
                self.metrics["total_prompt_tokens"] += response.usage.prompt_tokens
                self.metrics["total_completion_tokens"] += response.usage.completion_tokens
                self.metrics["total_tokens"] += response.usage.total_tokens
                
                # Calculate cost (GPT-3.5-turbo pricing)
                cost = self._calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
                self.metrics["total_cost"] += cost
                
                # Store in cache
                if use_cache:
                    self.cache[cache_key] = response_text
                
                return response_text
                
            except Exception as e:
                self.metrics["errors"] += 1
                
                if attempt < max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Request failed after {max_retries} attempts: {str(e)}")
                    return None
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model and token usage."""
        
        # Pricing as of 2025
        pricing = {
            "gpt-3.5-turbo": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        }
        
        model_pricing = pricing.get(self.model, pricing["gpt-4"])
        
        return (
            prompt_tokens * model_pricing["prompt"] +
            completion_tokens * model_pricing["completion"]
        )
    
    def get_metrics(self) -> Dict:
        """Return current usage metrics with calculated efficiency metrics."""
        
        total_requests = self.metrics["requests_made"] + self.metrics["cache_hits"]
        
        # Calculate derived metrics
        metrics = self.metrics.copy()
        
        if total_requests > 0:
            metrics["cache_hit_rate"] = self.metrics["cache_hits"] / total_requests
        else:
            metrics["cache_hit_rate"] = 0.0
        
        if self.metrics["requests_made"] > 0:
            metrics["avg_tokens_per_request"] = (
                self.metrics["total_tokens"] / self.metrics["requests_made"]
            )
            metrics["avg_cost_per_request"] = (
                self.metrics["total_cost"] / self.metrics["requests_made"]
            )
        else:
            metrics["avg_tokens_per_request"] = 0.0
            metrics["avg_cost_per_request"] = 0.0
        
        # Calculate elapsed time
        metrics["elapsed_time_seconds"] = (
            datetime.now() - self.metrics["start_time"]
        ).total_seconds()
        
        return metrics


# ============================================================================
# PART 2: Text Processing Functions (20 min)
# ============================================================================

class EnhancedTextProcessor(TextProcessor):
    """
    Extended TextProcessor with specific NLP task methods.
    """
    
    def summarize(self, text: str, max_words: int = 50, 
                  style: str = "concise") -> Optional[str]:
        """
        Summarize the given text with different style options.
        """
        
        # System prompts for different styles
        system_prompts = {
            "concise": "You are a summarization expert. Create brief, factual summaries that capture the core message.",
            "detailed": "You are a summarization expert. Create comprehensive summaries that preserve important details and nuances.",
            "bullet": "You are a summarization expert. Create well-organized bullet-point summaries that highlight key information."
        }
        
        system_prompt = system_prompts.get(style, system_prompts["concise"])
        
        # Create user prompt
        user_prompt = f"""Summarize the following text in approximately {max_words} words:

{text}

Provide only the summary, without any preamble or explanation."""
        
        # Make request
        return self._make_request(system_prompt, user_prompt)
    
    def translate(self, text: str, target_language: str, 
                  preserve_formatting: bool = True) -> Optional[str]:
        """
        Translate text to the target language.
        """
        
        system_prompt = f"You are a professional translator. Translate text to {target_language} accurately and naturally."
        
        formatting_instruction = ""
        if preserve_formatting:
            formatting_instruction = " Preserve the original formatting, paragraph breaks, and structure."
        
        user_prompt = f"""Translate the following text to {target_language}:{formatting_instruction}

{text}

Provide only the translation, without any explanations."""
        
        return self._make_request(system_prompt, user_prompt)
    
    def analyze_sentiment(self, text: str, detailed: bool = False) -> Optional[Dict]:
        """
        Analyze the sentiment of the given text.
        """
        
        system_prompt = "You are a sentiment analysis expert. Analyze the emotional tone and sentiment of text accurately."
        
        if detailed:
            user_prompt = f"""Analyze the sentiment of the following text. Provide your response as a JSON object with these fields:
- sentiment: "positive", "negative", or "neutral"
- confidence: a number between 0 and 1
- reasoning: brief explanation of your analysis
- key_phrases: list of 3-5 phrases that influenced your assessment

Text to analyze:
{text}

Respond with only the JSON object, no additional text."""
        else:
            user_prompt = f"""Analyze the sentiment of the following text. Provide your response as a JSON object with these fields:
- sentiment: "positive", "negative", or "neutral"
- confidence: a number between 0 and 1

Text to analyze:
{text}

Respond with only the JSON object, no additional text."""
        
        response = self._make_request(system_prompt, user_prompt, temperature=0.3)
        
        if response:
            try:
                # Parse JSON response
                # Handle markdown code blocks if present
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                return json.loads(response)
            except json.JSONDecodeError:
                print(f"Failed to parse sentiment response as JSON: {response}")
                return None
        
        return None
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> Optional[List[str]]:
        """
        Extract important keywords from the text.
        """
        
        system_prompt = "You are a keyword extraction expert. Identify the most important and relevant keywords from text."
        
        user_prompt = f"""Extract up to {max_keywords} important keywords from the following text.
Provide them as a comma-separated list, in order of importance.

Text:
{text}

Respond with only the comma-separated keywords, no additional text."""
        
        response = self._make_request(system_prompt, user_prompt, temperature=0.3)
        
        if response:
            # Parse comma-separated list
            keywords = [k.strip() for k in response.split(',')]
            return keywords[:max_keywords]
        
        return None
    
    def improve_writing(self, text: str, aspects: List[str] = None) -> Optional[str]:
        """
        Improve the writing quality of the given text.
        """
        
        if aspects is None:
            aspects = ["grammar", "clarity", "conciseness"]
        
        aspects_str = ", ".join(aspects)
        
        system_prompt = f"You are a professional editor. Improve text by focusing on: {aspects_str}."
        
        user_prompt = f"""Improve the following text, focusing on {aspects_str}.
Provide the improved version followed by a brief explanation of key changes.

Original text:
{text}

Format your response as:
IMPROVED TEXT:
[improved version here]

CHANGES:
[brief explanation of key improvements]"""
        
        return self._make_request(system_prompt, user_prompt, temperature=0.5)


# ============================================================================
# PART 3: Batch Processing (10 min)
# ============================================================================

def batch_process(processor: EnhancedTextProcessor, texts: List[str], 
                  operation: str, **kwargs) -> List[Optional[str]]:
    """
    Process multiple texts with the same operation.
    """
    
    # Validate operation
    valid_operations = ["summarize", "translate", "improve_writing", "extract_keywords"]
    if operation not in valid_operations:
        raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")
    
    # Get the method to call
    method = getattr(processor, operation)
    
    results = []
    total = len(texts)
    
    print(f"\nBatch processing {total} texts with operation: {operation}")
    
    for i, text in enumerate(texts, 1):
        try:
            # Progress reporting every 5 items
            if i % 5 == 0 or i == total:
                print(f"Processing {i}/{total}...")
            
            # Call the operation
            result = method(text, **kwargs)
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            results.append(None)
    
    print(f"Batch processing complete: {sum(1 for r in results if r is not None)}/{total} successful")
    
    return results


# ============================================================================
# PART 4: Performance Testing (Bonus)
# ============================================================================

def benchmark_processor(sample_text: str):
    """
    Benchmark the TextProcessor performance.
    """
    
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKING")
    print("="*70)
    
    results = {}
    
    # Test 1: Speed
    print("\n1. Speed Test (10 summarization requests)")
    processor = EnhancedTextProcessor()
    
    start_time = time.time()
    for i in range(10):
        processor.summarize(sample_text, max_words=30)
    elapsed_time = time.time() - start_time
    
    results["speed_test"] = {
        "total_time": elapsed_time,
        "avg_time_per_request": elapsed_time / 10
    }
    print(f"   Total time: {elapsed_time:.2f}s")
    print(f"   Avg per request: {elapsed_time/10:.2f}s")
    
    # Test 2: Cache Efficiency
    print("\n2. Cache Efficiency Test")
    processor2 = EnhancedTextProcessor()
    
    # Make same request 5 times
    for i in range(5):
        processor2.summarize(sample_text, max_words=30)
    
    metrics = processor2.get_metrics()
    results["cache_test"] = {
        "cache_hit_rate": metrics["cache_hit_rate"],
        "requests_made": metrics["requests_made"],
        "cache_hits": metrics["cache_hits"]
    }
    print(f"   Cache hit rate: {metrics['cache_hit_rate']*100:.1f}%")
    print(f"   API requests made: {metrics['requests_made']}")
    print(f"   Cache hits: {metrics['cache_hits']}")
    
    # Test 3: Cost Efficiency
    print("\n3. Cost Efficiency Test (GPT-3.5 vs GPT-4)")
    
    # GPT-3.5-turbo
    processor_35 = EnhancedTextProcessor(model="gpt-3.5-turbo")
    processor_35.summarize(sample_text, max_words=30, use_cache=False)
    metrics_35 = processor_35.get_metrics()
    
    # GPT-4 (simulated - uncomment if you have access)
    # processor_4 = EnhancedTextProcessor(model="gpt-4")
    # processor_4.summarize(sample_text, max_words=30, use_cache=False)
    # metrics_4 = processor_4.get_metrics()
    
    results["cost_test"] = {
        "gpt_35_cost": metrics_35["avg_cost_per_request"],
        # "gpt_4_cost": metrics_4["avg_cost_per_request"],
        "gpt_35_tokens": metrics_35["avg_tokens_per_request"]
    }
    print(f"   GPT-3.5-turbo cost: ${metrics_35['avg_cost_per_request']:.6f}")
    print(f"   GPT-3.5-turbo tokens: {metrics_35['avg_tokens_per_request']:.1f}")
    
    # Test 4: Error Handling
    print("\n4. Error Handling Test")
    processor_err = EnhancedTextProcessor()
    
    # Test with empty string
    result = processor_err.summarize("")
    
    # Test with very long text (might hit token limit)
    long_text = sample_text * 100
    result = processor_err.summarize(long_text, max_words=50)
    
    metrics_err = processor_err.get_metrics()
    results["error_test"] = {
        "errors_encountered": metrics_err["errors"],
        "error_rate": metrics_err["errors"] / max(metrics_err["requests_made"], 1)
    }
    print(f"   Errors encountered: {metrics_err['errors']}")
    
    return results


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def run_tests():
    """
    Run all exercise tests.
    """
    
    print("=" * 70)
    print("EXERCISE 3: CUSTOM TEXT PROCESSOR - SOLUTION")
    print("=" * 70)
    
    # Sample texts for testing
    sample_text = """
    Artificial intelligence (AI) is transforming the way we live and work. 
    Machine learning algorithms can now perform tasks that once required human 
    intelligence, from recognizing images to translating languages. However, 
    this rapid advancement also raises important ethical questions about privacy, 
    bias, and the future of employment. As AI systems become more sophisticated, 
    it's crucial that we develop them responsibly and ensure they benefit all 
    of humanity.
    """
    
    # Test Part 1: Basic TextProcessor
    print("\n--- Part 1: TextProcessor Initialization ---")
    processor = EnhancedTextProcessor()
    print(f"Processor initialized with model: {processor.model}")
    initial_metrics = processor.get_metrics()
    print(f"Initial metrics:")
    print(f"  Requests: {initial_metrics['requests_made']}")
    print(f"  Cache hits: {initial_metrics['cache_hits']}")
    print(f"  Total cost: ${initial_metrics['total_cost']:.6f}")
    
    # Test Part 2a: Summarization
    print("\n--- Part 2a: Summarization ---")
    summary = processor.summarize(sample_text, max_words=30, style="concise")
    print(f"Concise summary: {summary}")
    
    bullet_summary = processor.summarize(sample_text, max_words=50, style="bullet")
    print(f"\nBullet summary:\n{bullet_summary}")
    
    # Test Part 2b: Translation
    print("\n--- Part 2b: Translation ---")
    spanish = processor.translate(sample_text, "Spanish")
    print(f"Spanish translation: {spanish[:200]}...")
    
    # Test Part 2c: Sentiment Analysis
    print("\n--- Part 2c: Sentiment Analysis ---")
    sentiment = processor.analyze_sentiment(sample_text, detailed=True)
    if sentiment:
        print(f"Sentiment: {sentiment.get('sentiment', 'N/A')}")
        print(f"Confidence: {sentiment.get('confidence', 'N/A')}")
        if 'reasoning' in sentiment:
            print(f"Reasoning: {sentiment['reasoning']}")
    
    # Test Part 2d: Keyword Extraction
    print("\n--- Part 2d: Keyword Extraction ---")
    keywords = processor.extract_keywords(sample_text, max_keywords=5)
    if keywords:
        print(f"Keywords: {', '.join(keywords)}")
    
    # Test Part 2e: Writing Improvement
    print("\n--- Part 2e: Writing Improvement ---")
    poor_text = "This text not very good. It have grammar errors and unclear meaning."
    improved = processor.improve_writing(poor_text, aspects=["grammar", "clarity"])
    print(f"Original: {poor_text}")
    print(f"Improved:\n{improved}")
    
    # Test Part 3: Batch Processing
    print("\n--- Part 3: Batch Processing ---")
    texts = [
        "AI is changing the world.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Generative AI can create new content like text, images, and code."
    ]
    summaries = batch_process(processor, texts, "summarize", max_words=10, style="concise")
    print("\nBatch summaries:")
    for i, summary in enumerate(summaries, 1):
        if summary:
            print(f"  {i}. {summary}")
    
    # Test caching
    print("\n--- Testing Cache ---")
    print("Making duplicate request...")
    summary2 = processor.summarize(sample_text, max_words=30, style="concise")
    print(f"Second summary (should be cached): {summary2[:100]}...")
    
    final_metrics = processor.get_metrics()
    print(f"\nFinal metrics:")
    print(f"  Total requests: {final_metrics['requests_made']}")
    print(f"  Cache hits: {final_metrics['cache_hits']}")
    print(f"  Cache hit rate: {final_metrics['cache_hit_rate']*100:.1f}%")
    print(f"  Total tokens: {final_metrics['total_tokens']}")
    print(f"  Total cost: ${final_metrics['total_cost']:.6f}")
    print(f"  Avg cost per request: ${final_metrics['avg_cost_per_request']:.6f}")
    
    # Test Part 4: Benchmarking (Bonus)
    print("\n--- Part 4: Benchmarking (Bonus) ---")
    benchmark_results = benchmark_processor(sample_text)
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


# ============================================================================
# REFLECTION QUESTIONS - ANSWERS
# ============================================================================

"""
REFLECTION ANSWERS:

1. Why is caching important for a text processing system?

   Benefits of caching:
   - **Cost reduction**: Avoid paying for duplicate API calls
   - **Performance**: Instant responses for cached queries (no API latency)
   - **Rate limit management**: Reduce total API calls, staying within limits
   - **Consistency**: Same input always returns same output
   - **Reliability**: System works even if API is temporarily unavailable
   
   Implementation considerations:
   - Use hash of input as cache key to handle exact matches
   - Consider TTL (time-to-live) for cache entries
   - Monitor cache size and implement eviction policies
   - Decide whether to cache errors or only successful responses

2. How would you handle API rate limits in a production system?

   Strategies:
   
   a) **Exponential backoff with retries**:
      - Catch rate limit errors (HTTP 429)
      - Wait and retry with increasing delays (1s, 2s, 4s, etc.)
      - Respect Retry-After header if provided
   
   b) **Request throttling**:
      - Implement token bucket or sliding window algorithm
      - Limit requests per minute/hour
      - Queue requests when approaching limits
   
   c) **Monitoring and alerting**:
      - Track request rates and usage patterns
      - Set up alerts before hitting limits
      - Use different API keys for different services/users
   
   d) **Graceful degradation**:
      - Return cached results when rate limited
      - Provide partial results or fallback responses
      - Queue non-urgent requests for later processing

3. What are the tradeoffs between using GPT-3.5-turbo vs GPT-4 for these tasks?

   GPT-3.5-turbo:
   - ✅ Much cheaper (~15-20x less expensive)
   - ✅ Faster response times
   - ✅ Sufficient for many tasks (summarization, translation)
   - ❌ Lower quality for complex reasoning
   - ❌ More prone to errors on nuanced tasks
   
   GPT-4:
   - ✅ Superior quality and accuracy
   - ✅ Better at following complex instructions
   - ✅ More reliable for critical applications
   - ✅ Better at nuanced sentiment analysis
   - ❌ Significantly more expensive
   - ❌ Slower response times
   
   Recommendation: Use GPT-3.5-turbo by default, upgrade to GPT-4 for:
   - High-stakes applications (legal, medical)
   - Complex reasoning tasks
   - When quality is more important than cost
   - When GPT-3.5 consistently fails to meet requirements

4. How could you improve error handling for network failures?

   Enhanced error handling strategies:
   
   a) **Retry with exponential backoff** (already implemented):
      - Distinguish transient vs permanent errors
      - Network timeouts: retry
      - Invalid API key: don't retry
   
   b) **Circuit breaker pattern**:
      - Track failure rate over time
      - If failures exceed threshold, "open circuit" (fail fast)
      - Periodically try again (half-open state)
      - Reset when successful (closed state)
   
   c) **Timeouts and cancellation**:
      - Set reasonable timeout values
      - Allow cancellation of long-running requests
      - Clean up resources properly
   
   d) **Fallback strategies**:
      - Return cached results if available
      - Use simpler/cheaper model as fallback
      - Provide graceful degradation messages
   
   e) **Monitoring and logging**:
      - Log all errors with context
      - Track error rates and patterns
      - Set up alerts for anomalies

5. What additional metrics would be useful to track?

   Performance metrics:
   - Request latency (p50, p95, p99)
   - Success rate (% of successful requests)
   - Retry count and success rate after retries
   - Time spent waiting in rate limit backoff
   
   Cost metrics:
   - Cost per operation type (summarize, translate, etc.)
   - Cost breakdown by model
   - Monthly/weekly cost trends
   - Cost per user (if multi-tenant)
   
   Quality metrics:
   - User feedback/satisfaction scores
   - Error rates by operation type
   - Cache effectiveness by operation
   - Response length distribution
   
   Business metrics:
   - Most popular operations
   - Peak usage times
   - User retention/engagement
   - Feature adoption rates

6. How would you modify this class to support multiple LLM providers 
   (Anthropic, Cohere)?

   Design approach:
   
   a) **Provider abstraction**:
      ```python
      class LLMProvider:
          def make_request(self, system, user, **kwargs):
              raise NotImplementedError
      
      class OpenAIProvider(LLMProvider):
          def make_request(self, system, user, **kwargs):
              # OpenAI-specific implementation
      
      class AnthropicProvider(LLMProvider):
          def make_request(self, system, user, **kwargs):
              # Anthropic-specific implementation
      ```
   
   b) **Unified parameter mapping**:
      - Map common parameters (temperature, max_tokens) to provider-specific names
      - Handle provider-specific features gracefully
   
   c) **Factory pattern for provider selection**:
      ```python
      def get_provider(provider_name):
          if provider_name == "openai":
              return OpenAIProvider()
          elif provider_name == "anthropic":
              return AnthropicProvider()
      ```
   
   d) **Fallback chain**:
      - Try primary provider
      - If it fails, try secondary provider
      - Track which provider was used in metrics

7. What security considerations should you keep in mind when processing user text?

   Security concerns:
   
   a) **API key protection**:
      - Never expose keys in code or logs
      - Use environment variables or secret managers
      - Rotate keys regularly
      - Use different keys for dev/staging/prod
   
   b) **Input validation**:
      - Limit input length to prevent abuse
      - Sanitize inputs to prevent injection attacks
      - Validate character encodings
   
   c) **Sensitive data handling**:
      - Don't log user inputs (may contain PII)
      - Be aware API providers may store data
      - Implement data retention policies
      - Consider on-premise models for sensitive data
   
   d) **Output sanitization**:
      - Filter harmful content in responses
      - Implement content moderation
      - Prevent prompt injection attacks
   
   e) **Rate limiting per user**:
      - Prevent abuse and DoS attacks
      - Implement usage quotas
      - Track and alert on suspicious patterns
   
   f) **Audit logging**:
      - Log who accessed what and when
      - Track cost per user
      - Enable forensic analysis if needed
"""


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Please create a .env file with your API key.")
    else:
        run_tests()
