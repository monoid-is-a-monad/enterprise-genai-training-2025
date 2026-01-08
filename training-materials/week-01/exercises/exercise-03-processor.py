"""
Exercise 3: Custom Text Processor

Time: 45 minutes
Difficulty: Intermediate
Focus: Building a multi-function text processing system using OpenAI API

OBJECTIVES:
1. Create a reusable TextProcessor class
2. Implement multiple text processing functions (summarize, translate, analyze sentiment)
3. Handle errors and edge cases
4. Add caching for repeated requests
5. Track performance metrics

SETUP:
- Ensure your .env file has OPENAI_API_KEY set
- Install required packages: openai, python-dotenv
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import time
from typing import Optional, Dict, List
from datetime import datetime

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
    
    TODO: Complete this class implementation with the following features:
    1. Initialize with model selection and default parameters
    2. Implement a generic _make_request() method for API calls
    3. Add error handling and retry logic
    4. Implement caching to avoid duplicate requests
    5. Track metrics (requests made, tokens used, total cost)
    """
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=500):
        """
        Initialize the TextProcessor.
        
        TODO: Set up instance variables:
        - self.client (OpenAI client)
        - self.model
        - self.temperature
        - self.max_tokens
        - self.cache (dict to store previous results)
        - self.metrics (dict to track usage statistics)
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # TODO: Initialize cache and metrics
        self.cache = {}
        self.metrics = {
            "requests_made": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": 0
        }
    
    def _make_request(self, system_prompt: str, user_prompt: str, 
                      use_cache: bool = True, **kwargs) -> Optional[str]:
        """
        Make a request to OpenAI API with error handling and caching.
        
        TODO: Implement this method with:
        1. Check cache if use_cache is True (use system_prompt + user_prompt as key)
        2. Make API request with error handling (try/except)
        3. Update metrics (requests, tokens, cost)
        4. Store result in cache if successful
        5. Return the response text
        
        Args:
            system_prompt: System message for the model
            user_prompt: User message/input text
            use_cache: Whether to use cached results
            **kwargs: Additional parameters to pass to API
        
        Returns:
            Response text or None if error occurred
        """
        
        # TODO: Implement caching logic
        cache_key = f"{system_prompt}:{user_prompt}"
        
        # TODO: Check cache
        
        # TODO: Make API request with error handling
        try:
            # Merge default parameters with any overrides from kwargs
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            # TODO: Call client.chat.completions.create()
            # TODO: Extract response and usage information
            # TODO: Update metrics
            # TODO: Store in cache
            
            pass
            
        except Exception as e:
            # TODO: Handle errors, update error count
            self.metrics["errors"] += 1
            print(f"Error in API request: {str(e)}")
            return None
    
    def get_metrics(self) -> Dict:
        """
        Return current usage metrics.
        
        TODO: Return a copy of self.metrics with calculated efficiency metrics:
        - Add cache_hit_rate (cache_hits / total_requests)
        - Add average_tokens_per_request
        - Add average_cost_per_request
        """
        
        # TODO: Calculate and return enhanced metrics
        pass


# ============================================================================
# PART 2: Text Processing Functions (20 min)
# ============================================================================

class TextProcessor(TextProcessor):
    """
    Extended TextProcessor with specific NLP task methods.
    """
    
    def summarize(self, text: str, max_words: int = 50, 
                  style: str = "concise") -> Optional[str]:
        """
        Summarize the given text.
        
        TODO: Implement summarization with:
        1. Create appropriate system prompt based on style
           - "concise": Brief, factual summary
           - "detailed": Comprehensive summary with key points
           - "bullet": Bullet-point format
        2. Create user prompt with the text and max_words constraint
        3. Call _make_request()
        4. Return the summary
        
        Args:
            text: Text to summarize
            max_words: Maximum words in summary
            style: Summary style ("concise", "detailed", "bullet")
        
        Returns:
            Summarized text
        """
        
        # TODO: Create system prompt based on style
        system_prompts = {
            "concise": "You are a summarization expert. Create brief, factual summaries.",
            "detailed": "You are a summarization expert. Create comprehensive summaries with key points.",
            "bullet": "You are a summarization expert. Create bullet-point summaries."
        }
        
        # TODO: Get appropriate system prompt or use default
        
        # TODO: Create user prompt
        
        # TODO: Call _make_request() and return result
        
        pass
    
    def translate(self, text: str, target_language: str, 
                  preserve_formatting: bool = True) -> Optional[str]:
        """
        Translate text to the target language.
        
        TODO: Implement translation with:
        1. Create system prompt for translation
        2. Specify target language and formatting requirements
        3. Call _make_request()
        4. Return translated text
        
        Args:
            text: Text to translate
            target_language: Target language (e.g., "Spanish", "French", "Japanese")
            preserve_formatting: Whether to maintain original formatting
        
        Returns:
            Translated text
        """
        
        # TODO: Create system prompt
        
        # TODO: Create user prompt with formatting instructions
        
        # TODO: Call _make_request() and return result
        
        pass
    
    def analyze_sentiment(self, text: str, detailed: bool = False) -> Optional[Dict]:
        """
        Analyze the sentiment of the given text.
        
        TODO: Implement sentiment analysis with:
        1. Create system prompt for sentiment analysis
        2. Request structured output (sentiment label + confidence score)
        3. If detailed=True, also request reasoning and key phrases
        4. Parse the response into a dictionary
        5. Return structured sentiment data
        
        Args:
            text: Text to analyze
            detailed: Whether to include detailed analysis
        
        Returns:
            Dictionary with sentiment analysis results:
            {
                "sentiment": "positive" | "negative" | "neutral",
                "confidence": float (0-1),
                "reasoning": str (if detailed=True),
                "key_phrases": list (if detailed=True)
            }
        """
        
        # TODO: Create system prompt
        
        # TODO: Create user prompt with output format instructions
        
        # TODO: Call _make_request()
        
        # TODO: Parse response into dictionary (handle JSON parsing)
        
        pass
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> Optional[List[str]]:
        """
        Extract important keywords from the text.
        
        TODO: Implement keyword extraction with:
        1. Create system prompt for keyword extraction
        2. Specify max_keywords limit
        3. Request comma-separated list of keywords
        4. Parse response into list
        5. Return keywords list
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to extract
        
        Returns:
            List of keywords
        """
        
        # TODO: Implement keyword extraction
        
        pass
    
    def improve_writing(self, text: str, aspects: List[str] = None) -> Optional[str]:
        """
        Improve the writing quality of the given text.
        
        TODO: Implement writing improvement with:
        1. Default aspects: ["grammar", "clarity", "conciseness"]
        2. Create system prompt for editing
        3. Specify which aspects to focus on
        4. Request improved version with brief explanation of changes
        5. Return improved text
        
        Args:
            text: Text to improve
            aspects: List of aspects to focus on 
                    (e.g., ["grammar", "clarity", "conciseness", "tone", "structure"])
        
        Returns:
            Improved text
        """
        
        if aspects is None:
            aspects = ["grammar", "clarity", "conciseness"]
        
        # TODO: Implement writing improvement
        
        pass


# ============================================================================
# PART 3: Batch Processing (10 min)
# ============================================================================

def batch_process(processor: TextProcessor, texts: List[str], 
                  operation: str, **kwargs) -> List[Optional[str]]:
    """
    Process multiple texts with the same operation.
    
    TODO: Implement batch processing with:
    1. Validate operation is one of: ["summarize", "translate", "improve_writing"]
    2. For each text, call the appropriate method on processor
    3. Add progress reporting (print progress every 5 items)
    4. Handle errors gracefully (continue processing even if one fails)
    5. Return list of results
    
    Args:
        processor: TextProcessor instance
        texts: List of texts to process
        operation: Operation to perform ("summarize", "translate", etc.)
        **kwargs: Additional arguments to pass to the operation
    
    Returns:
        List of processed texts (None for failed items)
    """
    
    # TODO: Implement batch processing
    
    pass


# ============================================================================
# PART 4: Performance Testing (Bonus)
# ============================================================================

def benchmark_processor(sample_text: str):
    """
    BONUS TODO: Benchmark the TextProcessor performance.
    
    Test scenarios:
    1. Speed: Time taken for 10 summarization requests
    2. Cache efficiency: Make duplicate requests and measure cache hit rate
    3. Cost efficiency: Compare cost of different models (gpt-3.5-turbo vs gpt-4)
    4. Error handling: Test with invalid inputs
    
    Args:
        sample_text: Text to use for benchmarking
    
    Returns:
        Dictionary with benchmark results
    """
    
    # TODO: Implement benchmarking tests
    
    pass


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def run_tests():
    """
    Test runner for all exercises.
    Uncomment each section as you complete it.
    """
    
    print("=" * 70)
    print("EXERCISE 3: CUSTOM TEXT PROCESSOR")
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
    # print("\n--- Part 1: TextProcessor Initialization ---")
    # processor = TextProcessor()
    # print(f"Processor initialized with model: {processor.model}")
    # print(f"Initial metrics: {processor.get_metrics()}")
    
    # Test Part 2a: Summarization
    # print("\n--- Part 2a: Summarization ---")
    # summary = processor.summarize(sample_text, max_words=30, style="concise")
    # print(f"Concise summary: {summary}")
    # 
    # bullet_summary = processor.summarize(sample_text, max_words=50, style="bullet")
    # print(f"Bullet summary:\n{bullet_summary}")
    
    # Test Part 2b: Translation
    # print("\n--- Part 2b: Translation ---")
    # spanish = processor.translate(sample_text, "Spanish")
    # print(f"Spanish translation: {spanish}")
    
    # Test Part 2c: Sentiment Analysis
    # print("\n--- Part 2c: Sentiment Analysis ---")
    # sentiment = processor.analyze_sentiment(sample_text, detailed=True)
    # print(f"Sentiment: {sentiment}")
    
    # Test Part 2d: Keyword Extraction
    # print("\n--- Part 2d: Keyword Extraction ---")
    # keywords = processor.extract_keywords(sample_text, max_keywords=5)
    # print(f"Keywords: {', '.join(keywords)}")
    
    # Test Part 2e: Writing Improvement
    # print("\n--- Part 2e: Writing Improvement ---")
    # poor_text = "This text not very good. It have grammar errors and unclear meaning."
    # improved = processor.improve_writing(poor_text, aspects=["grammar", "clarity"])
    # print(f"Original: {poor_text}")
    # print(f"Improved: {improved}")
    
    # Test Part 3: Batch Processing
    # print("\n--- Part 3: Batch Processing ---")
    # texts = [
    #     "AI is changing the world.",
    #     "Machine learning enables computers to learn from data.",
    #     "Natural language processing helps computers understand human language."
    # ]
    # summaries = batch_process(processor, texts, "summarize", max_words=10)
    # for i, summary in enumerate(summaries, 1):
    #     print(f"{i}. {summary}")
    
    # Test caching
    # print("\n--- Testing Cache ---")
    # print("Making duplicate request...")
    # summary2 = processor.summarize(sample_text, max_words=30, style="concise")
    # print(f"Second summary (cached): {summary2}")
    # print(f"Metrics after caching: {processor.get_metrics()}")
    
    # Test Part 4: Benchmarking (Bonus)
    # print("\n--- Part 4: Benchmarking (Bonus) ---")
    # benchmark_results = benchmark_processor(sample_text)
    # print(f"Benchmark results: {benchmark_results}")
    
    print("\n" + "=" * 70)
    print("Complete all TODOs and uncomment test sections to validate!")
    print("=" * 70)


# ============================================================================
# REFLECTION QUESTIONS
# ============================================================================

"""
After completing this exercise, answer these questions:

1. Why is caching important for a text processing system?
   Your answer:

2. How would you handle API rate limits in a production system?
   Your answer:

3. What are the tradeoffs between using GPT-3.5-turbo vs GPT-4 for these tasks?
   Your answer:

4. How could you improve error handling for network failures?
   Your answer:

5. What additional metrics would be useful to track?
   Your answer:

6. How would you modify this class to support multiple LLM providers (Anthropic, Cohere)?
   Your answer:

7. What security considerations should you keep in mind when processing user text?
   Your answer:
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
