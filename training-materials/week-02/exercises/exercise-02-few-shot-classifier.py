"""
Exercise 2: Few-Shot Classifier Implementation

Time: 60 minutes
Difficulty: Intermediate
Focus: Building production-ready few-shot classifiers

OBJECTIVES:
1. Build a flexible few-shot classification system
2. Implement dynamic example selection
3. Handle multiple classification tasks
4. Measure and optimize performance
5. Deploy as a reusable component

SETUP:
- Ensure your .env file has OPENAI_API_KEY set
- Install required packages: openai, python-dotenv, numpy
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional
import json
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# PART 1: Basic Few-Shot Classifier (15 min)
# ============================================================================

class FewShotClassifier:
    """
    A reusable few-shot classification system.
    
    TODO: Complete this class with the following functionality:
    1. Initialize with examples and categories
    2. Build prompts dynamically
    3. Make classification requests
    4. Parse and validate responses
    5. Track performance metrics
    """
    
    def __init__(self, categories: List[str], examples: List[Tuple[str, str]] = None):
        """
        Initialize the classifier.
        
        Args:
            categories: List of possible categories
            examples: List of (text, category) tuples for few-shot learning
        
        TODO: 
        - Store categories and examples
        - Initialize metrics tracking
        - Validate that examples match categories
        """
        self.categories = categories
        self.examples = examples or []
        self.client = client
        
        # TODO: Initialize metrics
        self.metrics = {
            "total_classifications": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        # TODO: Initialize cache
        self.cache = {}
        
        # TODO: Validate examples
        self._validate_examples()
    
    def _validate_examples(self):
        """
        Validate that all examples have categories from the allowed list.
        
        TODO: Implement validation
        """
        pass
    
    def _build_prompt(self, text: str, examples: List[Tuple[str, str]] = None) -> str:
        """
        Build a few-shot prompt for classification.
        
        Args:
            text: Text to classify
            examples: Optional specific examples to use (if None, use self.examples)
        
        Returns:
            Formatted prompt string
        
        TODO: Create a well-structured prompt with:
        1. Task description
        2. Category list
        3. Few-shot examples
        4. Input to classify
        5. Output format specification
        """
        pass
    
    def classify(self, text: str, use_cache: bool = True) -> Dict[str, any]:
        """
        Classify a text using few-shot learning.
        
        Args:
            text: Text to classify
            use_cache: Whether to use cached results
        
        Returns:
            Dictionary with:
            - category: The predicted category
            - confidence: Optional confidence score
            - reasoning: Optional explanation
        
        TODO: Implement classification with:
        1. Cache checking
        2. Prompt building
        3. API call
        4. Response parsing
        5. Metrics updating
        6. Error handling
        """
        pass
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Classify multiple texts.
        
        TODO: Implement batch classification with progress tracking
        """
        pass
    
    def add_example(self, text: str, category: str):
        """
        Add a new example to the classifier.
        
        TODO: Implement with validation
        """
        pass
    
    def get_metrics(self) -> Dict[str, any]:
        """
        Get performance metrics.
        
        TODO: Return comprehensive metrics including:
        - Total classifications
        - Cache hit rate
        - API calls made
        - Error rate
        """
        pass


# ============================================================================
# PART 2: Dynamic Example Selection (15 min)
# ============================================================================

class SmartFewShotClassifier(FewShotClassifier):
    """
    Enhanced classifier with intelligent example selection.
    
    TODO: Extend the base classifier with:
    1. Similarity-based example selection
    2. Diverse example selection
    3. Performance-based example ranking
    """
    
    def __init__(self, categories: List[str], example_pool: List[Tuple[str, str]]):
        """
        Initialize with a pool of examples to select from.
        
        TODO: 
        - Call parent init
        - Store example pool
        - Initialize selection strategy
        """
        super().__init__(categories, [])
        self.example_pool = example_pool
        # TODO: Add any additional initialization
    
    def _select_examples_by_length(self, text: str, n: int = 3) -> List[Tuple[str, str]]:
        """
        Select examples with similar length to the input text.
        
        TODO: Implement length-based selection:
        1. Calculate text length (words or chars)
        2. Calculate length difference for each example
        3. Sort by similarity
        4. Return top n examples
        """
        pass
    
    def _select_examples_by_keywords(self, text: str, n: int = 3) -> List[Tuple[str, str]]:
        """
        Select examples with overlapping keywords.
        
        TODO: Implement keyword-based selection:
        1. Extract keywords from input text
        2. Calculate keyword overlap with each example
        3. Sort by overlap score
        4. Return top n examples
        """
        pass
    
    def _select_diverse_examples(self, n: int = 5) -> List[Tuple[str, str]]:
        """
        Select diverse examples covering all categories.
        
        TODO: Implement diverse selection:
        1. Group examples by category
        2. Select balanced samples from each category
        3. Ensure variety in length and content
        """
        pass
    
    def classify(self, text: str, selection_strategy: str = "keywords", 
                 n_examples: int = 3) -> Dict[str, any]:
        """
        Classify with dynamic example selection.
        
        Args:
            text: Text to classify
            selection_strategy: "length", "keywords", or "diverse"
            n_examples: Number of examples to use
        
        TODO: Implement classification with selected examples:
        1. Select appropriate examples based on strategy
        2. Build prompt with selected examples
        3. Make classification
        4. Return result
        """
        pass


# ============================================================================
# PART 3: Multi-Task Classifier (15 min)
# ============================================================================

class MultiTaskClassifier:
    """
    Classifier that can handle multiple different classification tasks.
    
    TODO: Implement a system that manages multiple classification tasks:
    - Sentiment analysis
    - Intent detection
    - Topic classification
    - Language detection
    - Priority/urgency classification
    """
    
    def __init__(self):
        """
        Initialize with multiple task-specific classifiers.
        
        TODO: Set up classifiers for different tasks
        """
        self.tasks = {}
        # TODO: Initialize task-specific classifiers
    
    def register_task(self, task_name: str, categories: List[str], 
                     examples: List[Tuple[str, str]]):
        """
        Register a new classification task.
        
        TODO: Create and store a classifier for this task
        """
        pass
    
    def classify(self, text: str, task: str) -> Dict[str, any]:
        """
        Classify text for a specific task.
        
        TODO: Route to appropriate task classifier
        """
        pass
    
    def classify_multi(self, text: str, tasks: List[str]) -> Dict[str, Dict]:
        """
        Classify text across multiple tasks simultaneously.
        
        TODO: Run multiple classifications and combine results
        """
        pass


# ============================================================================
# PART 4: Evaluation and Testing (15 min)
# ============================================================================

def evaluate_classifier(classifier, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Evaluate classifier performance on test data.
    
    Args:
        classifier: Classifier instance to test
        test_data: List of (text, true_category) tuples
    
    Returns:
        Dictionary with evaluation metrics
    
    TODO: Implement comprehensive evaluation:
    1. Run classifications on test data
    2. Calculate accuracy
    3. Build confusion matrix
    4. Calculate per-category precision/recall
    5. Measure API efficiency (cache hits, etc.)
    """
    
    results = {
        "accuracy": 0.0,
        "precision_per_category": {},
        "recall_per_category": {},
        "f1_per_category": {},
        "confusion_matrix": {},
        "total_correct": 0,
        "total_tests": len(test_data)
    }
    
    # TODO: Implement evaluation logic
    
    return results


def compare_strategies(text_samples: List[str], ground_truth: List[str],
                      example_pool: List[Tuple[str, str]], 
                      categories: List[str]) -> Dict[str, Dict]:
    """
    Compare different example selection strategies.
    
    TODO: Test multiple strategies and compare:
    1. Random selection
    2. Length-based selection
    3. Keyword-based selection
    4. Diverse selection
    5. No examples (zero-shot)
    
    Return comparative results showing which works best.
    """
    pass


def optimize_n_examples(classifier, test_data: List[Tuple[str, str]], 
                       max_n: int = 10) -> Dict[int, float]:
    """
    Find optimal number of examples to use.
    
    TODO: Test with different numbers of examples (1 to max_n)
    and find the sweet spot between accuracy and efficiency.
    """
    pass


# ============================================================================
# PART 5: Production Deployment (Bonus)
# ============================================================================

class ProductionClassifier(SmartFewShotClassifier):
    """
    Production-ready classifier with additional features.
    
    TODO: Add production features:
    1. Retry logic for failed API calls
    2. Rate limiting
    3. Detailed logging
    4. Monitoring and alerts
    5. A/B testing support
    6. Model versioning
    """
    
    def __init__(self, categories: List[str], example_pool: List[Tuple[str, str]],
                 max_retries: int = 3, rate_limit: int = 60):
        """
        Initialize production classifier with reliability features.
        
        TODO: Add retry and rate limiting configuration
        """
        super().__init__(categories, example_pool)
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        # TODO: Initialize rate limiter, logger, etc.
    
    def classify_with_retry(self, text: str) -> Dict[str, any]:
        """
        Classify with automatic retry on failure.
        
        TODO: Implement exponential backoff retry logic
        """
        pass
    
    def monitor_performance(self) -> Dict[str, any]:
        """
        Get real-time performance monitoring data.
        
        TODO: Return metrics suitable for dashboard/alerting
        """
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
    print("EXERCISE 2: FEW-SHOT CLASSIFIER IMPLEMENTATION")
    print("=" * 70)
    
    # Sample data for testing
    categories = ["Positive", "Negative", "Neutral"]
    
    examples = [
        ("This product is amazing! Best purchase ever.", "Positive"),
        ("Love it! Exceeded my expectations.", "Positive"),
        ("Terrible quality. Complete waste of money.", "Negative"),
        ("Very disappointed. Does not work as advertised.", "Negative"),
        ("It's okay. Nothing special but does the job.", "Neutral"),
        ("Average product. Works as described.", "Neutral"),
    ]
    
    test_texts = [
        "Absolutely fantastic! Would buy again.",
        "Don't waste your money on this garbage.",
        "Pretty decent. No complaints.",
    ]
    
    # Test Part 1: Basic Classifier
    # print("\n--- Part 1: Basic Few-Shot Classifier ---")
    # classifier = FewShotClassifier(categories, examples)
    # for text in test_texts:
    #     result = classifier.classify(text)
    #     print(f"\nText: {text}")
    #     print(f"Classification: {result}")
    
    # print(f"\nMetrics: {classifier.get_metrics()}")
    
    # Test Part 2: Smart Classifier
    # print("\n--- Part 2: Smart Example Selection ---")
    # smart_classifier = SmartFewShotClassifier(categories, examples)
    # 
    # for strategy in ["length", "keywords", "diverse"]:
    #     print(f"\nUsing {strategy} strategy:")
    #     result = smart_classifier.classify(test_texts[0], selection_strategy=strategy)
    #     print(f"Result: {result}")
    
    # Test Part 3: Multi-Task Classifier
    # print("\n--- Part 3: Multi-Task Classification ---")
    # multi_classifier = MultiTaskClassifier()
    # 
    # # Register tasks
    # multi_classifier.register_task("sentiment", ["Positive", "Negative", "Neutral"], examples)
    # # Add more task registrations
    # 
    # # Test multi-task classification
    # text = "This is great but shipping was slow"
    # results = multi_classifier.classify_multi(text, ["sentiment", "intent"])
    # print(f"Multi-task results: {results}")
    
    # Test Part 4: Evaluation
    # print("\n--- Part 4: Evaluation ---")
    # test_data = [
    #     ("Amazing product!", "Positive"),
    #     ("Terrible experience.", "Negative"),
    #     ("It's okay.", "Neutral"),
    # ]
    # 
    # evaluation = evaluate_classifier(classifier, test_data)
    # print(f"Evaluation Results:")
    # print(f"  Accuracy: {evaluation['accuracy']*100:.1f}%")
    # print(f"  Total Correct: {evaluation['total_correct']}/{evaluation['total_tests']}")
    
    print("\n" + "=" * 70)
    print("Complete all TODOs and uncomment test sections to validate!")
    print("=" * 70)


# ============================================================================
# REFLECTION QUESTIONS
# ============================================================================

"""
After completing this exercise, answer these questions:

1. How does the number of examples affect classification accuracy?
   Your answer:

2. Which example selection strategy worked best for your use case?
   Your answer:

3. What are the tradeoffs between few-shot and fine-tuning?
   Your answer:

4. How would you handle ambiguous cases where the text could fit multiple categories?
   Your answer:

5. What production considerations did you encounter (error handling, rate limits, etc.)?
   Your answer:

6. How would you continuously improve the classifier based on production data?
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
