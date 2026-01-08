"""
Exercise 2: API Parameter Tuning

Time: 45 minutes
Difficulty: Beginner-Intermediate
Focus: Understanding OpenAI API parameters and their effects

OBJECTIVES:
1. Experiment with different temperature values
2. Understand max_tokens impact on responses
3. Compare top_p vs temperature
4. Measure token usage and costs
5. Implement parameter optimization strategies

SETUP:
- Ensure your .env file has OPENAI_API_KEY set
- Install required packages: openai, python-dotenv
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# PART 1: Temperature Experiments (15 min)
# ============================================================================

def experiment_with_temperature():
    """
    TODO: Complete this function to experiment with different temperature values.
    
    Requirements:
    1. Use the same prompt for all requests
    2. Test temperature values: 0, 0.3, 0.7, 1.0, 1.5, 2.0
    3. Make 3 requests for each temperature value
    4. Store and compare the results
    5. Calculate diversity metrics (unique first words, unique responses)
    
    The prompt should be something creative like:
    "Write a creative opening line for a science fiction story."
    
    Return a dictionary with temperature as key and list of responses as value.
    """
    
    prompt = "Write a creative opening line for a science fiction story."
    temperatures = [0, 0.3, 0.7, 1.0, 1.5, 2.0]
    results = {}
    
    # TODO: Implement the experiment
    # Hint: Use client.chat.completions.create() with different temperature values
    # Store responses in results dictionary
    
    pass  # Remove this and add your code


def analyze_temperature_results(results):
    """
    TODO: Analyze the temperature experiment results.
    
    Requirements:
    1. Count unique responses for each temperature
    2. Calculate the average response length for each temperature
    3. Identify which temperatures produced the most variation
    4. Print a summary report
    
    Args:
        results: Dictionary from experiment_with_temperature()
    """
    
    # TODO: Implement analysis
    # Calculate metrics and print findings
    
    pass


# ============================================================================
# PART 2: Max Tokens Control (10 min)
# ============================================================================

def test_max_tokens():
    """
    TODO: Test the impact of max_tokens parameter.
    
    Requirements:
    1. Use a prompt that naturally generates long responses:
       "Explain the concept of machine learning in detail."
    2. Test max_tokens values: 10, 50, 100, 200, 500
    3. Record actual tokens used vs max_tokens set
    4. Observe how responses get truncated
    
    Return a list of dictionaries with:
    - max_tokens_set
    - actual_tokens_used
    - response_text
    - is_complete (boolean indicating if response seems complete)
    """
    
    prompt = "Explain the concept of machine learning in detail."
    max_tokens_values = [10, 50, 100, 200, 500]
    results = []
    
    # TODO: Implement the test
    # For each max_tokens value, make a request and record results
    
    pass


# ============================================================================
# PART 3: Top_p vs Temperature (10 min)
# ============================================================================

def compare_sampling_methods():
    """
    TODO: Compare temperature-based sampling vs nucleus (top_p) sampling.
    
    Requirements:
    1. Use the same creative prompt for all tests
    2. Test configurations:
       - temperature=0.7, top_p=1.0 (temperature only)
       - temperature=1.0, top_p=0.5 (top_p only)
       - temperature=0.7, top_p=0.9 (both)
       - temperature=0.1, top_p=0.1 (both low)
       - temperature=1.5, top_p=0.95 (both high)
    3. Make 3 requests for each configuration
    4. Compare the creativity and coherence
    
    Return results with configuration as key and responses as value.
    """
    
    prompt = "Describe an unusual invention that doesn't exist yet."
    
    configs = [
        {"temperature": 0.7, "top_p": 1.0, "name": "temperature_only"},
        {"temperature": 1.0, "top_p": 0.5, "name": "top_p_only"},
        {"temperature": 0.7, "top_p": 0.9, "name": "balanced"},
        {"temperature": 0.1, "top_p": 0.1, "name": "both_low"},
        {"temperature": 1.5, "top_p": 0.95, "name": "both_high"},
    ]
    
    results = {}
    
    # TODO: Implement comparison
    # Test each configuration and store results
    
    pass


# ============================================================================
# PART 4: Token Usage and Cost Tracking (10 min)
# ============================================================================

def track_token_usage():
    """
    TODO: Implement token usage tracking and cost calculation.
    
    Requirements:
    1. Make 5 different requests with varying prompt lengths
    2. Track:
       - Prompt tokens
       - Completion tokens
       - Total tokens
       - Estimated cost (use $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens for GPT-4)
    3. Calculate total cost for all requests
    4. Identify which requests were most expensive and why
    
    Return a summary dictionary with usage statistics.
    """
    
    prompts = [
        "Hi",
        "What is Python?",
        "Explain object-oriented programming with examples.",
        "Write a detailed guide on RESTful API design principles and best practices.",
        "Provide a comprehensive analysis of the differences between SQL and NoSQL databases, including use cases, advantages, disadvantages, and specific examples of each type."
    ]
    
    usage_data = []
    
    # TODO: Implement tracking
    # For each prompt, make a request and extract usage information
    # Calculate costs based on token usage
    
    pass


def calculate_cost(prompt_tokens, completion_tokens, model="gpt-4"):
    """
    Helper function to calculate API costs.
    
    TODO: Implement cost calculation based on token counts.
    
    Pricing (as of 2025):
    - GPT-4: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
    - GPT-3.5-turbo: $0.0015 per 1K prompt tokens, $0.002 per 1K completion tokens
    
    Args:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        model: Model name (default: "gpt-4")
    
    Returns:
        Cost in dollars (float)
    """
    
    # TODO: Implement cost calculation
    
    pass


# ============================================================================
# PART 5: Parameter Optimization Challenge (Bonus)
# ============================================================================

def optimize_for_task(task_type):
    """
    BONUS TODO: Suggest optimal parameters for different task types.
    
    Task types:
    1. "factual" - Answering factual questions (needs consistency)
    2. "creative" - Creative writing (needs diversity)
    3. "code" - Code generation (needs precision)
    4. "summary" - Text summarization (needs conciseness)
    
    For each task type, return recommended:
    - temperature
    - top_p
    - max_tokens
    - frequency_penalty
    - presence_penalty
    
    Also provide reasoning for each choice.
    
    Args:
        task_type: One of ["factual", "creative", "code", "summary"]
    
    Returns:
        Dictionary with recommended parameters and reasoning
    """
    
    # TODO: Implement parameter recommendations
    
    recommendations = {
        "factual": {
            "parameters": {},
            "reasoning": ""
        },
        "creative": {
            "parameters": {},
            "reasoning": ""
        },
        "code": {
            "parameters": {},
            "reasoning": ""
        },
        "summary": {
            "parameters": {},
            "reasoning": ""
        }
    }
    
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
    print("EXERCISE 2: API PARAMETER TUNING")
    print("=" * 70)
    
    # Test Part 1
    # print("\n--- Part 1: Temperature Experiments ---")
    # temp_results = experiment_with_temperature()
    # analyze_temperature_results(temp_results)
    
    # Test Part 2
    # print("\n--- Part 2: Max Tokens Control ---")
    # max_tokens_results = test_max_tokens()
    # for result in max_tokens_results:
    #     print(f"Max tokens set: {result['max_tokens_set']}, "
    #           f"Used: {result['actual_tokens_used']}, "
    #           f"Complete: {result['is_complete']}")
    
    # Test Part 3
    # print("\n--- Part 3: Top_p vs Temperature ---")
    # sampling_results = compare_sampling_methods()
    # for config_name, responses in sampling_results.items():
    #     print(f"\n{config_name}:")
    #     print(f"  Unique responses: {len(set(responses))}/{len(responses)}")
    
    # Test Part 4
    # print("\n--- Part 4: Token Usage Tracking ---")
    # usage_summary = track_token_usage()
    # print(f"Total cost: ${usage_summary['total_cost']:.4f}")
    
    # Test Part 5 (Bonus)
    # print("\n--- Part 5: Parameter Optimization (Bonus) ---")
    # for task_type in ["factual", "creative", "code", "summary"]:
    #     recommendations = optimize_for_task(task_type)
    #     print(f"\n{task_type.upper()}:")
    #     print(f"  Parameters: {recommendations['parameters']}")
    #     print(f"  Reasoning: {recommendations['reasoning']}")
    
    print("\n" + "=" * 70)
    print("Complete all TODOs and uncomment test sections to validate!")
    print("=" * 70)


# ============================================================================
# REFLECTION QUESTIONS
# ============================================================================

"""
After completing this exercise, answer these questions:

1. When would you use a high temperature value (>1.0)?
   Your answer:

2. What's the difference between setting temperature=0 vs temperature=0.1?
   Your answer:

3. When would you prefer top_p over temperature for controlling randomness?
   Your answer:

4. How does max_tokens affect cost, and when should you set it conservatively?
   Your answer:

5. What parameters would you use for a chatbot that needs to give consistent,
   factual answers?
   Your answer:

6. What patterns did you notice in token usage across different prompt lengths?
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
