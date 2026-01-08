"""
Exercise 2: API Parameter Tuning - SOLUTION

This solution demonstrates best practices for experimenting with OpenAI API parameters
and understanding their effects on model behavior.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import json
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# PART 1: Temperature Experiments (15 min)
# ============================================================================

def experiment_with_temperature():
    """
    Experiment with different temperature values to understand randomness control.
    """
    
    prompt = "Write a creative opening line for a science fiction story."
    temperatures = [0, 0.3, 0.7, 1.0, 1.5, 2.0]
    results = {}
    
    print("Experimenting with temperature values...")
    
    for temp in temperatures:
        print(f"\nTesting temperature={temp}")
        responses = []
        
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=50
                )
                
                response_text = response.choices[0].message.content
                responses.append(response_text)
                print(f"  Response {i+1}: {response_text}")
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error in request {i+1}: {str(e)}")
                responses.append(None)
        
        results[temp] = responses
    
    return results


def analyze_temperature_results(results):
    """
    Analyze the temperature experiment results.
    """
    
    print("\n" + "="*70)
    print("TEMPERATURE ANALYSIS")
    print("="*70)
    
    for temp, responses in results.items():
        # Filter out None values (errors)
        valid_responses = [r for r in responses if r is not None]
        
        if not valid_responses:
            print(f"\nTemperature {temp}: No valid responses")
            continue
        
        # Count unique responses
        unique_responses = len(set(valid_responses))
        
        # Calculate average length
        avg_length = sum(len(r.split()) for r in valid_responses) / len(valid_responses)
        
        # Analyze first words
        first_words = [r.split()[0] if r.split() else "" for r in valid_responses]
        unique_first_words = len(set(first_words))
        
        print(f"\nTemperature {temp}:")
        print(f"  Unique responses: {unique_responses}/{len(valid_responses)}")
        print(f"  Unique first words: {unique_first_words}")
        print(f"  Average length: {avg_length:.1f} words")
        print(f"  Variation: {'Low' if unique_responses == 1 else 'Medium' if unique_responses == 2 else 'High'}")
    
    print("\nKey Insights:")
    print("- Temperature 0: Completely deterministic, same output every time")
    print("- Temperature 0-0.3: Minimal variation, good for factual tasks")
    print("- Temperature 0.7-1.0: Balanced creativity, good for general use")
    print("- Temperature 1.5-2.0: High variation, good for brainstorming")


# ============================================================================
# PART 2: Max Tokens Control (10 min)
# ============================================================================

def test_max_tokens():
    """
    Test the impact of max_tokens parameter on response length and completeness.
    """
    
    prompt = "Explain the concept of machine learning in detail."
    max_tokens_values = [10, 50, 100, 200, 500]
    results = []
    
    print("\nTesting max_tokens parameter...")
    
    for max_tokens in max_tokens_values:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            actual_tokens = response.usage.completion_tokens
            
            # Check if response seems complete (doesn't end mid-sentence)
            is_complete = (
                response_text.endswith(('.', '!', '?')) and 
                actual_tokens < max_tokens
            )
            
            result = {
                "max_tokens_set": max_tokens,
                "actual_tokens_used": actual_tokens,
                "response_text": response_text,
                "is_complete": is_complete
            }
            
            results.append(result)
            
            print(f"\nMax tokens: {max_tokens}")
            print(f"  Actual tokens used: {actual_tokens}")
            print(f"  Complete: {is_complete}")
            print(f"  Response: {response_text[:100]}...")
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error with max_tokens={max_tokens}: {str(e)}")
    
    return results


# ============================================================================
# PART 3: Top_p vs Temperature (10 min)
# ============================================================================

def compare_sampling_methods():
    """
    Compare temperature-based sampling vs nucleus (top_p) sampling.
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
    
    print("\nComparing sampling methods...")
    
    for config in configs:
        print(f"\nTesting {config['name']} (temp={config['temperature']}, top_p={config['top_p']})")
        responses = []
        
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    max_tokens=100
                )
                
                response_text = response.choices[0].message.content
                responses.append(response_text)
                print(f"  Response {i+1}: {response_text[:80]}...")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                responses.append(None)
        
        results[config['name']] = {
            "config": config,
            "responses": responses
        }
    
    # Analysis
    print("\n" + "="*70)
    print("SAMPLING METHOD ANALYSIS")
    print("="*70)
    
    for name, data in results.items():
        valid_responses = [r for r in data['responses'] if r is not None]
        unique_count = len(set(valid_responses))
        
        print(f"\n{name}:")
        print(f"  Config: temp={data['config']['temperature']}, top_p={data['config']['top_p']}")
        print(f"  Unique responses: {unique_count}/{len(valid_responses)}")
        
    return results


# ============================================================================
# PART 4: Token Usage and Cost Tracking (10 min)
# ============================================================================

def track_token_usage():
    """
    Track token usage and calculate costs for different prompt lengths.
    """
    
    prompts = [
        "Hi",
        "What is Python?",
        "Explain object-oriented programming with examples.",
        "Write a detailed guide on RESTful API design principles and best practices.",
        "Provide a comprehensive analysis of the differences between SQL and NoSQL databases, including use cases, advantages, disadvantages, and specific examples of each type."
    ]
    
    usage_data = []
    
    print("\nTracking token usage and costs...")
    
    for i, prompt in enumerate(prompts, 1):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Calculate cost for GPT-3.5-turbo
            cost = calculate_cost(prompt_tokens, completion_tokens, model="gpt-3.5-turbo")
            
            data = {
                "prompt_num": i,
                "prompt_length_chars": len(prompt),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost
            }
            
            usage_data.append(data)
            
            print(f"\nPrompt {i} (length: {len(prompt)} chars):")
            print(f"  Prompt tokens: {prompt_tokens}")
            print(f"  Completion tokens: {completion_tokens}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Cost: ${cost:.6f}")
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error with prompt {i}: {str(e)}")
    
    # Calculate totals
    total_cost = sum(d['cost'] for d in usage_data)
    total_tokens = sum(d['total_tokens'] for d in usage_data)
    
    print("\n" + "="*70)
    print("USAGE SUMMARY")
    print("="*70)
    print(f"Total requests: {len(usage_data)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Average cost per request: ${total_cost/len(usage_data):.6f}")
    
    # Identify most expensive
    most_expensive = max(usage_data, key=lambda x: x['cost'])
    print(f"\nMost expensive request: Prompt {most_expensive['prompt_num']}")
    print(f"  Cost: ${most_expensive['cost']:.6f}")
    print(f"  Total tokens: {most_expensive['total_tokens']}")
    
    return {
        "usage_data": usage_data,
        "total_cost": total_cost,
        "total_tokens": total_tokens
    }


def calculate_cost(prompt_tokens, completion_tokens, model="gpt-4"):
    """
    Calculate API costs based on token counts.
    
    Pricing (as of 2025):
    - GPT-4: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
    - GPT-3.5-turbo: $0.0015 per 1K prompt tokens, $0.002 per 1K completion tokens
    """
    
    pricing = {
        "gpt-4": {
            "prompt": 0.03 / 1000,
            "completion": 0.06 / 1000
        },
        "gpt-3.5-turbo": {
            "prompt": 0.0015 / 1000,
            "completion": 0.002 / 1000
        }
    }
    
    if model not in pricing:
        model = "gpt-4"  # Default to GPT-4 pricing
    
    prompt_cost = prompt_tokens * pricing[model]["prompt"]
    completion_cost = completion_tokens * pricing[model]["completion"]
    
    return prompt_cost + completion_cost


# ============================================================================
# PART 5: Parameter Optimization Challenge (Bonus)
# ============================================================================

def optimize_for_task(task_type):
    """
    Suggest optimal parameters for different task types.
    """
    
    recommendations = {
        "factual": {
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.1,
                "max_tokens": 500,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "reasoning": (
                "For factual questions, we want consistent, accurate responses. "
                "Low temperature (0.1) ensures deterministic outputs. "
                "Low top_p further constrains randomness. "
                "No penalties needed as we want straightforward answers."
            )
        },
        "creative": {
            "parameters": {
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 1000,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3
            },
            "reasoning": (
                "Creative writing benefits from higher randomness (temp=1.0). "
                "High top_p allows diverse word choices. "
                "Frequency penalty reduces repetition. "
                "Presence penalty encourages exploring new topics. "
                "Higher max_tokens allows for longer, more developed content."
            )
        },
        "code": {
            "parameters": {
                "temperature": 0.2,
                "top_p": 0.1,
                "max_tokens": 2000,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "reasoning": (
                "Code generation requires precision and correctness. "
                "Very low temperature (0.2) ensures consistent, reliable outputs. "
                "Low top_p further constrains randomness. "
                "Higher max_tokens accommodates longer code blocks. "
                "No penalties as code often requires repetition of patterns and keywords."
            )
        },
        "summary": {
            "parameters": {
                "temperature": 0.3,
                "top_p": 0.5,
                "max_tokens": 300,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.0
            },
            "reasoning": (
                "Summarization needs consistency but some variation in word choice. "
                "Low-medium temperature (0.3) provides this balance. "
                "Medium top_p allows some lexical diversity. "
                "Limited max_tokens enforces conciseness. "
                "Frequency penalty helps avoid repetitive phrasing."
            )
        }
    }
    
    if task_type not in recommendations:
        return {"error": f"Unknown task type: {task_type}"}
    
    return recommendations[task_type]


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def run_tests():
    """
    Run all exercise tests.
    """
    
    print("=" * 70)
    print("EXERCISE 2: API PARAMETER TUNING - SOLUTION")
    print("=" * 70)
    
    # Test Part 1
    print("\n--- Part 1: Temperature Experiments ---")
    temp_results = experiment_with_temperature()
    analyze_temperature_results(temp_results)
    
    # Test Part 2
    print("\n--- Part 2: Max Tokens Control ---")
    max_tokens_results = test_max_tokens()
    
    # Test Part 3
    print("\n--- Part 3: Top_p vs Temperature ---")
    sampling_results = compare_sampling_methods()
    
    # Test Part 4
    print("\n--- Part 4: Token Usage Tracking ---")
    usage_summary = track_token_usage()
    
    # Test Part 5 (Bonus)
    print("\n--- Part 5: Parameter Optimization (Bonus) ---")
    for task_type in ["factual", "creative", "code", "summary"]:
        recommendations = optimize_for_task(task_type)
        print(f"\n{task_type.upper()}:")
        print(f"  Parameters: {recommendations['parameters']}")
        print(f"  Reasoning: {recommendations['reasoning']}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


# ============================================================================
# REFLECTION QUESTIONS - ANSWERS
# ============================================================================

"""
REFLECTION ANSWERS:

1. When would you use a high temperature value (>1.0)?
   
   Use high temperature (1.0-2.0) for:
   - Brainstorming and idea generation
   - Creative writing where diverse outputs are desired
   - Exploring multiple solution approaches
   - When you want maximum variety across multiple requests
   
   Caution: Very high temperatures can produce less coherent outputs.

2. What's the difference between setting temperature=0 vs temperature=0.1?
   
   - Temperature=0: Completely deterministic. Always selects the most likely next token.
     Same input will always produce identical output.
   
   - Temperature=0.1: Nearly deterministic with minimal randomness. Allows tiny variations
     but still produces very consistent outputs. Useful when you want consistency with
     slight natural language variation.

3. When would you prefer top_p over temperature for controlling randomness?
   
   Prefer top_p (nucleus sampling) when:
   - You want more consistent quality across different contexts
   - Temperature alone produces too much variation in quality
   - You need to constrain the "vocabulary" of possible tokens without affecting
     the distribution over those tokens
   
   top_p can provide better quality control because it eliminates low-probability
   tokens that might not make sense, regardless of temperature.

4. How does max_tokens affect cost, and when should you set it conservatively?
   
   Impact on cost:
   - You're only charged for actual tokens used, not max_tokens limit
   - However, max_tokens controls how long responses can be
   - Longer responses = more completion tokens = higher cost
   
   Set conservatively when:
   - Cost is a critical concern
   - You need short, concise responses
   - You're making many requests (batch processing)
   - You have strict latency requirements (fewer tokens = faster response)
   
   Note: Too low max_tokens can truncate responses mid-sentence.

5. What parameters would you use for a chatbot that needs to give consistent,
   factual answers?
   
   Recommended parameters:
   - temperature: 0.1-0.3 (low randomness for consistency)
   - top_p: 0.1-0.5 (constrain token choices)
   - max_tokens: 300-500 (sufficient for detailed answers without verbosity)
   - frequency_penalty: 0.0 (allow natural repetition of key facts)
   - presence_penalty: 0.0 (no need to encourage topic diversity)
   
   Additional strategies:
   - Use system message to enforce factual, helpful tone
   - Implement fact-checking or RAG (Retrieval-Augmented Generation)
   - Add citations or sources when possible

6. What patterns did you notice in token usage across different prompt lengths?
   
   Key patterns observed:
   
   - Prompt tokens scale linearly with character count (roughly 4 chars = 1 token)
   - Very short prompts (<10 chars) have overhead from system messages
   - Completion tokens don't correlate directly with prompt length
   - Longer, more complex prompts may generate longer responses
   - Token efficiency improves with clear, specific prompts vs vague ones
   
   Cost implications:
   - Both prompt and completion tokens contribute to cost
   - Prompt tokens have lower per-token cost than completion tokens
   - Total cost is dominated by completion tokens for typical requests
   - Batch processing with varied prompt lengths shows predictable cost patterns
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
