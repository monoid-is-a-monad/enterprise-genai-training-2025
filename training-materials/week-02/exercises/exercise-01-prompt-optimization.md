# Exercise 1: Prompt Optimization Challenge

**Time:** 60 minutes  
**Difficulty:** Intermediate  
**Focus:** Iteratively improving prompts for better results

---

## Objective

Learn the art of prompt optimization by starting with a basic prompt and systematically improving it through multiple iterations. You'll measure the impact of each change and develop an intuition for what makes prompts effective.

---

## Scenario

You're building a customer support email classifier that needs to:
1. Categorize incoming emails by type (Technical, Billing, Product Info, Complaint, General)
2. Determine urgency level (High, Medium, Low)
3. Extract key entities (product names, order numbers, dates)
4. Suggest a response template to use

Your task is to create and optimize a prompt that accomplishes all these goals accurately and consistently.

---

## Part 1: Baseline Prompt (10 min)

### Task 1.1: Create Your Initial Prompt

Start with a simple, direct prompt. Don't overthink it—just write what comes naturally.

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def baseline_classifier(email_text):
    """
    TODO: Write your initial prompt here.
    Keep it simple and direct.
    """
    
    prompt = f"""
    [YOUR PROMPT HERE]
    
    Email: {email_text}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    
    return response.choices[0].message.content

# Test with sample emails
test_emails = [
    "My order #12345 arrived damaged. I need a replacement ASAP.",
    "What are the differences between the Pro and Enterprise plans?",
    "I was charged twice for last month's subscription.",
    "How do I export my data to CSV format?",
    "This is the worst service ever! Cancel my account immediately!"
]

print("BASELINE RESULTS:")
print("=" * 80)
for i, email in enumerate(test_emails, 1):
    result = baseline_classifier(email)
    print(f"\n{i}. Email: {email}")
    print(f"   Result: {result}")
```

### Task 1.2: Evaluate Your Baseline

After running your baseline, identify issues:
- [ ] Is the output format consistent?
- [ ] Are all required fields present?
- [ ] Is the categorization accurate?
- [ ] Can you easily parse the output programmatically?

**Document issues you found:**
```
Issue 1: 
Issue 2:
Issue 3:
```

---

## Part 2: Iteration 1 - Structure the Output (10 min)

### Task 2.1: Add Output Format Specification

Improve your prompt by clearly defining the expected output format.

```python
def iteration1_classifier(email_text):
    """
    TODO: Improve the prompt by specifying output format.
    
    Consider using:
    - JSON format for structured data
    - Clear field names
    - Explicit format instructions
    """
    
    prompt = f"""
    [YOUR IMPROVED PROMPT HERE - FOCUS ON OUTPUT FORMAT]
    
    Email: {email_text}
    
    Provide your response as a JSON object with these fields:
    - category
    - urgency
    - entities
    - suggested_template
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250
    )
    
    return response.choices[0].message.content

print("\nITERATION 1 RESULTS:")
print("=" * 80)
for i, email in enumerate(test_emails, 1):
    result = iteration1_classifier(email)
    print(f"\n{i}. Email: {email}")
    print(f"   Result: {result}")
```

### Task 2.2: Measure Improvement

Compare Iteration 1 to Baseline:
- [ ] Is the format more consistent?
- [ ] Can you parse the output as JSON?
- [ ] Are all fields present in every response?

**Improvement notes:**
```
What got better:
What still needs work:
```

---

## Part 3: Iteration 2 - Add Context and Examples (15 min)

### Task 3.1: Enhance with Few-Shot Examples

Add examples to guide the model's behavior.

```python
def iteration2_classifier(email_text):
    """
    TODO: Add few-shot examples to your prompt.
    
    Include 2-3 examples showing:
    - Different email types
    - Correct categorization
    - Proper urgency assessment
    - Expected JSON format
    """
    
    prompt = f"""Classify customer support emails.

Example 1:
Email: "I can't log into my account. This is urgent!"
Output:
{{
  "category": "Technical",
  "urgency": "High",
  "entities": {{"issue": "login problem"}},
  "suggested_template": "technical_support_immediate"
}}

Example 2:
[ADD YOUR SECOND EXAMPLE]

Example 3:
[ADD YOUR THIRD EXAMPLE]

Now classify this email:

Email: {email_text}

Output:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250
    )
    
    return response.choices[0].message.content

print("\nITERATION 2 RESULTS:")
print("=" * 80)
for i, email in enumerate(test_emails, 1):
    result = iteration2_classifier(email)
    print(f"\n{i}. Email: {email}")
    print(f"   Result: {result}")
```

### Task 3.2: Evaluate Example Impact

- [ ] Did examples improve accuracy?
- [ ] Is the model following the format better?
- [ ] Are edge cases handled better?

**Impact assessment:**
```
Accuracy improvement: [better/same/worse]
Consistency improvement: [better/same/worse]
Key insight:
```

---

## Part 4: Iteration 3 - Define Categories and Rules (10 min)

### Task 4.1: Add Clear Definitions

Provide explicit definitions for categories and urgency levels.

```python
def iteration3_classifier(email_text):
    """
    TODO: Add category definitions and urgency rules.
    
    Define:
    - What each category means
    - Criteria for urgency levels
    - Edge case handling
    """
    
    prompt = f"""Classify customer support emails into categories with urgency levels.

CATEGORY DEFINITIONS:
- Technical: [YOUR DEFINITION]
- Billing: [YOUR DEFINITION]
- Product Info: [YOUR DEFINITION]
- Complaint: [YOUR DEFINITION]
- General: [YOUR DEFINITION]

URGENCY LEVELS:
- High: [WHEN TO USE]
- Medium: [WHEN TO USE]
- Low: [WHEN TO USE]

ENTITY EXTRACTION:
- Extract: order numbers (format: #12345), product names, dates, amounts

RESPONSE TEMPLATES:
- technical_support_immediate
- technical_support_standard
- billing_inquiry
- product_information
- complaint_resolution
- general_response

[ADD YOUR FEW-SHOT EXAMPLES HERE]

Email: {email_text}

Output (JSON):"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # Lower temperature for more consistency
        max_tokens=300
    )
    
    return response.choices[0].message.content

print("\nITERATION 3 RESULTS:")
print("=" * 80)
for i, email in enumerate(test_emails, 1):
    result = iteration3_classifier(email)
    print(f"\n{i}. Email: {email}")
    print(f"   Result: {result}")
```

---

## Part 5: Final Optimization (15 min)

### Task 5.1: System Message + Parameter Tuning

Use system messages and optimize parameters for production.

```python
def final_classifier(email_text):
    """
    TODO: Create your final, optimized version.
    
    Use:
    - System message for role/behavior
    - User message for task/input
    - Optimized temperature
    - Appropriate max_tokens
    """
    
    system_message = """[YOUR SYSTEM MESSAGE - Define role and behavior]"""
    
    user_message = f"""[YOUR USER MESSAGE - Task and examples]

Email to classify:
{email_text}"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,  # Tune this
        max_tokens=300,   # Tune this
        top_p=0.95       # Tune this if needed
    )
    
    return response.choices[0].message.content

print("\nFINAL OPTIMIZED RESULTS:")
print("=" * 80)
for i, email in enumerate(test_emails, 1):
    result = final_classifier(email)
    print(f"\n{i}. Email: {email}")
    print(f"   Result: {result}")
```

### Task 5.2: Test on Edge Cases

Create challenging test cases and verify your prompt handles them:

```python
edge_case_emails = [
    # TODO: Add 5 challenging test cases:
    # - Ambiguous category
    # - Multiple issues in one email
    # - Sarcasm or unclear tone
    # - Missing information
    # - Very short/long email
]

print("\nEDGE CASE TESTING:")
print("=" * 80)
for email in edge_case_emails:
    result = final_classifier(email)
    print(f"\nEmail: {email}")
    print(f"Result: {result}")
    print("-" * 80)
```

---

## Part 6: Measurement and Analysis (10 min)

### Task 6.1: Create Performance Metrics

Build a simple evaluation framework.

```python
import json

def evaluate_classifier(classifier_func, test_cases):
    """
    Evaluate classifier performance.
    
    test_cases: List of (email, expected_category, expected_urgency) tuples
    """
    
    results = {
        "correct_category": 0,
        "correct_urgency": 0,
        "valid_json": 0,
        "has_all_fields": 0,
        "total": len(test_cases)
    }
    
    for email, expected_cat, expected_urg in test_cases:
        output = classifier_func(email)
        
        # TODO: Implement evaluation logic
        # 1. Try to parse JSON
        # 2. Check if all required fields present
        # 3. Compare category and urgency with expected values
        # 4. Update results dictionary
        
        pass
    
    # Calculate percentages
    for key in results:
        if key != "total":
            results[f"{key}_pct"] = (results[key] / results["total"]) * 100
    
    return results

# TODO: Create test cases with expected outputs
test_cases = [
    ("My order #12345 is damaged", "Technical", "High"),
    # Add more test cases
]

# Evaluate each iteration
baseline_score = evaluate_classifier(baseline_classifier, test_cases)
iteration1_score = evaluate_classifier(iteration1_classifier, test_cases)
iteration2_score = evaluate_classifier(iteration2_classifier, test_cases)
iteration3_score = evaluate_classifier(iteration3_classifier, test_cases)
final_score = evaluate_classifier(final_classifier, test_cases)

# Print comparison
print("\nPERFORMANCE COMPARISON:")
print("=" * 80)
print(f"Metric          | Baseline | Iter 1 | Iter 2 | Iter 3 | Final")
print("-" * 80)
# TODO: Print metrics table
```

---

## Reflection Questions

After completing all iterations, answer these questions:

### 1. What had the biggest impact on performance?
```
Your answer:
```

### 2. Which iteration showed the most improvement?
```
Your answer:
```

### 3. What would you do differently if starting over?
```
Your answer:
```

### 4. What surprised you during the optimization process?
```
Your answer:
```

### 5. How would you further improve this prompt?
```
Your answer:
```

### 6. What parameters (temperature, max_tokens) worked best and why?
```
Your answer:
```

---

## Bonus Challenges

### Challenge 1: Multi-Language Support
Modify your prompt to handle emails in multiple languages.

### Challenge 2: Confidence Scores
Add confidence levels to the classification output.

### Challenge 3: Automated Routing
Generate specific routing instructions (which team/person should handle this).

### Challenge 4: Response Draft
Generate a draft response email, not just a template name.

---

## Submission Checklist

- [ ] Completed all 5 iterations with working code
- [ ] Tested on provided test emails
- [ ] Added your own edge case tests
- [ ] Implemented evaluation framework
- [ ] Documented improvements at each iteration
- [ ] Answered all reflection questions
- [ ] Final prompt achieves >80% accuracy on test cases

---

## Tips for Success

1. **Be systematic** - Change one thing at a time to isolate impact
2. **Measure everything** - Track metrics at each iteration
3. **Test diverse inputs** - Don't just test "happy path" cases
4. **Document insights** - Write down what works and why
5. **Iterate rapidly** - Don't overthink, just try it and measure
6. **Learn from failures** - Failed attempts teach you as much as successes

---

## Resources

- [Solution Guide](../solutions/exercise-01-prompt-optimization-solution.md)
- [Prompt Engineering Best Practices](../resources/prompt-cheatsheet.md)
- [Example Prompts Library](../resources/example-prompts.md)
