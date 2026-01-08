# Lab 3: Chain-of-Thought Implementation - Solutions

**Duration:** 120 minutes  
**Difficulty:** Intermediate-Advanced

---

## Overview

This solution guide demonstrates how to implement Chain-of-Thought (CoT) prompting for complex reasoning tasks. Chain-of-Thought prompting encourages the model to show its reasoning steps, significantly improving performance on problems requiring multi-step logic.

---

## Part 1: Basic Chain-of-Thought

### Exercise 1.1: Math Word Problems

**Task:** Solve math word problems using CoT reasoning.

**Solution:**

```python
from openai import OpenAI

client = OpenAI()

def chain_of_thought_math(problem):
    """Solve math word problems with step-by-step reasoning."""
    
    prompt = """Solve the following math word problems by showing your reasoning step by step.

Example 1:
Problem: Sarah has 3 bags of apples. Each bag contains 7 apples. She gives 5 apples to her friend. How many apples does she have left?

Solution:
Let's think step by step:
1. First, find the total number of apples: 3 bags × 7 apples per bag = 21 apples
2. Then subtract the apples she gave away: 21 - 5 = 16 apples
3. Therefore, Sarah has 16 apples left.

Answer: 16 apples

---

Example 2:
Problem: A store sells notebooks for $3 each. If you buy 4 notebooks, you get a 20% discount. How much does it cost to buy 4 notebooks?

Solution:
Let's think step by step:
1. Calculate the original price: 4 notebooks × $3 = $12
2. Calculate the discount amount: 20% of $12 = 0.20 × $12 = $2.40
3. Subtract the discount: $12 - $2.40 = $9.60
4. Therefore, buying 4 notebooks costs $9.60

Answer: $9.60

---

Example 3:
Problem: A train travels at 60 miles per hour. How far will it travel in 2.5 hours?

Solution:
Let's think step by step:
1. Use the formula: Distance = Speed × Time
2. Substitute the values: Distance = 60 mph × 2.5 hours
3. Calculate: 60 × 2.5 = 150 miles
4. Therefore, the train will travel 150 miles.

Answer: 150 miles

---

Now solve this problem:

Problem: {problem}

Solution:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(problem=problem)}],
        temperature=0.1,  # Low temperature for logical reasoning
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()

# Test problems
test_problems = [
    "A bakery sells cupcakes for $2.50 each. If Tom buys 8 cupcakes and pays with a $25 bill, how much change will he receive?",
    
    "Emily reads 15 pages of a book each day. If the book has 240 pages, how many days will it take her to finish the book?",
    
    "A rectangular garden is 12 meters long and 8 meters wide. What is the area of the garden? If fencing costs $15 per meter, how much will it cost to fence the entire perimeter?"
]

print("Chain-of-Thought Math Problem Solving:")
print("=" * 80)

for i, problem in enumerate(test_problems, 1):
    solution = chain_of_thought_math(problem)
    print(f"\nProblem {i}:")
    print(f"{problem}")
    print(f"\n{solution}")
    print("-" * 80)
```

**Expected Output for Problem 1:**
```
Solution:
Let's think step by step:
1. Calculate the total cost: 8 cupcakes × $2.50 = $20.00
2. Calculate the change: $25.00 - $20.00 = $5.00
3. Therefore, Tom will receive $5.00 in change.

Answer: $5.00
```

**Why This Works:**
- Examples show explicit reasoning steps
- Clear format: "Let's think step by step:"
- Each step builds on previous ones
- Final answer clearly marked
- Low temperature ensures logical consistency

---

### Exercise 1.2: Logic Puzzles

**Task:** Solve logic puzzles using CoT reasoning.

**Solution:**

```python
def chain_of_thought_logic(puzzle):
    """Solve logic puzzles with detailed reasoning."""
    
    prompt = """Solve the following logic puzzles by reasoning through them step by step.

Example:
Puzzle: Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a bird. 
- Alice doesn't have a dog
- Carol is allergic to cats
- Who has which pet?

Solution:
Let's work through this logically:
1. Start with what we know:
   - Alice doesn't have a dog (given)
   - Carol is allergic to cats, so Carol doesn't have a cat (implied)

2. From clue 2: Carol must have either a dog or a bird. Since Carol doesn't have a cat.

3. If Carol has a dog:
   - Then Alice has either a cat or a bird
   - Since Alice doesn't have a dog (clue 1), and Carol has the dog, Alice could have cat or bird
   - That would leave Bob with the remaining pet

4. Let's test: If Carol has the dog:
   - Alice doesn't have the dog (clue 1), so Alice has cat or bird
   - This means Bob has the remaining pet
   - This works!

5. But let's also test: If Carol has the bird:
   - Alice doesn't have the dog (clue 1)
   - Carol doesn't have the cat (clue 2)
   - So Alice has the cat and Bob has the dog
   - This also works!

6. We need more information... Wait, let me reconsider:
   - Carol is allergic to cats, so Carol definitely doesn't have a cat
   - Carol must have either dog or bird
   - Alice doesn't have a dog, so Alice has cat or bird
   - If Carol has bird, Alice must have cat (since Alice can't have dog), leaving Bob with dog ✓
   - If Carol has dog, Alice has cat or bird, but then we can't determine uniquely...

Actually, the most constrained solution:
- Carol: dog or bird (not cat due to allergy)
- Alice: cat or bird (not dog)
- If Carol has dog → Alice has bird or cat → Bob gets the other
- If Carol has bird → Alice must have cat (can't have dog) → Bob has dog

Without additional constraints, we have two valid solutions. However, typically such puzzles have unique solutions, so there may be an implicit constraint.

Most likely answer based on typical puzzle construction:
- Alice: cat
- Bob: dog  
- Carol: bird

Answer: Alice has the cat, Bob has the dog, Carol has the bird.

---

Now solve this puzzle:

Puzzle: {puzzle}

Solution:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(puzzle=puzzle)}],
        temperature=0.2,
        max_tokens=600
    )
    
    return response.choices[0].message.content.strip()

# Test puzzles
test_puzzles = [
    """Four people - Amy, Ben, Claire, and Dan - finished a race in different positions (1st, 2nd, 3rd, 4th).
    - Amy finished before Ben
    - Claire finished right after Amy
    - Dan did not finish last
    What position did each person finish in?""",
    
    """Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. 
    You can pick one fruit from one box. Which box should you pick from to correctly label all boxes?"""
]

print("\nChain-of-Thought Logic Puzzle Solving:")
print("=" * 80)

for i, puzzle in enumerate(test_puzzles, 1):
    solution = chain_of_thought_logic(puzzle)
    print(f"\nPuzzle {i}:")
    print(f"{puzzle}")
    print(f"\n{solution}")
    print("-" * 80)
```

---

## Part 2: Advanced Chain-of-Thought Techniques

### Exercise 2.1: Self-Consistency

**Task:** Use multiple reasoning paths and select the most consistent answer.

**Solution:**

```python
from collections import Counter

def self_consistency_cot(problem, num_samples=5):
    """
    Use self-consistency: generate multiple reasoning paths and 
    select the most common answer.
    """
    
    prompt = f"""Solve this problem step by step:

Problem: {problem}

Let's think through this carefully:"""
    
    answers = []
    reasonings = []
    
    for i in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temperature for diverse reasoning
            max_tokens=300
        )
        
        full_response = response.choices[0].message.content.strip()
        reasonings.append(full_response)
        
        # Extract the final answer (last line typically)
        lines = full_response.split('\n')
        answer_line = [line for line in lines if 'answer:' in line.lower()]
        if answer_line:
            answer = answer_line[-1].split(':')[-1].strip()
            answers.append(answer)
    
    # Find most common answer
    if answers:
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        confidence = count / num_samples
        
        return {
            "final_answer": most_common_answer,
            "confidence": confidence,
            "all_answers": answers,
            "reasoning_paths": reasonings
        }
    else:
        return {"error": "Could not extract answers"}

# Test problem
problem = """A garden has roses and tulips. There are 3 times as many tulips as roses. 
If there are 48 flowers in total, how many roses are there?"""

print("Self-Consistency Chain-of-Thought:")
print("=" * 80)
print(f"\nProblem: {problem}\n")

result = self_consistency_cot(problem, num_samples=5)

print(f"Final Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']*100:.0f}% ({result['confidence']*5:.0f}/5 agreed)")
print(f"\nAll Answers: {result['all_answers']}")
print(f"\nAnswer Distribution:")
for answer, count in Counter(result['all_answers']).most_common():
    print(f"  {answer}: {count}/{len(result['all_answers'])} times")

print("\n" + "=" * 80)
print("Sample Reasoning Paths:")
for i, reasoning in enumerate(result['reasoning_paths'][:2], 1):
    print(f"\nPath {i}:")
    print(reasoning)
    print("-" * 80)
```

**Expected Output:**
```
Final Answer: 12 roses
Confidence: 100% (5/5 agreed)

All Answers: ['12 roses', '12 roses', '12 roses', '12 roses', '12 roses']

Answer Distribution:
  12 roses: 5/5 times
```

**Why Self-Consistency Works:**
- Multiple reasoning paths catch errors
- Most common answer likely correct
- Confidence score indicates reliability
- Diverse temperature generates varied approaches
- Aggregation reduces random errors

---

### Exercise 2.2: Least-to-Most Prompting

**Task:** Break complex problems into subproblems.

**Solution:**

```python
def least_to_most_prompting(problem):
    """
    Solve complex problems by breaking them into simpler subproblems.
    """
    
    # Step 1: Decompose the problem
    decomposition_prompt = f"""Break down this problem into simpler subproblems that need to be solved in order:

Problem: {problem}

List the subproblems:"""
    
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": decomposition_prompt}],
        temperature=0.3,
        max_tokens=200
    )
    
    subproblems = response1.choices[0].message.content.strip()
    print("Subproblems Identified:")
    print(subproblems)
    print("\n" + "=" * 80 + "\n")
    
    # Step 2: Solve each subproblem sequentially
    solving_prompt = f"""Original Problem: {problem}

Subproblems to solve:
{subproblems}

Now solve each subproblem step by step, using the solution from each to help solve the next:"""
    
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": solving_prompt}],
        temperature=0.2,
        max_tokens=500
    )
    
    solution = response2.choices[0].message.content.strip()
    
    return {
        "subproblems": subproblems,
        "solution": solution
    }

# Test with complex problem
complex_problem = """A company has 120 employees. 60% work in the main office, and the rest work remotely. 
Of the remote workers, 40% are in the engineering department. The company wants to give a $500 bonus to 
each remote engineer and a $300 bonus to all other employees. What is the total bonus budget needed?"""

print("Least-to-Most Prompting:")
print("=" * 80)
print(f"\nComplex Problem:\n{complex_problem}\n")
print("=" * 80 + "\n")

result = least_to_most_prompting(complex_problem)

print("Step-by-Step Solution:")
print(result["solution"])
```

**Expected Decomposition:**
```
Subproblems Identified:
1. Calculate how many employees work in the main office
2. Calculate how many employees work remotely
3. Calculate how many remote workers are in engineering
4. Calculate how many remote workers are not in engineering
5. Calculate bonus for remote engineers
6. Calculate bonus for all other employees (main office + non-engineer remote)
7. Calculate total bonus budget
```

---

## Part 3: Domain-Specific Chain-of-Thought

### Exercise 3.1: Code Debugging with CoT

**Task:** Debug code by reasoning through the logic.

**Solution:**

```python
def cot_code_debugging(code, error_description):
    """Debug code using chain-of-thought reasoning."""
    
    prompt = f"""Debug the following code by reasoning through it step by step.

Code:
```python
{code}
```

Error/Issue: {error_description}

Debugging Process:
Let's analyze this systematically:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )
    
    return response.choices[0].message.content.strip()

# Test case
buggy_code = """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

scores = [85, 90, 78, 92, 88]
average = calculate_average(scores)
print(f"Average score: {average}")

# Now calculate average of empty list
empty_scores = []
empty_average = calculate_average(empty_scores)
print(f"Average of empty list: {empty_average}")"""

error_desc = "The code crashes with 'ZeroDivisionError: division by zero' when calculating average of empty list."

print("Code Debugging with Chain-of-Thought:")
print("=" * 80)

debug_analysis = cot_code_debugging(buggy_code, error_desc)
print(debug_analysis)
```

**Expected Analysis:**
```
Let's analyze this systematically:

1. **Identify the error location:**
   - The error occurs in the calculate_average function
   - Specifically at: `return total / len(numbers)`
   - When numbers is an empty list, len(numbers) = 0

2. **Understand why it fails:**
   - Division by zero is undefined in Python
   - The function doesn't handle the edge case of an empty list

3. **Determine the fix:**
   - Add a check for empty list before division
   - Decide what to return for empty list (None, 0, or raise informative error)

4. **Corrected code:**
```python
def calculate_average(numbers):
    if not numbers:  # Check if list is empty
        return None  # or raise ValueError("Cannot calculate average of empty list")
    
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
```

5. **Explanation:**
   - The `if not numbers:` check catches empty lists
   - Returns None (or raises descriptive error) instead of crashing
   - Function now handles both normal and edge cases gracefully
```

---

### Exercise 3.2: Data Analysis with CoT

**Task:** Analyze data and draw conclusions using step-by-step reasoning.

**Solution:**

```python
def cot_data_analysis(data_description, question):
    """Analyze data with chain-of-thought reasoning."""
    
    prompt = f"""Analyze the following data by reasoning through it step by step.

Data: {data_description}

Question: {question}

Analysis:
Let's examine this data systematically:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

# Test case
data = """E-commerce sales data for Q1 2025:
- January: 1,200 orders, $45,000 revenue
- February: 1,400 orders, $49,000 revenue  
- March: 1,800 orders, $63,000 revenue

Customer acquisition cost per month:
- January: $12,000
- February: $14,000
- March: $18,000"""

question = "Is our marketing efficiency improving or declining? Calculate the relevant metrics and explain the trend."

print("Data Analysis with Chain-of-Thought:")
print("=" * 80)

analysis = cot_data_analysis(data, question)
print(analysis)
```

**Expected Analysis:**
```
Let's examine this data systematically:

1. **Calculate Average Order Value (AOV) for each month:**
   - January: $45,000 / 1,200 = $37.50 per order
   - February: $49,000 / 1,400 = $35.00 per order
   - March: $63,000 / 1,800 = $35.00 per order
   
2. **Calculate Cost Per Acquisition (CPA):**
   - January: $12,000 / 1,200 = $10.00 per customer
   - February: $14,000 / 1,400 = $10.00 per customer
   - March: $18,000 / 1,800 = $10.00 per customer

3. **Calculate Customer Lifetime Value / CAC Ratio:**
   - January: $37.50 / $10.00 = 3.75
   - February: $35.00 / $10.00 = 3.50
   - March: $35.00 / $10.00 = 3.50

4. **Analyze trends:**
   - CPA remains stable at $10 per customer (good consistency)
   - AOV decreased from $37.50 to $35.00 (6.7% decline)
   - Revenue grew 40% (Jan to Mar), but orders grew 50%
   - This means growth is driven by volume, not value per order

5. **Conclusion:**
   Marketing efficiency is **slightly declining**:
   - We're maintaining customer acquisition costs
   - BUT customers are spending less per order
   - The LTV/CAC ratio dropped from 3.75 to 3.50
   - While still profitable (ratio > 3), the trend is concerning

**Recommendation:** Investigate why AOV is declining. Consider upselling strategies or focus on higher-value customer segments.
```

---

## Part 4: Verifying Chain-of-Thought Reasoning

### Exercise 4.1: Verification Prompting

**Task:** Ask the model to verify its own reasoning.

**Solution:**

```python
def verified_cot(problem):
    """Generate answer, then verify the reasoning."""
    
    # Step 1: Generate initial solution
    solve_prompt = f"""Solve this problem step by step:

Problem: {problem}

Solution:"""
    
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": solve_prompt}],
        temperature=0.2,
        max_tokens=300
    )
    
    initial_solution = response1.choices[0].message.content.strip()
    
    # Step 2: Verify the solution
    verify_prompt = f"""Review this solution and check if the reasoning is correct:

Problem: {problem}

Solution:
{initial_solution}

Verification:
Let's check each step carefully:"""
    
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": verify_prompt}],
        temperature=0.2,
        max_tokens=300
    )
    
    verification = response2.choices[0].message.content.strip()
    
    return {
        "initial_solution": initial_solution,
        "verification": verification
    }

# Test problem
problem = """If 5 machines can produce 5 widgets in 5 minutes, 
how many machines are needed to produce 100 widgets in 100 minutes?"""

print("Verified Chain-of-Thought:")
print("=" * 80)
print(f"\nProblem: {problem}\n")
print("=" * 80)

result = verified_cot(problem)

print("\nInitial Solution:")
print(result["initial_solution"])
print("\n" + "=" * 80)
print("\nVerification:")
print(result["verification"])
```

---

## Part 5: Best Practices and Patterns

### Key CoT Patterns

#### 1. Explicit Reasoning Trigger
```python
# Good
"Let's think step by step:"
"Let's solve this systematically:"
"Let's work through this carefully:"

# Less effective
"Solve this."
"What's the answer?"
```

#### 2. Numbered Steps
```python
prompt = """
Problem: [problem]

Solution:
1. [First step]
2. [Second step]
3. [Third step]
Therefore: [answer]
"""
```

#### 3. Show Your Work Format
```python
prompt = """
Solve this problem. Show all your work.

Problem: [problem]

Working:
[Step-by-step calculation]

Final Answer: [result]
"""
```

---

### When to Use Chain-of-Thought ✅

1. **Math word problems** - Multi-step arithmetic
2. **Logic puzzles** - Deductive reasoning required
3. **Code debugging** - Trace execution flow
4. **Data analysis** - Calculate metrics, draw conclusions
5. **Planning tasks** - Break down complex goals
6. **Comparison tasks** - Evaluate multiple options
7. **Debugging** - Systematic error analysis

### When NOT to Use CoT ❌

1. **Simple classifications** - "Is this positive or negative?"
2. **Fact retrieval** - "What is the capital of France?"
3. **Direct lookups** - "What does API stand for?"
4. **Creative generation** - May constrain creativity
5. **Very short responses** - Overhead not worth it

---

## Performance Metrics

### Accuracy Improvements (Research Findings)

| Task Type | Zero-Shot | CoT | Improvement |
|-----------|-----------|-----|-------------|
| Math Problems | ~30% | ~65% | **+35pp** |
| Logic Puzzles | ~40% | ~75% | **+35pp** |
| Commonsense Reasoning | ~50% | ~70% | **+20pp** |
| Symbolic Manipulation | ~35% | ~80% | **+45pp** |

### Cost Considerations

- **Token Usage:** CoT uses 2-3x more tokens (examples + reasoning)
- **Latency:** Longer responses = more time
- **Trade-off:** Higher accuracy worth the cost for critical tasks

---

## Key Takeaways

### Do's ✅
1. **Use trigger phrases** - "Let's think step by step"
2. **Show examples** - Include CoT examples in few-shot prompts
3. **Number steps** - Makes reasoning easy to follow
4. **Verify answers** - Use self-consistency or verification
5. **Break down complex** - Use least-to-most for hard problems
6. **Low temperature** - 0.1-0.3 for logical consistency

### Don'ts ❌
1. **Don't skip examples** - CoT needs demonstration
2. **Don't use for simple tasks** - Wastes tokens
3. **Don't ignore errors** - Verify critical calculations
4. **Don't rush** - Let model work through steps
5. **Don't mix styles** - Be consistent in format

---

## Next Steps

1. **Practice on your domain** - Apply CoT to your specific problems
2. **Build CoT templates** - Create reusable patterns
3. **Measure improvement** - Compare CoT vs direct prompting
4. **Combine techniques** - CoT + few-shot + verification
5. **Explore research** - Read latest papers on prompting methods

---

## Additional Resources

- Research: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
- Research: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
- [Prompt Engineering Guide](../resources/prompt-cheatsheet.md)
- [Advanced Prompting Techniques](../resources/references.md)
