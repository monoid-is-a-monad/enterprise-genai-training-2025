"""
Exercise 3: Reasoning Task with Chain-of-Thought Prompting

Time: 60 minutes
Difficulty: Intermediate-Advanced
Focus: Implementing Chain-of-Thought reasoning for complex problem-solving

OBJECTIVES:
1. Implement Chain-of-Thought prompting for different problem types
2. Compare CoT vs direct prompting performance
3. Build a reasoning system with verification
4. Handle multi-step reasoning tasks
5. Apply CoT to real-world business problems

SETUP:
- Ensure your .env file has OPENAI_API_KEY set
- Install required packages: openai, python-dotenv
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# PART 1: Basic Chain-of-Thought Implementation (15 min)
# ============================================================================

class ChainOfThoughtReasoner:
    """
    A reasoning system using Chain-of-Thought prompting.
    
    TODO: Complete this class to:
    1. Generate step-by-step reasoning for problems
    2. Extract final answers from reasoning chains
    3. Compare CoT vs direct prompting
    4. Track reasoning quality
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the reasoner.
        
        TODO: Set up client and metrics tracking
        """
        self.model = model
        self.client = client
        
        # TODO: Initialize metrics
        self.metrics = {
            "cot_requests": 0,
            "direct_requests": 0,
            "cot_correct": 0,
            "direct_correct": 0
        }
    
    def _build_cot_prompt(self, problem: str, examples: List[Tuple[str, str]] = None) -> str:
        """
        Build a Chain-of-Thought prompt.
        
        Args:
            problem: The problem to solve
            examples: Optional list of (problem, cot_solution) example pairs
        
        Returns:
            Formatted prompt with CoT examples
        
        TODO: Create prompt with:
        1. Task description
        2. Few-shot examples showing reasoning steps
        3. "Let's think step by step" trigger phrase
        4. Clear answer format
        """
        pass
    
    def solve_with_cot(self, problem: str, examples: List[Tuple[str, str]] = None) -> Dict[str, str]:
        """
        Solve a problem using Chain-of-Thought reasoning.
        
        Returns:
            Dictionary with:
            - reasoning: The step-by-step reasoning
            - answer: The extracted final answer
            - steps: List of individual reasoning steps
        
        TODO: Implement CoT solving with:
        1. Build CoT prompt
        2. Get LLM response
        3. Parse reasoning and extract answer
        4. Update metrics
        """
        pass
    
    def solve_direct(self, problem: str) -> str:
        """
        Solve a problem with direct prompting (no CoT).
        
        TODO: Implement direct solving for comparison:
        1. Create simple prompt asking for answer
        2. Get response
        3. Extract answer
        4. Update metrics
        """
        pass
    
    def compare_methods(self, problem: str, correct_answer: str) -> Dict[str, any]:
        """
        Compare CoT vs direct prompting on a problem.
        
        TODO: Implement comparison:
        1. Solve with both methods
        2. Check correctness of each
        3. Compare token usage
        4. Return comparison results
        """
        pass
    
    def _extract_answer(self, text: str) -> str:
        """
        Extract the final answer from reasoning text.
        
        TODO: Parse text to find final answer
        Common patterns:
        - "Therefore, the answer is X"
        - "Answer: X"
        - Last sentence
        """
        pass


# ============================================================================
# PART 2: Math Word Problems (15 min)
# ============================================================================

def math_problem_examples() -> List[Tuple[str, str]]:
    """
    Provide examples for math word problems.
    
    TODO: Create 3-5 high-quality examples showing:
    1. Problem statement
    2. Step-by-step reasoning
    3. Calculations
    4. Final answer
    
    Format: List of (problem, cot_solution) tuples
    """
    
    examples = [
        # Example 1
        (
            "Sarah has 3 boxes of cookies. Each box contains 12 cookies. She gives 7 cookies to her friend. How many cookies does Sarah have left?",
            """Let's solve this step by step:
1. First, find the total number of cookies: 3 boxes × 12 cookies per box = 36 cookies
2. Then subtract the cookies she gave away: 36 - 7 = 29 cookies
3. Therefore, Sarah has 29 cookies left.

Answer: 29 cookies"""
        ),
        
        # TODO: Add 2-4 more examples with varying complexity
        
    ]
    
    return examples


def test_math_reasoning():
    """
    Test Chain-of-Thought reasoning on math problems.
    
    TODO: Create test problems and evaluate performance
    """
    
    reasoner = ChainOfThoughtReasoner()
    
    # TODO: Create test problems with known answers
    test_problems = [
        # ("problem text", "expected answer"),
    ]
    
    # TODO: Test each problem with both methods
    # TODO: Calculate accuracy for each method
    # TODO: Print results and comparison
    
    pass


# ============================================================================
# PART 3: Logic Puzzles (15 min)
# ============================================================================

def logic_puzzle_examples() -> List[Tuple[str, str]]:
    """
    Provide examples for logic puzzles.
    
    TODO: Create examples showing logical reasoning:
    1. State the given information
    2. Apply logical deductions step by step
    3. Eliminate impossible options
    4. Arrive at conclusion
    """
    
    examples = [
        (
            """Three people - Alice, Bob, and Carol - each own a different pet: a cat, a dog, and a bird.
- Alice doesn't own a dog
- Carol is allergic to cats
Who owns which pet?""",
            """Let's work through this logically:

1. Start with what we know:
   - Alice doesn't own a dog (given)
   - Carol is allergic to cats, so Carol doesn't own a cat (implied)

2. From the constraints:
   - Alice: cat or bird (not dog)
   - Carol: dog or bird (not cat)
   - Bob: any of the three

3. If Carol owns the bird:
   - Then Alice must own the cat (since Alice can't own the dog)
   - That leaves Bob with the dog
   - This works! ✓

4. If Carol owns the dog:
   - Then Alice owns cat or bird
   - But we can't determine uniquely... need to check further
   - Actually, if Carol has dog and Alice can't have dog, Alice has cat or bird
   - If Alice has cat, Bob has bird
   - If Alice has bird, Bob has cat
   - Both work, so we need more info... but typically puzzles have unique solutions

Most likely solution based on typical puzzle construction:
- Alice: cat
- Bob: dog
- Carol: bird

Answer: Alice owns the cat, Bob owns the dog, Carol owns the bird."""
        ),
        
        # TODO: Add more logic puzzle examples
    ]
    
    return examples


class LogicPuzzleSolver:
    """
    Specialized solver for logic puzzles using CoT.
    
    TODO: Implement logic puzzle solving with:
    1. Constraint tracking
    2. Systematic deduction
    3. Backtracking if needed
    4. Solution verification
    """
    
    def __init__(self):
        """TODO: Initialize solver"""
        self.reasoner = ChainOfThoughtReasoner()
    
    def solve(self, puzzle: str) -> Dict[str, any]:
        """
        Solve a logic puzzle.
        
        TODO: Implement solving with:
        1. Parse puzzle constraints
        2. Apply CoT reasoning
        3. Verify solution consistency
        4. Return solution and reasoning
        """
        pass
    
    def verify_solution(self, puzzle: str, solution: Dict[str, str]) -> bool:
        """
        Verify that a solution satisfies all puzzle constraints.
        
        TODO: Check solution against original constraints
        """
        pass


# ============================================================================
# PART 4: Business Problem Solving (15 min)
# ============================================================================

def business_problem_examples() -> List[Tuple[str, str]]:
    """
    Provide examples for business problem solving.
    
    TODO: Create examples for:
    1. Financial calculations
    2. Decision analysis
    3. Resource allocation
    4. ROI analysis
    """
    pass


class BusinessProblemSolver:
    """
    Solver for business and analytical problems.
    
    TODO: Implement solver for business problems:
    1. Financial analysis
    2. Decision trees
    3. Risk assessment
    4. Cost-benefit analysis
    """
    
    def __init__(self):
        """TODO: Initialize business solver"""
        self.reasoner = ChainOfThoughtReasoner()
    
    def analyze_financial(self, problem: str) -> Dict[str, any]:
        """
        Analyze a financial problem.
        
        TODO: Handle problems involving:
        - Revenue/profit calculations
        - Break-even analysis
        - ROI calculations
        - Growth projections
        """
        pass
    
    def make_decision(self, problem: str, options: List[str]) -> Dict[str, any]:
        """
        Make a decision between multiple options.
        
        TODO: Implement decision analysis:
        1. List pros/cons for each option
        2. Weight factors
        3. Calculate scores
        4. Recommend best option with reasoning
        """
        pass
    
    def optimize_allocation(self, problem: str) -> Dict[str, any]:
        """
        Solve a resource allocation problem.
        
        TODO: Handle problems like:
        - Budget allocation
        - Staff scheduling
        - Inventory distribution
        """
        pass


# ============================================================================
# PART 5: Self-Consistency and Verification (Bonus)
# ============================================================================

class SelfConsistentReasoner(ChainOfThoughtReasoner):
    """
    Enhanced reasoner using self-consistency.
    
    TODO: Implement self-consistency:
    1. Generate multiple reasoning paths
    2. Extract answers from each path
    3. Use majority voting
    4. Return most consistent answer
    """
    
    def solve_with_self_consistency(self, problem: str, n_samples: int = 5) -> Dict[str, any]:
        """
        Solve using self-consistency (multiple reasoning paths).
        
        Args:
            problem: Problem to solve
            n_samples: Number of reasoning paths to generate
        
        Returns:
            Dictionary with:
            - final_answer: Most common answer
            - confidence: Agreement percentage
            - all_answers: All generated answers
            - reasoning_paths: All reasoning chains
        
        TODO: Implement self-consistency:
        1. Generate n different reasoning paths (use temperature=0.7)
        2. Extract answer from each
        3. Find most common answer
        4. Calculate confidence based on agreement
        """
        pass
    
    def verify_reasoning(self, problem: str, reasoning: str, answer: str) -> Dict[str, any]:
        """
        Verify the correctness of reasoning.
        
        TODO: Implement verification:
        1. Ask model to check the reasoning
        2. Identify any logical errors
        3. Verify calculations
        4. Return verification result
        """
        pass


# ============================================================================
# PART 6: Performance Analysis
# ============================================================================

def benchmark_cot_performance():
    """
    Benchmark Chain-of-Thought vs direct prompting.
    
    TODO: Implement comprehensive benchmarking:
    1. Create test suite of problems (math, logic, business)
    2. Solve each with CoT and direct methods
    3. Measure:
       - Accuracy
       - Token usage
       - Response time
       - Success rate on complex problems
    4. Generate comparison report
    """
    pass


def analyze_reasoning_quality(reasoning: str) -> Dict[str, any]:
    """
    Analyze the quality of reasoning.
    
    TODO: Implement quality metrics:
    1. Number of steps
    2. Logical flow (do steps build on each other?)
    3. Calculation accuracy
    4. Completeness (all aspects addressed?)
    5. Clarity of explanation
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
    print("EXERCISE 3: CHAIN-OF-THOUGHT REASONING")
    print("=" * 70)
    
    # Test Part 1: Basic CoT
    # print("\n--- Part 1: Basic Chain-of-Thought ---")
    # reasoner = ChainOfThoughtReasoner()
    # 
    # problem = "A store sells notebooks for $3 each. If you buy 5 notebooks, you get 20% off. How much do you pay for 5 notebooks?"
    # correct_answer = "$12"
    # 
    # comparison = reasoner.compare_methods(problem, correct_answer)
    # print(f"Problem: {problem}")
    # print(f"\nCoT Result: {comparison['cot_result']}")
    # print(f"Direct Result: {comparison['direct_result']}")
    # print(f"\nCoT Correct: {comparison['cot_correct']}")
    # print(f"Direct Correct: {comparison['direct_correct']}")
    
    # Test Part 2: Math Problems
    # print("\n--- Part 2: Math Word Problems ---")
    # test_math_reasoning()
    
    # Test Part 3: Logic Puzzles
    # print("\n--- Part 3: Logic Puzzles ---")
    # logic_solver = LogicPuzzleSolver()
    # puzzle = "Your logic puzzle here"
    # solution = logic_solver.solve(puzzle)
    # print(f"Solution: {solution}")
    
    # Test Part 4: Business Problems
    # print("\n--- Part 4: Business Problem Solving ---")
    # business_solver = BusinessProblemSolver()
    # problem = "Your business problem here"
    # analysis = business_solver.analyze_financial(problem)
    # print(f"Analysis: {analysis}")
    
    # Test Part 5: Self-Consistency (Bonus)
    # print("\n--- Part 5: Self-Consistency ---")
    # sc_reasoner = SelfConsistentReasoner()
    # problem = "Your problem here"
    # result = sc_reasoner.solve_with_self_consistency(problem, n_samples=5)
    # print(f"Final Answer: {result['final_answer']}")
    # print(f"Confidence: {result['confidence']*100:.0f}%")
    
    # Test Part 6: Benchmarking
    # print("\n--- Part 6: Performance Benchmarking ---")
    # benchmark_cot_performance()
    
    print("\n" + "=" * 70)
    print("Complete all TODOs and uncomment test sections to validate!")
    print("=" * 70)


# ============================================================================
# REFLECTION QUESTIONS
# ============================================================================

"""
After completing this exercise, answer these questions:

1. How much did CoT improve accuracy compared to direct prompting?
   Your answer:

2. What types of problems benefit most from Chain-of-Thought?
   Your answer:

3. What are the trade-offs of using CoT (cost, latency, accuracy)?
   Your answer:

4. When would you NOT want to use Chain-of-Thought prompting?
   Your answer:

5. How does self-consistency improve reliability? What's the cost?
   Your answer:

6. How would you apply CoT to your specific use case or domain?
   Your answer:

7. What did you learn about how LLMs reason through complex problems?
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
