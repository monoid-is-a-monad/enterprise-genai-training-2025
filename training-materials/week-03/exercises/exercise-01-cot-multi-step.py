"""
Week 3 - Exercise 1: Chain-of-Thought Prompting for Multi-Step Problems

Learning Objectives:
- Apply chain-of-thought prompting to complex reasoning tasks
- Implement self-consistency for improved accuracy
- Use prompt chaining to break down multi-step problems
- Evaluate reasoning quality programmatically

Scenario:
You're building a financial planning assistant that helps users make complex
decisions. The assistant needs to analyze multiple factors, show its reasoning,
and arrive at well-justified recommendations.

Time: 90 minutes
"""

import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re

# TODO: Initialize your OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Part 1: Basic Chain-of-Thought Implementation (25 minutes)
# ============================================================================

def solve_with_cot(problem: str, model: str = "gpt-4") -> Dict[str, Any]:
    """
    TODO: Implement a basic CoT prompt that:
    1. Includes "Let's think step by step" trigger
    2. Gets the model to show its reasoning
    3. Extracts the final answer from the reasoning
    4. Returns both reasoning and answer
    
    Test with this problem:
    "Sarah has $50,000 in savings. She wants to invest 60% in stocks and the 
    rest in bonds. If stocks average 8% annual return and bonds 3%, how much 
    will she have after 5 years with annual compounding?"
    
    Expected structure:
    {
        "reasoning": "Step-by-step thought process...",
        "answer": "Final numerical answer",
        "confidence": "Model's stated confidence if available"
    }
    """
    # TODO: Implement this function
    pass


def test_cot_basic():
    """Test basic CoT implementation"""
    problem = """
    Sarah has $50,000 in savings. She wants to invest 60% in stocks and the 
    rest in bonds. If stocks average 8% annual return and bonds 3%, how much 
    will she have after 5 years with annual compounding?
    """
    
    result = solve_with_cot(problem)
    
    print("Problem:", problem.strip())
    print("\nReasoning:")
    print(result["reasoning"])
    print("\nFinal Answer:", result["answer"])
    
    # TODO: Verify your answer is close to: ~$66,641
    # ($30k stocks → ~$44,080, $20k bonds → ~$23,185)


# ============================================================================
# Part 2: Self-Consistency Implementation (30 minutes)
# ============================================================================

@dataclass
class ReasoningPath:
    """Represents one reasoning path"""
    reasoning: str
    answer: str
    extracted_value: Optional[float] = None


class SelfConsistencyChecker:
    """
    TODO: Implement self-consistency by:
    1. Generating multiple reasoning paths (n=5)
    2. Extracting numerical answers from each
    3. Using majority voting to select most consistent answer
    4. Calculating confidence based on agreement
    """
    
    def __init__(self, client: OpenAI, n_paths: int = 5):
        """
        Args:
            client: OpenAI client instance
            n_paths: Number of reasoning paths to generate
        """
        # TODO: Initialize instance variables
        pass
    
    def generate_paths(self, problem: str, model: str = "gpt-4") -> List[ReasoningPath]:
        """
        TODO: Generate n_paths different reasoning paths
        
        Hints:
        - Use temperature > 0.7 for diversity
        - Each path should use CoT prompting
        - Parse out the reasoning and answer from each
        
        Returns:
            List of ReasoningPath objects
        """
        # TODO: Implement this method
        pass
    
    def extract_numerical_answer(self, answer_text: str) -> Optional[float]:
        """
        TODO: Extract numerical value from answer text
        
        Examples:
        - "The answer is $66,641.23" -> 66641.23
        - "She will have approximately 66,641 dollars" -> 66641.0
        - "Total: $66,641" -> 66641.0
        
        Hints:
        - Use regex to find numbers
        - Handle currency symbols, commas
        - Return None if no clear number found
        """
        # TODO: Implement this method
        pass
    
    def majority_vote(self, paths: List[ReasoningPath]) -> Dict[str, Any]:
        """
        TODO: Determine most consistent answer via majority voting
        
        Should return:
        {
            "selected_answer": "Most common answer",
            "confidence": 0.8,  # Fraction that agreed
            "all_paths": [...],  # All reasoning paths
            "vote_distribution": {"66641": 3, "66640": 1, "66642": 1}
        }
        
        Hints:
        - Group answers that are within 1% of each other
        - Calculate confidence as: (votes for winner) / (total paths)
        """
        # TODO: Implement this method
        pass
    
    def solve_with_consistency(self, problem: str) -> Dict[str, Any]:
        """
        TODO: Complete pipeline:
        1. Generate multiple reasoning paths
        2. Extract numerical answers
        3. Apply majority voting
        4. Return result with confidence
        """
        # TODO: Implement this method
        pass


def test_self_consistency():
    """Test self-consistency implementation"""
    # TODO: Initialize your checker
    # checker = SelfConsistencyChecker(client, n_paths=5)
    
    problem = """
    Tom wants to retire in 20 years with $1 million. He currently has $200,000
    invested. What annual return rate does he need to achieve this goal?
    Assume no additional contributions.
    """
    
    # TODO: Uncomment and test
    # result = checker.solve_with_consistency(problem)
    # print("Problem:", problem.strip())
    # print(f"\nSelected Answer: {result['selected_answer']}")
    # print(f"Confidence: {result['confidence']:.0%}")
    # print(f"\nVote Distribution: {result['vote_distribution']}")
    
    # Expected answer: ~8.4% annual return
    # Formula: (1,000,000 / 200,000)^(1/20) - 1


# ============================================================================
# Part 3: Prompt Chaining for Complex Problems (25 minutes)
# ============================================================================

class FinancialAdvisorChain:
    """
    TODO: Implement a prompt chain that:
    1. Analyzes the user's financial situation
    2. Identifies goals and constraints
    3. Generates investment recommendations
    4. Calculates expected outcomes
    5. Produces final advice
    
    Each step should pass context to the next step
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.chain_history = []
    
    def step1_analyze_situation(self, user_input: str) -> Dict[str, Any]:
        """
        TODO: Extract structured information from user's description:
        - Current savings
        - Income
        - Time horizon
        - Risk tolerance
        - Financial goals
        
        Return structured data for next step
        """
        # TODO: Implement this step
        pass
    
    def step2_identify_constraints(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Based on situation, identify:
        - Budget constraints
        - Time constraints
        - Risk constraints
        - Tax implications
        
        Return constraints for next step
        """
        # TODO: Implement this step
        pass
    
    def step3_generate_recommendations(
        self, 
        situation: Dict[str, Any], 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TODO: Generate 2-3 investment strategy recommendations
        
        Each recommendation should include:
        - Asset allocation
        - Expected return
        - Risk level
        - Reasoning
        """
        # TODO: Implement this step
        pass
    
    def step4_calculate_outcomes(
        self,
        recommendations: Dict[str, Any],
        situation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TODO: For each recommendation, calculate:
        - Projected value over time horizon
        - Best case / worst case scenarios
        - Probability of reaching goal
        
        Use CoT for calculations
        """
        # TODO: Implement this step
        pass
    
    def step5_final_advice(
        self,
        situation: Dict[str, Any],
        constraints: Dict[str, Any],
        recommendations: Dict[str, Any],
        outcomes: Dict[str, Any]
    ) -> str:
        """
        TODO: Synthesize all information into clear, actionable advice
        
        Should include:
        - Recommended strategy with reasoning
        - Action steps
        - Risk warnings
        - Monitoring plan
        """
        # TODO: Implement this step
        pass
    
    def run_chain(self, user_input: str) -> Dict[str, Any]:
        """
        TODO: Execute the full chain and return results
        
        Should track each step in chain_history for debugging
        """
        # TODO: Implement this method
        pass


def test_prompt_chaining():
    """Test prompt chaining implementation"""
    # TODO: Initialize your chain
    # advisor = FinancialAdvisorChain(client)
    
    user_input = """
    I'm 35 years old and want to retire at 55. I have $150,000 saved and 
    can invest $2,000 per month. I need $2 million by retirement. I'm 
    comfortable with moderate risk. What should I do?
    """
    
    # TODO: Uncomment and test
    # result = advisor.run_chain(user_input)
    # print("User Input:", user_input.strip())
    # print("\n" + "="*80)
    # print("FINAL ADVICE:")
    # print("="*80)
    # print(result['final_advice'])


# ============================================================================
# Part 4: Reasoning Quality Evaluation (10 minutes)
# ============================================================================

class ReasoningEvaluator:
    """
    TODO: Build an automatic evaluator that scores reasoning quality
    
    Evaluate on these dimensions:
    1. Logical coherence (do steps follow logically?)
    2. Completeness (are all aspects addressed?)
    3. Mathematical accuracy (are calculations correct?)
    4. Clarity (is reasoning easy to follow?)
    
    Use GPT-4 to evaluate GPT-3.5's reasoning
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def evaluate_reasoning(
        self, 
        problem: str, 
        reasoning: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        TODO: Get GPT-4 to evaluate the reasoning
        
        Return scores (0-10) for each dimension plus overall feedback
        """
        # TODO: Implement this method
        pass


# ============================================================================
# Reflection Questions
# ============================================================================

"""
After completing the exercises, reflect on these questions:

1. EFFECTIVENESS:
   - When did CoT prompting significantly improve accuracy?
   - When did self-consistency help most?
   - What types of problems benefit most from prompt chaining?

2. COST CONSIDERATIONS:
   - How much more expensive is self-consistency vs basic CoT?
   - Is the accuracy improvement worth the extra cost?
   - How would you decide when to use which approach in production?

3. CHAIN DESIGN:
   - How did you decide where to break the chain into steps?
   - What information needed to pass between steps?
   - How did you handle errors in intermediate steps?

4. EVALUATION:
   - Was automatic reasoning evaluation reliable?
   - What dimensions were hardest to evaluate?
   - How would you validate evaluation quality?

5. PRODUCTION CONSIDERATIONS:
   - How would you monitor CoT quality in production?
   - What failure modes should you watch for?
   - How would you A/B test CoT vs direct prompting?

Write your reflections in a separate markdown file: exercise-01-reflections.md
"""


# ============================================================================
# Optional Challenge: Advanced Self-Consistency
# ============================================================================

"""
CHALLENGE: Implement weighted voting where more confident/detailed reasoning
paths get higher weight in the final vote.

Hints:
- Score each path based on length, specificity, confidence language
- Weight votes accordingly
- Compare results to simple majority voting
"""


if __name__ == "__main__":
    print("Week 3 - Exercise 1: Chain-of-Thought Prompting")
    print("="*80)
    
    # Uncomment to run tests as you complete each part
    # test_cot_basic()
    # test_self_consistency()
    # test_prompt_chaining()
