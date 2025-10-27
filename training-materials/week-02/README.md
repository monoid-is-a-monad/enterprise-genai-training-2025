# Week 2: Prompt Engineering & LLM Basics

**Provided by:** ADC ENGINEERING & CONSULTING LTD

**Duration:** 20 hours

## Overview

This week focuses on mastering the art and science of prompt engineering. You'll learn how to effectively communicate with Large Language Models to achieve desired outcomes, understand different prompting techniques, and apply best practices for reliable AI interactions.

## Learning Objectives

By the end of this week, participants will be able to:

- [ ] Understand the fundamentals of prompt engineering and why it matters
- [ ] Apply zero-shot, one-shot, and few-shot learning techniques
- [ ] Implement chain-of-thought prompting for complex reasoning tasks
- [ ] Design effective prompts using proven patterns and templates
- [ ] Handle common prompting challenges (hallucinations, biases, constraints)
- [ ] Evaluate prompt quality and iterate for improvement
- [ ] Use system messages, user messages, and assistant messages effectively
- [ ] Apply prompt engineering best practices in real-world scenarios

## Prerequisites

- Completion of Week 1 materials
- Working OpenAI API setup
- Understanding of LLM basics and tokens
- Familiarity with Python and Jupyter notebooks

## Content Structure

### Lessons
1. **Introduction to Prompt Engineering** - Fundamentals, importance, and core concepts
2. **Zero-Shot and Few-Shot Learning** - Learning paradigms and when to use each
3. **Advanced Prompting Techniques** - Chain-of-thought, self-consistency, and reasoning
4. **Prompt Patterns & Best Practices** - Reusable templates and design principles

### Labs
1. **Basic Prompt Engineering** - Hands-on practice with different prompt styles
2. **Few-Shot Learning Experiments** - Building effective few-shot prompts
3. **Chain-of-Thought Implementation** - Complex reasoning tasks

### Exercises
1. Prompt optimization challenge
2. Few-shot classifier implementation
3. Reasoning task with CoT prompting

## Session Structure

**Day 1: Foundations**
- Lesson 1: Introduction to Prompt Engineering
- Lesson 2: Zero-Shot and Few-Shot Learning
- Lab 1: Basic Prompt Engineering

**Day 2: Advanced Techniques**
- Lesson 3: Advanced Prompting Techniques
- Lesson 4: Prompt Patterns & Best Practices
- Lab 2: Few-Shot Learning Experiments

**Day 3: Hands-On Practice**
- Lab 3: Chain-of-Thought Implementation
- Exercise practice and optimization

**Day 4: Application & Review**
- Real-world prompt engineering scenarios
- Project work: Build a prompt library
- Week review and Q&A

## Key Concepts

### Prompt Engineering Fundamentals
- What is a prompt and why structure matters
- Instructions, context, input data, and output indicators
- Temperature and other parameters
- Iterative prompt development

### Learning Paradigms
- Zero-shot learning: Tasks without examples
- One-shot learning: Single example guidance
- Few-shot learning: Multiple examples for pattern learning
- When to use each approach

### Advanced Techniques
- Chain-of-Thought (CoT): Step-by-step reasoning
- Self-consistency: Multiple reasoning paths
- Tree of Thoughts: Exploring multiple solutions
- Prompt chaining: Breaking complex tasks into steps

### Prompt Patterns
- Persona patterns: Role-playing for better responses
- Format patterns: Structured output templates
- Constraint patterns: Setting boundaries and rules
- Meta-prompts: Prompts about prompting

## Assessment Criteria

### Knowledge Check
- Understanding of prompting paradigms (zero-shot, few-shot, CoT)
- Ability to identify appropriate techniques for different tasks
- Knowledge of prompt structure and components

### Practical Skills
- Can design effective prompts for various use cases
- Demonstrates proper use of few-shot examples
- Implements chain-of-thought reasoning correctly
- Iterates and improves prompts based on results

### Lab Completion
- All three labs completed with working implementations
- Exercises demonstrate understanding of concepts
- Code follows best practices and includes error handling

## Resources

### Required Materials
- Week 2 lesson materials (in `lessons/` folder)
- Jupyter notebooks for labs (in `labs/` folder)
- Exercise templates (in `exercises/` folder)

### Additional Resources
- OpenAI Prompt Engineering Guide
- Anthropic Prompt Engineering Documentation
- Research papers on prompting techniques
- Community prompt libraries and examples

### Tools & Libraries
```python
openai>=1.0.0
tiktoken
python-dotenv
pandas
```

## Common Challenges & Solutions

### Challenge 1: Inconsistent Responses
**Problem:** Model gives different answers to the same prompt  
**Solution:** Use lower temperature, add constraints, implement self-consistency

### Challenge 2: Not Following Instructions
**Problem:** Model ignores parts of the prompt  
**Solution:** Restructure prompt, use clearer language, add examples, repeat key instructions

### Challenge 3: Verbose Outputs
**Problem:** Responses are too long or off-topic  
**Solution:** Set explicit length constraints, use format patterns, specify output structure

### Challenge 4: Poor Few-Shot Performance
**Problem:** Examples don't improve results  
**Solution:** Ensure example diversity, check example quality, balance example set, verify format consistency

## Deliverables

By the end of Week 2, participants should have:

1. **Prompt Library** - Collection of reusable prompt templates
2. **Few-Shot Classifier** - Working implementation using few-shot learning
3. **CoT Reasoning System** - Solution to complex problem using chain-of-thought
4. **Best Practices Document** - Personal guide with learnings and patterns

## Success Metrics

- [ ] Can write effective zero-shot prompts for common tasks
- [ ] Successfully implements few-shot learning with appropriate examples
- [ ] Applies chain-of-thought prompting to complex reasoning problems
- [ ] Demonstrates understanding of prompt patterns and when to use them
- [ ] Can debug and improve poor-performing prompts
- [ ] Completes all labs and exercises with working code

## Weekly Project: Prompt Engineering Toolkit

Build a Python toolkit that includes:
- Prompt template library with common patterns
- Few-shot example manager
- Chain-of-thought prompt generator
- Prompt evaluation and testing utilities

**Requirements:**
- Modular, reusable code
- Documentation and examples
- Unit tests for key functions
- README with usage instructions

---

**Week Coordinator:** Training Team  
**Last Updated:** October 27, 2025
