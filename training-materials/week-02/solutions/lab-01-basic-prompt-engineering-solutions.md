# Lab 1: Basic Prompt Engineering - Solutions

**Duration:** 120 minutes  
**Difficulty:** Beginner-Intermediate

---

## Overview

This solution guide provides reference answers and best practices for Lab 1: Basic Prompt Engineering. Use this after attempting the lab exercises to compare your approaches and learn alternative techniques.

---

## Part 1: Understanding Prompt Components

### Exercise 1.1: Decompose a Prompt

**Task:** Identify the components of the following prompt:

```
You are a professional copywriter specializing in technology products.

Write a compelling product description for a new wireless keyboard that features:
- Mechanical switches
- RGB lighting
- 3-device Bluetooth connectivity
- 40-hour battery life

The description should be 100-150 words and target tech enthusiasts.
```

**Solution:**

| Component | Content | Purpose |
|-----------|---------|---------|
| **System/Persona** | "You are a professional copywriter specializing in technology products." | Sets the role and expertise level |
| **Task Instruction** | "Write a compelling product description" | Defines the primary action |
| **Context/Input** | Features list (mechanical switches, RGB, etc.) | Provides necessary information |
| **Constraints** | "100-150 words", "target tech enthusiasts" | Sets boundaries and audience |
| **Output Format** | Product description | Specifies expected output type |

**Key Insights:**
- Clear persona improves tone consistency
- Specific constraints prevent overly long responses
- Target audience specification helps tailor language and focus
- Structured input (bullet points) makes information easy to process

---

### Exercise 1.2: Build Your Own Prompt

**Task:** Create a prompt to generate email subject lines for a marketing campaign.

**Sample Solution:**

```
You are an expert email marketer with 10 years of experience in SaaS companies.

Generate 5 compelling email subject lines for a campaign promoting a new project management tool. The tool's key benefits are:
- Real-time collaboration
- AI-powered task prioritization
- Integration with 50+ apps
- Mobile-first design

Requirements:
- Each subject line should be 6-10 words
- Create urgency without being pushy
- Include at least one benefit or value proposition
- Avoid spam trigger words (free, act now, etc.)
- Target busy project managers at tech companies

Format: Return as a numbered list.
```

**Why This Works:**
1. **Specific persona** grounds the tone and expertise
2. **Clear deliverable** (5 subject lines) prevents endless generation
3. **Constraints** (word count, style guidelines) ensure quality
4. **Context** (tool benefits) provides necessary information
5. **Format specification** makes output immediately usable

---

## Part 2: Zero-Shot Prompting

### Exercise 2.1: Simple Classification

**Task:** Create a zero-shot prompt to classify customer feedback sentiment.

**Solution:**

```python
prompt = """Classify the sentiment of the following customer feedback as Positive, Negative, or Neutral.

Customer Feedback: "{feedback}"

Sentiment:"""

# Test examples
test_feedback = [
    "The product exceeded my expectations! Fast shipping too.",
    "Item arrived damaged and customer service was unhelpful.",
    "The color is slightly different from the photo but it's okay.",
]

# Expected outputs: Positive, Negative, Neutral
```

**Advanced Solution with Reasoning:**

```python
prompt = """Analyze the sentiment of the customer feedback below and classify it as Positive, Negative, or Neutral.

Customer Feedback: "{feedback}"

Provide your answer in this format:
Sentiment: [Your classification]
Confidence: [High/Medium/Low]
Key phrases: [Words or phrases that influenced your decision]

Your response:"""
```

**Why the Advanced Version is Better:**
- Requests confidence level helps identify uncertain cases
- Key phrases provide transparency for debugging
- Structured output format ensures consistency
- Can be easily parsed programmatically

---

### Exercise 2.2: Information Extraction

**Task:** Extract key information from unstructured text.

**Solution:**

```python
prompt = """Extract the following information from the job posting below:
- Job Title
- Company Name
- Location (city, state/country, or "Remote")
- Salary Range (if mentioned, otherwise "Not specified")
- Years of Experience Required (if mentioned, otherwise "Not specified")
- Key Skills (list top 5)

Job Posting:
\"\"\"{job_posting}\"\"\"

Provide your answer as a JSON object with these exact keys:
{
  "job_title": "",
  "company": "",
  "location": "",
  "salary_range": "",
  "experience_required": "",
  "key_skills": []
}
"""

# Example job posting
job_posting = """
Senior Python Developer - TechCorp Inc.

We're seeking an experienced Python developer for our San Francisco office or remote within US.
Compensation: $140,000 - $180,000 depending on experience.

Requirements:
- 5+ years of Python development
- Strong experience with Django and FastAPI
- Cloud platforms (AWS or GCP)
- Docker and Kubernetes
- CI/CD pipelines
- SQL and NoSQL databases
"""

# Expected JSON output:
{
  "job_title": "Senior Python Developer",
  "company": "TechCorp Inc.",
  "location": "San Francisco, CA or Remote (US)",
  "salary_range": "$140,000 - $180,000",
  "experience_required": "5+ years",
  "key_skills": ["Python", "Django", "FastAPI", "AWS/GCP", "Docker/Kubernetes"]
}
```

**Best Practices Demonstrated:**
1. **Explicit field list** - No ambiguity about what to extract
2. **Handling missing data** - Specifies what to do if information isn't present
3. **Structured output** - JSON format for easy parsing
4. **Clear example format** - Shows exact structure expected

---

## Part 3: Temperature and Parameter Effects

### Exercise 3.1: Temperature Experimentation

**Task:** Test different temperature values for creative vs factual tasks.

**Solution & Analysis:**

```python
import openai
from openai import OpenAI

client = OpenAI()

def test_temperature(prompt, temperatures, iterations=3):
    """Test prompt with different temperature values."""
    results = {}
    
    for temp in temperatures:
        results[temp] = []
        print(f"\n=== Temperature: {temp} ===")
        
        for i in range(iterations):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=100
            )
            
            text = response.choices[0].message.content
            results[temp].append(text)
            print(f"Iteration {i+1}: {text[:80]}...")
    
    return results

# Test 1: Factual Task (Temperature should be LOW)
factual_prompt = "What is the capital of France?"
factual_results = test_temperature(factual_prompt, [0, 0.3, 0.7, 1.0])

# Test 2: Creative Task (Temperature can be HIGHER)
creative_prompt = "Write the opening line of a mystery novel set in Tokyo."
creative_results = test_temperature(creative_prompt, [0, 0.3, 0.7, 1.0])
```

**Expected Observations:**

| Temperature | Factual Task (Capital of France) | Creative Task (Mystery Opening) |
|-------------|----------------------------------|----------------------------------|
| **0** | Identical answer every time: "The capital of France is Paris." | Same opening line repeated |
| **0.3** | Minor wording variations, always correct | Slight variations, similar themes |
| **0.7** | May add extra context, still accurate | Good diversity, different approaches |
| **1.0** | Could add conversational elements | High creativity, very diverse outputs |

**Recommendations:**
- **Factual/Analytical:** temperature=0 to 0.3
- **General Purpose:** temperature=0.5 to 0.7  
- **Creative Writing:** temperature=0.7 to 1.0
- **Brainstorming:** temperature=0.9 to 1.2

---

### Exercise 3.2: Max Tokens Control

**Task:** Test how max_tokens affects response completeness.

**Solution:**

```python
def test_max_tokens(prompt, token_limits):
    """Test how max_tokens affects response quality."""
    
    for max_tok in token_limits:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tok,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        actual_tokens = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason
        
        print(f"\nMax Tokens: {max_tok}")
        print(f"Actual Tokens Used: {actual_tokens}")
        print(f"Finish Reason: {finish_reason}")
        print(f"Response: {content}")
        print(f"Complete: {finish_reason == 'stop'}")
        print("-" * 80)

# Test with a prompt that naturally generates long responses
prompt = "Explain the concept of object-oriented programming with examples."

test_max_tokens(prompt, [20, 50, 100, 200, 500])
```

**Analysis:**

| Max Tokens | Outcome | Finish Reason | Usability |
|------------|---------|---------------|-----------|
| **20** | Truncated mid-sentence | `length` | ❌ Unusable |
| **50** | Incomplete explanation | `length` | ⚠️ Partial |
| **100** | Basic explanation, may be cut | `length` or `stop` | ⚠️ Adequate |
| **200** | Complete explanation | `stop` | ✅ Good |
| **500** | Full explanation with examples | `stop` | ✅ Excellent |

**Best Practices:**
- Monitor `finish_reason`: `stop` = complete, `length` = truncated
- Set max_tokens 20-30% higher than expected length
- For user-facing apps, detect truncation and request completion
- Balance completeness with cost considerations

---

## Part 4: System Messages and Roles

### Exercise 4.1: Effective System Messages

**Task:** Create system messages for different personas.

**Solution Examples:**

#### 1. Technical Support Bot
```python
system_message = """You are a friendly and patient technical support specialist for CloudStore, a cloud storage service.

Your responsibilities:
- Help users troubleshoot common issues
- Explain technical concepts in simple terms
- Provide step-by-step instructions
- Escalate complex issues to human agents when appropriate

Guidelines:
- Always maintain a helpful and empathetic tone
- Ask clarifying questions before assuming the problem
- Provide links to documentation when relevant
- Never make promises about features or timelines
- If you don't know something, admit it and offer to find out

Response format:
- Keep responses concise (2-3 paragraphs max)
- Use bullet points for steps
- Include relevant links in markdown format
"""
```

#### 2. Code Reviewer
```python
system_message = """You are an experienced senior software engineer conducting a code review.

Focus areas:
- Code correctness and potential bugs
- Performance and efficiency
- Security vulnerabilities
- Code readability and maintainability
- Adherence to best practices

Review style:
- Be constructive and respectful
- Explain the "why" behind suggestions
- Prioritize issues (critical, important, nice-to-have)
- Provide specific examples for improvements
- Acknowledge good practices when you see them

Format your review as:
1. Summary (2-3 sentences)
2. Critical Issues (if any)
3. Suggestions for Improvement
4. Positive Observations
"""
```

#### 3. Data Analyst Assistant
```python
system_message = """You are an expert data analyst assistant specializing in business intelligence and data interpretation.

Your capabilities:
- Analyze data trends and patterns
- Provide statistical insights
- Suggest appropriate visualizations
- Explain technical concepts to non-technical stakeholders
- Recommend data-driven actions

Your approach:
- Start with high-level insights, then provide details
- Use clear, jargon-free language
- Include relevant metrics and percentages
- Connect findings to business impact
- Ask questions to understand the business context

Response structure:
- Key Findings (bullet points)
- Supporting Evidence (data/metrics)
- Recommendations (actionable next steps)
"""
```

**Why These Work:**
1. **Clear role definition** - No ambiguity about the AI's function
2. **Specific guidelines** - Behavior boundaries and expectations
3. **Response format** - Ensures consistent, structured outputs
4. **Tone specification** - Matches use case requirements
5. **Limitations acknowledged** - Sets realistic expectations

---

### Exercise 4.2: Multi-Turn Conversations

**Task:** Design a conversation flow with proper role management.

**Solution:**

```python
def create_conversation_history(user_inputs, system_msg):
    """Build a conversation history with proper roles."""
    
    messages = [{"role": "system", "content": system_msg}]
    
    for user_input, assistant_response in user_inputs:
        messages.append({"role": "user", "content": user_input})
        if assistant_response:
            messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# Example: Career Counselor Conversation
system_msg = """You are a professional career counselor helping someone explore career options. 
Ask thoughtful follow-up questions to understand their interests, skills, and goals. 
Provide specific, actionable advice."""

conversation = [
    ("I'm thinking about changing careers but I'm not sure what to do.", 
     "I'd be happy to help you explore career options! To better understand your situation, could you tell me: What field are you currently in, and what aspects of your current role do you find most and least fulfilling?"),
    
    ("I'm in accounting but I find it too repetitive. I enjoy the analysis part but not the routine bookkeeping.",
     "That's helpful insight! The analytical aspect of accounting could translate well to several careers. Are you interested in staying in a finance-related field, or would you be open to transitioning to something different? Also, what other skills or interests do you have outside of work?"),
    
    ("I'm pretty good with data and I've been learning Python in my spare time. I'm open to something different.",
     None)  # This is where we'll get the next AI response
]

messages = create_conversation_history(conversation, system_msg)

# Get next response
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.7
)

print(response.choices[0].message.content)
```

**Expected Response:**
```
That's a fantastic combination! Your accounting background, analytical skills, and Python knowledge 
position you well for several exciting career paths:

1. **Data Analyst/Business Analyst**: Leverage your financial analysis experience while working with 
   data more broadly. Python is highly valued here.

2. **Financial Data Scientist**: Combine finance domain expertise with data science/ML.

3. **FinTech roles**: Companies like Stripe, Plaid, or financial institutions need people who 
   understand both finance and technology.

To explore these options, I'd recommend:
- Build a portfolio project using financial data (stock analysis, budget optimization, etc.)
- Consider a data analytics certificate or bootcamp to formalize your Python skills
- Network with people in these roles via LinkedIn or local meetups

Which of these directions sounds most appealing to you?
```

**Best Practices for Multi-Turn Conversations:**
1. **Maintain full history** - Include all previous messages
2. **Use assistant role** - Store AI responses to maintain context
3. **System message persists** - Define behavior once, applies to all turns
4. **Manage context window** - Summarize or truncate old messages when needed
5. **Track conversation state** - Know where user is in the flow

---

## Part 5: Common Patterns and Templates

### Exercise 5.1: Create Reusable Prompt Templates

**Solution: Prompt Template Library**

```python
class PromptTemplates:
    """Collection of reusable prompt templates."""
    
    @staticmethod
    def summarization(text, max_words=100, style="concise"):
        """Template for text summarization."""
        styles = {
            "concise": "Create a brief, factual summary focusing only on the main points.",
            "detailed": "Create a comprehensive summary that preserves important details and context.",
            "executive": "Create an executive summary highlighting key decisions, metrics, and action items.",
            "eli5": "Explain the main ideas in simple terms that a 5-year-old could understand."
        }
        
        return f"""Summarize the following text in approximately {max_words} words.
        
Style: {styles.get(style, styles['concise'])}

Text to summarize:
\"\"\"{text}\"\"\"

Summary:"""
    
    @staticmethod
    def classification(text, categories, provide_confidence=True):
        """Template for text classification."""
        categories_str = ", ".join(categories)
        
        confidence_instruction = ""
        if provide_confidence:
            confidence_instruction = "\nConfidence: [High/Medium/Low]"
        
        return f"""Classify the following text into one of these categories: {categories_str}

Text: "{text}"

Category: [Your answer]{confidence_instruction}"""
    
    @staticmethod
    def extraction(text, fields):
        """Template for information extraction."""
        fields_str = "\n".join(f"- {field}" for field in fields)
        
        return f"""Extract the following information from the text below:
{fields_str}

Text:
\"\"\"{text}\"\"\"

Extracted Information:"""
    
    @staticmethod
    def comparison(item1, item2, criteria):
        """Template for comparing two items."""
        criteria_str = "\n".join(f"- {criterion}" for criterion in criteria)
        
        return f"""Compare {item1} and {item2} based on the following criteria:
{criteria_str}

Provide a balanced comparison covering each criterion, then give an overall assessment.

Format your response as:
**{item1}:**
[Analysis]

**{item2}:**
[Analysis]

**Overall:**
[Conclusion]"""
    
    @staticmethod
    def format_conversion(text, input_format, output_format):
        """Template for converting between formats."""
        return f"""Convert the following {input_format} to {output_format}.

Input ({input_format}):
\"\"\"{text}\"\"\"

Output ({output_format}):"""
    
    @staticmethod
    def question_answering(context, question, require_citation=True):
        """Template for question answering with context."""
        citation_instruction = ""
        if require_citation:
            citation_instruction = "\n\nIf the answer is in the context, cite the relevant portion. If the answer is not in the context, say 'The provided context does not contain this information.'"
        
        return f"""Answer the following question based on the context provided.

Context:
\"\"\"{context}\"\"\"

Question: {question}{citation_instruction}

Answer:"""

# Usage examples
templates = PromptTemplates()

# Example 1: Summarization
text = "Long article text here..."
prompt = templates.summarization(text, max_words=50, style="executive")

# Example 2: Classification
feedback = "The product is okay but shipping took forever."
prompt = templates.classification(
    feedback, 
    categories=["Positive", "Negative", "Neutral", "Mixed"],
    provide_confidence=True
)

# Example 3: Comparison
prompt = templates.comparison(
    "Python",
    "JavaScript",
    criteria=["Performance", "Learning Curve", "Use Cases", "Community Support"]
)
```

---

### Exercise 5.2: Chain Multiple Prompts

**Task:** Break a complex task into multiple simpler prompts.

**Solution: Article Analysis Pipeline**

```python
def analyze_article_pipeline(article_text):
    """Multi-step analysis pipeline for an article."""
    
    results = {}
    
    # Step 1: Extract key information
    extraction_prompt = f"""Extract the following information from this article:
    - Main Topic
    - Key Arguments (list 3-5)
    - Target Audience
    - Tone (e.g., informative, persuasive, critical)
    
    Article:
    \"\"\"{article_text}\"\"\"
    
    Provide as JSON."""
    
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.3
    )
    results["extraction"] = response1.choices[0].message.content
    
    # Step 2: Summarize (using extraction results)
    summary_prompt = f"""Based on this article analysis:
    {results["extraction"]}
    
    Create a 2-paragraph summary suitable for a newsletter."""
    
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.5
    )
    results["summary"] = response2.choices[0].message.content
    
    # Step 3: Generate discussion questions
    questions_prompt = f"""Based on this article about {article_text[:200]}...
    
    Generate 5 thought-provoking discussion questions for a reading group.
    Questions should encourage critical thinking and diverse perspectives."""
    
    response3 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": questions_prompt}],
        temperature=0.7
    )
    results["questions"] = response3.choices[0].message.content
    
    # Step 4: Create social media posts
    social_prompt = f"""Based on this article summary:
    {results["summary"]}
    
    Create 3 social media posts:
    1. Twitter/X (280 characters max)
    2. LinkedIn (professional tone, 150 words)
    3. Instagram caption (engaging, with hashtags)"""
    
    response4 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": social_prompt}],
        temperature=0.8
    )
    results["social_posts"] = response4.choices[0].message.content
    
    return results

# Usage
article = """
[Your article text here]
"""

analysis = analyze_article_pipeline(article)

print("=== EXTRACTION ===")
print(analysis["extraction"])
print("\n=== SUMMARY ===")
print(analysis["summary"])
print("\n=== DISCUSSION QUESTIONS ===")
print(analysis["questions"])
print("\n=== SOCIAL MEDIA POSTS ===")
print(analysis["social_posts"])
```

**Why Chaining Works:**
1. **Each step has a focused goal** - Easier to optimize individual prompts
2. **Results build on each other** - Later steps use earlier outputs
3. **Different temperatures** - Match creativity level to task
4. **Error isolation** - If one step fails, others may still work
5. **Modularity** - Steps can be reused or rearranged

---

## Key Takeaways

### Do's ✅
1. **Be specific** - Clear instructions yield better results
2. **Provide context** - Give the model necessary background
3. **Use examples** - Show desired format when possible
4. **Set constraints** - Word limits, style guidelines, format requirements
5. **Iterate** - Test and refine prompts based on results
6. **Structure output** - Request JSON, markdown, or specific formats
7. **Test edge cases** - Try unusual inputs to find weaknesses

### Don'ts ❌
1. **Don't be vague** - "Write something good" is not helpful
2. **Don't overload** - Too many instructions in one prompt confuses the model
3. **Don't assume** - Model doesn't remember previous sessions
4. **Don't ignore parameters** - Temperature and max_tokens matter
5. **Don't skip testing** - Test prompts before production use
6. **Don't forget error handling** - API calls can fail
7. **Don't hardcode** - Use templates for reusable prompts

---

## Performance Benchmarks

### Typical Response Times (GPT-3.5-turbo)
- Simple classification: 1-2 seconds
- Short summarization (100 words): 2-3 seconds
- Long-form generation (500 words): 5-8 seconds
- Complex reasoning: 8-12 seconds

### Cost Estimates (GPT-3.5-turbo, 2025 pricing)
- Input: $0.0015 per 1K tokens
- Output: $0.002 per 1K tokens
- Average prompt cost: $0.0001 - $0.001
- 1000 requests: ~$0.50 - $1.00

---

## Next Steps

1. **Complete Lab 2** - Few-Shot Learning Experiments
2. **Review prompt patterns** - Study reusable templates
3. **Build your prompt library** - Create templates for common tasks
4. **Practice iteration** - Take a working prompt and make it better
5. **Explore advanced techniques** - Chain-of-thought (Lab 3)

---

## Additional Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Cheatsheet](../resources/prompt-cheatsheet.md)
- [Example Prompts Library](../resources/example-prompts.md)
- Research: "Language Models are Few-Shot Learners" (Brown et al., 2020)
