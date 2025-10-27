# Week 01 - Resources

## 📚 Essential References

### Official Documentation

- **[OpenAI Platform Documentation](https://platform.openai.com/docs)** - Complete API reference
- **[OpenAI Cookbook](https://github.com/openai/openai-cookbook)** - Code examples and guides
- **[Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)** - Enterprise deployment

### Research Papers

1. **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** (2017)
   - Original Transformer architecture paper
   - Foundational for all modern LLMs

2. **["BERT: Pre-training of Deep Bidirectional Transformers"](https://arxiv.org/abs/1810.04805)** (2018)
   - Bidirectional language understanding
   - Context-aware embeddings

3. **["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165)** (2020)
   - GPT-3 paper
   - Demonstrates emergence of capabilities

4. **["Training language models to follow instructions"](https://arxiv.org/abs/2203.02155)** (2022)
   - InstructGPT and RLHF
   - Alignment techniques

### Interactive Tutorials

- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** - Visual explanation
- **[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)** - Understanding GPT models
- **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** - Interactive visualization

### Video Content

- **[3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** - Visual math explanation
- **[Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)** - Building from scratch
- **[DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/)** - Free practical courses

## 🛠️ Tools & Libraries

### Essential Python Libraries

```python
# Core AI/ML
openai>=1.0.0           # OpenAI API client
anthropic>=0.7.0        # Claude API client
langchain>=0.1.0        # LLM orchestration framework
llama-index>=0.9.0      # Data framework for LLMs

# Data handling
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
python-dotenv>=1.0.0    # Environment variables

# Web & API
requests>=2.31.0        # HTTP library
httpx>=0.25.0           # Async HTTP client
fastapi>=0.104.0        # API framework

# Utilities
tiktoken>=0.5.0         # Token counting
tenacity>=8.2.0         # Retry logic
pydantic>=2.0.0         # Data validation
```

### Development Environment

**Recommended IDE:**
- VS Code with extensions:
  - Python
  - Jupyter
  - GitHub Copilot (optional)
  - Pylance

**Alternative:**
- PyCharm Professional
- JupyterLab
- Google Colab (for experimentation)

### Useful Command Line Tools

```bash
# Token counting
pip install tiktoken

# API testing
pip install httpie

# Environment management
pip install python-dotenv

# Notebook tools
pip install jupyter nbconvert
```

## 📖 Glossary of Terms

### Core Concepts

**API (Application Programming Interface)**
- Interface for programmatic access to services
- RESTful APIs use HTTP requests

**Attention Mechanism**
- Allows models to focus on relevant parts of input
- Core innovation in Transformer architecture

**Embeddings**
- Numerical representations of text
- Capture semantic meaning in vector space

**Fine-tuning**
- Training a pre-trained model on specific data
- Adapts model to particular tasks

**Few-shot Learning**
- Learning from a small number of examples
- Provided in the prompt

**Hallucination**
- When AI generates plausible but incorrect information
- Common challenge in LLMs

**Inference**
- Using a trained model to generate outputs
- Production deployment phase

**Large Language Model (LLM)**
- Neural network trained on massive text data
- Can understand and generate human language

**Parameters**
- Learned weights in neural network
- More parameters generally = more capability

**Prompt**
- Input text given to the model
- Instructions or context for generation

**Temperature**
- Controls randomness in generation
- Higher = more creative, lower = more focused

**Tokens**
- Basic units of text (words, subwords, characters)
- Pricing and limits based on token count

**Transformer**
- Neural network architecture using attention
- Foundation for modern LLMs

**Vector Database**
- Database optimized for similarity search
- Used in RAG systems

**Zero-shot Learning**
- Performing tasks without specific training examples
- Just from instructions

### Advanced Terms

**RLHF (Reinforcement Learning from Human Feedback)**
- Training technique using human preferences
- Aligns models with human values

**LoRA (Low-Rank Adaptation)**
- Efficient fine-tuning method
- Reduces computational requirements

**Retrieval-Augmented Generation (RAG)**
- Combining retrieval with generation
- Grounds responses in specific documents

**Chain-of-Thought**
- Prompting technique encouraging step-by-step reasoning
- Improves complex problem solving

**Function Calling**
- LLMs that can invoke external functions
- Enables tool use and integrations

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### API Key Problems

**Issue:** `AuthenticationError: Incorrect API key provided`

**Solutions:**
1. Check `.env` file exists and contains key
2. Verify key format: `sk-...`
3. Ensure `python-dotenv` is installed
4. Load environment: `load_dotenv()`

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"Key loaded: {bool(api_key)}")  # Should print True
```

#### Rate Limiting

**Issue:** `RateLimitError: Rate limit reached`

**Solutions:**
1. Implement exponential backoff
2. Check your tier limits
3. Add delays between requests
4. Use batch processing

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
def make_api_call():
    return client.chat.completions.create(...)
```

#### Token Limits

**Issue:** `InvalidRequestError: This model's maximum context length is...`

**Solutions:**
1. Count tokens before sending
2. Truncate input if too long
3. Use summarization for long documents
4. Consider models with larger context

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Check before sending
token_count = count_tokens(prompt)
if token_count > 8000:
    print(f"Warning: {token_count} tokens, may exceed limit")
```

#### Import Errors

**Issue:** `ModuleNotFoundError: No module named 'openai'`

**Solutions:**
1. Activate virtual environment
2. Install requirements: `pip install -r requirements.txt`
3. Check Python version: `python --version` (need 3.9+)
4. Verify pip points to venv: `which pip`

#### Response Quality

**Issue:** Generated text is repetitive or low quality

**Solutions:**
1. Adjust temperature (try 0.7-0.9)
2. Use presence_penalty to reduce repetition
3. Improve prompt clarity
4. Try different models

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    temperature=0.8,
    presence_penalty=0.6,  # Reduces repetition
    frequency_penalty=0.3   # Penalizes frequent tokens
)
```

## 📊 Comparison Charts

### Model Comparison Matrix

| Model | Provider | Context Length | Strengths | Best For |
|-------|----------|----------------|-----------|----------|
| GPT-4 | OpenAI | 128K | Most capable, multimodal | Complex tasks, reasoning |
| GPT-3.5-turbo | OpenAI | 16K | Fast, cost-effective | Simple tasks, high volume |
| Claude 3 | Anthropic | 200K | Long context, safe | Document analysis, research |
| Gemini Pro | Google | 32K | Multimodal, integrated | Google ecosystem |
| Llama 2 | Meta | 4K | Open source | Self-hosting, customization |

### Pricing Overview (Approximate)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Use Case |
|-------|----------------------|------------------------|----------|
| GPT-4-turbo | $10 | $30 | Production, complex |
| GPT-3.5-turbo | $0.50 | $1.50 | Development, simple |
| Claude 3 | $3-15 | $15-75 | Long documents |
| Open Source | Hosting costs | Hosting costs | Custom needs |

*Prices subject to change. Check provider websites for current rates.*

### Temperature Effects

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Deterministic, focused | Data extraction, factual Q&A |
| 0.3-0.5 | Mostly consistent | Technical writing, translations |
| 0.7-0.9 | Balanced creativity | General content, brainstorming |
| 1.0-1.5 | Highly creative | Creative writing, ideation |
| 1.5+ | Very random | Experimental, artistic |

## 🎯 Best Practices

### Prompt Engineering Basics

1. **Be Specific:**
   - ❌ "Write about AI"
   - ✅ "Write a 500-word explanation of transformer architecture for beginners"

2. **Provide Context:**
   ```
   You are an expert Python developer.
   Write a function that processes customer data...
   ```

3. **Use Examples:**
   ```
   Input: "Hello world"
   Output: "HELLO WORLD"
   
   Input: "Python is great"
   Output: [generate similar transformation]
   ```

4. **Set Constraints:**
   - Specify length, format, tone
   - Define what to include/exclude
   - Request structured output

### API Usage Best Practices

1. **Error Handling:**
   ```python
   try:
       response = client.chat.completions.create(...)
   except openai.RateLimitError:
       # Handle rate limiting
   except openai.APIError:
       # Handle API errors
   except Exception as e:
       # Handle other errors
   ```

2. **Logging:**
   ```python
   import logging
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   logger.info(f"API call with {token_count} tokens")
   ```

3. **Cost Monitoring:**
   ```python
   def estimate_cost(input_tokens, output_tokens, model="gpt-4"):
       input_cost = input_tokens / 1_000_000 * 10  # $10 per 1M
       output_cost = output_tokens / 1_000_000 * 30  # $30 per 1M
       return input_cost + output_cost
   ```

## 🔗 Additional Resources

### Community & Forums

- [OpenAI Community Forum](https://community.openai.com/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [r/LanguageTechnology](https://www.reddit.com/r/LanguageTechnology/)
- [Stack Overflow - LLM tag](https://stackoverflow.com/questions/tagged/llm)

### Newsletters & Blogs

- [The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/)
- [Import AI](https://importai.substack.com/)
- [Ahead of AI](https://magazine.sebastianraschka.com/)
- [The Gradient](https://thegradient.pub/)

### GitHub Repositories

- [Awesome LLM](https://github.com/Hannibal046/Awesome-LLM)
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)
- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)

---

**Last Updated:** October 27, 2025  
**Maintained By:** Training Team  
**Version:** 1.0
