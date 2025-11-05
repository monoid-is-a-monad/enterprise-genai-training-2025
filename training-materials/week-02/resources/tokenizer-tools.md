# Tokenizer & Cost Tools

Understand and control tokens for reliability and cost.

## Token Counting
- OpenAI Tokenizer: https://platform.openai.com/tokenizer
- tiktoken (Python): https://github.com/openai/tiktoken

### Quick Python Snippet (tiktoken)
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
text = "Your text here"
print(len(enc.encode(text)))
```

## Pricing & Budgets
- Model pricing: https://openai.com/api/pricing
- Track input vs output tokens separately
- Create per-request budgets (e.g., 3k context, 400 output)

## Tips
- Prefer bullet summaries to long prose in context
- De-duplicate near-identical chunks
- Keep schemas compact; avoid verbose descriptions
- Compress context with extractive summaries when needed
