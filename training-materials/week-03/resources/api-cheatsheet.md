# OpenAI API Cheatsheet (Python)

## Setup
```python
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## Chat Completion
```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize this: ..."}
    ],
    temperature=0.3,
    max_tokens=400
)
print(resp.choices[0].message.content)
```

## Parallel Tool Use (function calling)
```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[
        {
          "type": "function",
          "function": {
            "name": "search_docs",
            "description": "Semantic search",
            "parameters": {
              "type": "object",
              "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10}
              },
              "required": ["query"]
            }
          }
        }
    ],
    tool_choice="auto"
)
```

## Embeddings
```python
emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=["First text", "Second text"]
)
vectors = [d.embedding for d in emb.data]
```

## Images (optional)
```python
img = client.images.generate(
    model="gpt-image-1",
    prompt="diagram of a rag pipeline"
)
```

## Error Handling (basic)
```python
try:
    resp = client.chat.completions.create(...)
except Exception as e:
    # log and retry with backoff
    print("API error", e)
```
