# Function Calling & Tool Use Cheatsheet

## Define a Tool
```python
search_tool = {
  "type": "function",
  "function": {
    "name": "search_docs",
    "description": "Semantic search over documentation",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 20}
      },
      "required": ["query"]
    }
  }
}
```

## Execute Tool Calls
```python
tool_calls = resp.choices[0].message.tool_calls or []
for call in tool_calls:
    name = call.function.name
    args = json.loads(call.function.arguments)
    if name == "search_docs":
        result = my_search(**args)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": name,
            "content": json.dumps(result)
        })
```

## Safety & Control
- Maintain a whitelist of allowed tools
- Validate JSON with JSON Schema before execution
- Timeouts and circuit breakers around tool calls
- Log inputs/outputs; scrub sensitive data

## Debugging Tips
- Print raw tool_call arguments
- Echo back schema to the model in the prompt
- Start with single tool; add more incrementally
