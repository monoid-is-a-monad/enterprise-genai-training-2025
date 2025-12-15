# Lab 01 – First Steps with OpenAI API (Instructor Solution)

## Overview

This solution notebook demonstrates the reference implementations for every TODO in the learner lab. The snippets below presume the `OpenAI` client has already been initialised as `client` and that environment variables are loaded through `dotenv`.

---

## Exercise 1.1 – Helper Chat Function

```python
def chat(message: str, model: str = "gpt-3.5-turbo", **kwargs) -> str:
    """Send a prompt to the chat completions endpoint and return the text payload."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        **kwargs
    )
    return response.choices[0].message.content

# Example
print(chat("What is machine learning?", temperature=0.3))
```

The helper accepts arbitrary keyword overrides so students can reuse it throughout parameter experiments.

---

## Exercise 2.1 – Temperature Selection

```python
tasks = {
    "translation": {
        "prompt": "Translate to French: 'Hello, how are you?'",
        "temperature": 0.0
    },
    "creative": {
        "prompt": "Write a haiku about technology.",
        "temperature": 0.8
    },
    "factual": {
        "prompt": "What is the speed of light?",
        "temperature": 0.1
    },
    "brainstorm": {
        "prompt": "List 3 creative uses for a paperclip.",
        "temperature": 0.9
    }
}
```

Low temperatures are preferred for deterministic outputs (translation, factual), while ideation tasks benefit from more randomness.

---

## Exercise 3.1 – Persona System Messages

```python
customer_support_system = (
    "You are a compassionate customer support specialist. Respond with empathy, offer clear steps, "
    "and acknowledge the user's frustration."
)
code_reviewer_system = (
    "You are a meticulous senior engineer. Provide code review feedback focusing on correctness, "
    "readability, and potential edge cases."
)
creative_writer_system = (
    "You are an imaginative storyteller. Use vivid imagery, poetic language, and surprising twists."
)

test_message = "The application crashed when I clicked submit."
print(chat_with_system(test_message, customer_support_system))
print(chat_with_system(code_snippet, code_reviewer_system))
print(chat_with_system("A rainy day", creative_writer_system))
```

The responses should display tone changes that align with each persona’s guidance.

---

## Exercise 4.1 – Cost Comparison Summary

Run the supplied loop unchanged; instructor reference output (token counts vary slightly by run):

```
gpt-3.5-turbo: tokens≈180, cost≈$0.00027
GPT-4:         tokens≈220, cost≈$0.01320
```

Use this comparison to discuss trade-offs between quality and budget constraints.

---

## Challenge 1 – `SmartChatbot`

```python
class SmartChatbot:
    def __init__(self, system_message: str, max_tokens: int = 3500, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.max_tokens = max_tokens
        self.counter = tiktoken.encoding_for_model(model)
        self.messages = [{"role": "system", "content": system_message}]

    def _count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(self.counter.encode(msg["content"])) + 4 for msg in messages) + 2

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def summarize_history(self) -> None:
        summary_prompt = [
            {"role": "system", "content": "Summarise the following dialogue in three bullet points."},
            *self.messages[1:],
        ]
        summary = client.chat.completions.create(model=self.model, messages=summary_prompt)
        self.messages = [self.messages[0], {"role": "system", "content": f"Conversation summary: {summary.choices[0].message.content}"}]

    def get_response(self, user_message: str) -> str:
        self.add_message("user", user_message)
        if self._count_tokens(self.messages) > 0.8 * self.max_tokens:
            self.summarize_history()
            self.add_message("user", user_message)

        response = client.chat.completions.create(model=self.model, messages=self.messages)
        assistant_text = response.choices[0].message.content
        self.add_message("assistant", assistant_text)
        return assistant_text
```

---

## Challenge 2 – `APILogger`

```python
class APILogger:
    def __init__(self):
        self.calls: list[dict[str, float | str]] = []

    def log_call(self, model: str, prompt_tokens: int, completion_tokens: int, duration: float) -> None:
        cost = estimate_cost(prompt_tokens, completion_tokens, model=model)
        self.calls.append({
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": cost["total_tokens"],
            "cost": cost["total_cost"],
            "duration": duration,
        })

    def get_report(self) -> dict[str, float | int]:
        totals = {
            "calls": len(self.calls),
            "total_tokens": sum(call["total_tokens"] for call in self.calls),
            "total_cost": sum(call["cost"] for call in self.calls),
            "avg_latency": sum(call["duration"] for call in self.calls) / max(1, len(self.calls)),
        }
        return totals

    def export_csv(self, filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.calls[0].keys())
            writer.writeheader()
            writer.writerows(self.calls)
```

---

## Challenge 3 – `compare_models`

```python
def compare_models(prompt: str, models: list[str] = None) -> list[dict[str, object]]:
    models = models or ["gpt-3.5-turbo", "gpt-4"]
    results = []
    for model in models:
        start = time.time()
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        elapsed = time.time() - start
        usage = response.usage
        cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens, model=model)
        results.append({
            "model": model,
            "latency_sec": round(elapsed, 2),
            "tokens": usage.total_tokens,
            "cost_usd": round(cost["total_cost"], 6),
            "preview": response.choices[0].message.content[:160],
        })
    return results
```

Use the comparison report to debrief pros/cons of each model for different workloads.

---

These reference implementations cover all mandatory tasks for Lab 01. Feel free to extend them with provider-specific wrappers or additional logging in advanced cohorts.
