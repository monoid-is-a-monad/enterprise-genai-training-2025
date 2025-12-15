# Lab 02 – Text Generation Experiments (Instructor Solution)

## Exercise 1.1 – Persona-Aware Chat Helper

```python
def chat_with_persona(user_message: str, persona: str) -> str:
    personas = {
        "formal": "You are a formal business consultant. Use structured, professional language and back claims with evidence.",
        "casual": "You are a friendly buddy. Keep things light, add emojis sparingly, and encourage questions.",
        "technical": "You are a senior engineer. Provide precise, technically accurate explanations with short examples when helpful.",
        "creative": "You are an imaginative storyteller. Use vivid imagery, metaphors, and engaging narrative devices."
    }
    system_message = personas.get(persona, personas["formal"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

for persona in ["formal", "casual", "technical", "creative"]:
    print(persona.upper(), chat_with_persona("Explain what cloud computing is.", persona), sep="\n")
```

---

## Exercise 2.1 – Parameter Recommendations

```python
use_cases = {
    "legal_document": {
        "prompt": "Draft a privacy policy section about data collection.",
        "temperature": 0.1,
        "top_p": 0.4,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.0
    },
    "creative_story": {
        "prompt": "Write an opening paragraph for a mystery novel.",
        "temperature": 0.9,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.6
    },
    "technical_docs": {
        "prompt": "Explain how to implement binary search in Python.",
        "temperature": 0.2,
        "top_p": 0.5,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    },
    "marketing_copy": {
        "prompt": "Write 5 taglines for an eco-friendly water bottle.",
        "temperature": 0.85,
        "top_p": 0.9,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3
    }
}
```

Discuss with learners how penalties nudge novelty versus adherence to requirements.

---

## Exercise 3.1 – `EnhancedChatbot`

```python
class EnhancedChatbot(Chatbot):
    def __init__(self, system_message: str = "You are a helpful assistant.", model: str = "gpt-3.5-turbo", max_tokens: int = 4000):
        super().__init__(system_message, model)
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(model)

    def get_current_tokens(self) -> int:
        return self.token_counter.count_message_tokens(self.messages)

    def summarize_conversation(self) -> None:
        summary_prompt = [
            {"role": "system", "content": "Summarise the dialogue below in three bullet points highlighting key decisions and questions."},
            *self.messages[1:],
        ]
        summary = client.chat.completions.create(model=self.model, messages=summary_prompt, temperature=0.2)
        self.messages = [self.messages[0], {"role": "system", "content": f"Conversation summary: {summary.choices[0].message.content}"}]

    def check_and_manage_tokens(self) -> None:
        if self.get_current_tokens() >= 0.8 * self.max_tokens:
            self.summarize_conversation()

    def chat(self, user_message: str) -> str:
        self.check_and_manage_tokens()
        return super().chat(user_message)

    def save_conversation(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.messages, handle, indent=2)

    def load_conversation(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as handle:
            self.messages = json.load(handle)
```

---

## Exercise 4.1 – Cost-Aware Chatbot

```python
class CostAwareChatbot:
    def __init__(self, system_message: str, model: str = "gpt-3.5-turbo", cost_threshold: float = 0.10):
        self.model = model
        self.messages = [{"role": "system", "content": system_message}]
        self.cost_threshold = cost_threshold
        self.total_cost = 0.0
        self.counter = TokenCounter(model)
        self.history: list[dict[str, object]] = []

    def chat(self, user_message: str) -> dict[str, object]:
        self.messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(model=self.model, messages=self.messages, temperature=0.6)
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        usage = response.usage
        cost_info = self.counter.estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        self.total_cost += cost_info["total_cost"]
        entry = {
            "user": user_message,
            "assistant": assistant_message,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost": cost_info["total_cost"],
        }
        self.history.append(entry)
        alert = self.check_threshold()
        return {"response": assistant_message, "exchange_cost": entry["cost"], "total_cost": self.total_cost, "threshold_alert": alert}

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_cost_report(self) -> dict[str, object]:
        return {
            "exchanges": len(self.history),
            "total_cost": self.total_cost,
            "avg_cost": self.total_cost / max(1, len(self.history))
        }

    def check_threshold(self) -> bool:
        return self.total_cost >= self.cost_threshold
```

---

## Exercise 5.1 – Streaming Indicators

```python
class EnhancedStreamingChatbot(StreamingChatbot):
    def chat_stream_with_indicators(self, user_message: str):
        print("Assistant is thinking...", end="\r", flush=True)
        token_count = 0
        cost_estimate = 0.0
        stream = client.chat.completions.create(
            model=self.model,
            messages=self.messages + [{"role": "user", "content": user_message}],
            stream=True,
            temperature=0.7
        )
        full = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full += delta
                token_count += 1
                cost_estimate = (token_count / 1000) * 0.0015  # rough output estimate for gpt-3.5
                sys.stdout.write(f"Tokens: {token_count}  Cost≈${cost_estimate:.5f}\r")
                sys.stdout.flush()
        print("\n", full)
        self.messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full}
        ])
        return full
```

---

## Exercise 6.1 – `TextGenerator`

```python
class TextGenerator:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.counter = TokenCounter(model)
        self.generation_history = []

    def _build_prompt(self, content_type: str, topic: str, **kwargs) -> str:
        tone = kwargs.get("tone", "neutral")
        length = kwargs.get("length", "short")
        audience = kwargs.get("audience", "general")
        return (
            f"Generate {content_type} content about {topic}. Tone: {tone}. Length: {length}. "
            f"Audience: {audience}. Include the key points {kwargs.get('key_points', 'where relevant')}"
        )

    def generate(self, content_type: str, topic: str, **kwargs) -> dict[str, object]:
        prompt = self._build_prompt(content_type, topic, **kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 400)
        )
        content = response.choices[0].message.content
        usage = response.usage
        cost = self.counter.estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        record = {"content_type": content_type, "topic": topic, "content": content, **cost}
        self.generation_history.append(record)
        if filepath := kwargs.get("export_path"):
            with open(filepath, "w", encoding="utf-8") as handle:
                handle.write(content)
        return record

    def batch_generate(self, requests: list[dict[str, object]]) -> list[dict[str, object]]:
        return [self.generate(**request) for request in requests]
```

---

## Challenge Sketches

- **InteractiveStory** – Track `story_state` dict containing current segment, choices, and a token budget. Generate choices via a prompt such as *"Given the story so far, propose three branching options"*. Append user choice to history and generate the next segment with the choice embedded in the prompt.
- **TranslationService** – Wrap a `translate` method that formats prompts as *"Translate the text to {language} preserving tone:"*. Provide `translate_batch` (looping through inputs) and `back_translate` (call `translate` twice) to measure divergence.
- **ContentRepurposer** – Normalise input content, then for each platform call the model with platform-specific instructions (length, hashtags, CTA). Store outputs together with token/cost metadata so learners can inspect trade-offs.

The reference implementations above unblock the Week 1 learners while leaving room for instructors to discuss production hardening (async batching, retries, caching, etc.).
