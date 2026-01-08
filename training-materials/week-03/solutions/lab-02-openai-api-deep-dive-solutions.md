# Lab 2: OpenAI API Deep Dive - Solutions

**Week 3 - Advanced Prompting & OpenAI API**

**Provided by:** ADC ENGINEERING & CONSULTING LTD

---

## Table of Contents

1. [Part 1: Complete Parameter Guide](#part-1-complete-parameter-guide)
2. [Part 2: Error Handling & Retries](#part-2-error-handling--retries)
3. [Part 3: Streaming Responses](#part-3-streaming-responses)
4. [Part 4: Token Management](#part-4-token-management)
5. [Part 5: Rate Limiting & Optimization](#part-5-rate-limiting--optimization)
6. [Part 6: Production API Wrapper](#part-6-production-api-wrapper)
7. [Best Practices](#best-practices)

---

## Part 1: Complete Parameter Guide

### Exercise 1.1: Temperature Experiments

**Temperature Effects:**

```python
# Temperature: 0.0 (Most Deterministic)
Response: "In the year 2157, humanity discovered a signal from deep space."

# Temperature: 0.7 (Balanced)
Response: "The stars whispered secrets through the quantum radio, and she was the only one who could hear them."

# Temperature: 1.5 (Very Creative)
Response: "Zara's consciousness flickered between dimensions as the reality anchor failed, sending her spiraling through impossible geometries."
```

**Key Insights:**
- **0.0-0.3:** Use for factual, consistent outputs (code, data extraction, classification)
- **0.4-0.7:** Balanced creativity and coherence (writing, analysis, Q&A)
- **0.8-1.0:** Creative tasks where variety is desired
- **1.0+:** Experimental/artistic applications (can produce nonsense)

### Exercise 1.2: Top-P (Nucleus Sampling)

```python
# top_p: 0.1 (Very focused)
"Python is popular because it has simple syntax and extensive libraries."

# top_p: 0.5 (Moderately focused)
"Python's popularity stems from its readable syntax, vast ecosystem of packages, strong community support, and versatility across domains."

# top_p: 0.9 (Diverse)
"From web development to machine learning, Python's elegant design philosophy and batteries-included approach have made it a favorite among beginners and experts alike, fostering an incredible community..."

# top_p: 1.0 (All tokens considered)
"Python revolutionized programming by democratizing code accessibility, enabling rapid prototyping, and establishing itself as the lingua franca of data science and AI research."
```

**When to use top_p vs temperature:**
- **top_p:** More stable, easier to control
- **temperature:** More dramatic effects at extremes
- **Best practice:** Use ONE at a time (typically top_p=1.0 with temperature, OR temperature=1.0 with top_p)

### Exercise 1.3: Frequency & Presence Penalties

**Testing Repetition Control:**

```python
prompt = "List reasons why Python is popular. Start each reason with 'Python is'"

# No penalties (0.0, 0.0)
Response:
"""
Python is easy to learn.
Python is easy to read.
Python is easy to use.
Python is widely used in data science.
Python is popular in web development.
"""

# High frequency penalty (2.0, 0.0)
Response:
"""
Python is easy to learn.
The language offers excellent readability.
It has comprehensive libraries for various domains.
Strong community support drives adoption.
Versatility across applications attracts developers.
"""
# Reduced repetition of "Python is"

# High presence penalty (0.0, 2.0)
Response:
"""
Python is easy to learn and readable.
The syntax emphasizes clarity, enabling rapid development.
Extensive libraries support data science, web development, and automation.
A vibrant community provides resources, frameworks, and third-party packages.
Cross-platform compatibility ensures code runs anywhere.
"""
# Encourages mentioning new topics

# Both high (2.0, 2.0)
Response:
"""
Easy-to-learn syntax attracts beginners.
Comprehensive standard library reduces external dependencies.
Strong data science ecosystem with NumPy, pandas, and scikit-learn.
Web frameworks like Django and Flask enable rapid deployment.
Active community contributes packages and documentation.
"""
# Most diverse output, avoids repeating structure and topics
```

**Guidelines:**
- **frequency_penalty:** Reduces repetition of specific tokens (words)
- **presence_penalty:** Encourages introducing new topics/concepts
- **Range:** -2.0 to 2.0 (typically use 0.0 to 1.5)
- **Use cases:**
  - frequency_penalty: Avoiding repetitive phrases in long text
  - presence_penalty: Encouraging comprehensive coverage of topics

### Exercise 1.4: The `n` Parameter

```python
prompt = "Suggest a creative name for a coffee shop."

# n=3 generates 3 completions
Completion 1: "The Daily Grind"
Completion 2: "Brew Haven"
Completion 3: "Caffeine Chronicles"
```

**Cost consideration:** Generating n completions multiplies token costs by n.

**Best use cases:**
- A/B testing different responses
- Giving users choices
- Finding the best response (generate multiple, select best)
- Brainstorming sessions

### Exercise 1.5: Stop Sequences

```python
prompt = "Explain how photosynthesis works."

# No stop
Response: "Photosynthesis is the process by which plants convert sunlight into chemical energy. It occurs in the chloroplasts and involves two main stages: the light-dependent reactions and the Calvin cycle. During the light-dependent reactions, chlorophyll absorbs sunlight..."

# Stop at period ["."]
Response: "Photosynthesis is the process by which plants convert sunlight into chemical energy."

# Stop at newline ["\n"]
Response: "Photosynthesis is the process by which plants convert sunlight into chemical energy. It occurs in two main stages:"
# (Stops before listing stages)

# Multiple stops [".", "!", "\n"]
Response: "Photosynthesis converts sunlight into energy."
# (Stops at first matching sequence)
```

**Use cases:**
- Limiting response length
- Structured output (stop at delimiters)
- Extracting first sentence
- Preventing unwanted content

### Complete Parameter Reference

```python
class ParameterReference:
    """
    Complete OpenAI API parameter reference with best practices.
    """
    
    @staticmethod
    def get_config_for_use_case(use_case: str) -> dict:
        """
        Get optimal parameters for common use cases.
        
        Args:
            use_case: One of 'code', 'creative', 'factual', 'brainstorm', 
                     'analysis', 'chat'
        
        Returns:
            Dictionary of optimal parameters
        """
        configs = {
            "code": {
                "temperature": 0.2,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 1000,
                "stop": None,
                "description": "Deterministic, consistent code generation"
            },
            "creative": {
                "temperature": 0.9,
                "top_p": 1.0,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3,
                "max_tokens": 500,
                "stop": None,
                "description": "Creative writing with variety"
            },
            "factual": {
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 300,
                "stop": None,
                "description": "Factual, deterministic responses"
            },
            "brainstorm": {
                "temperature": 0.8,
                "top_p": 0.95,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
                "n": 3,  # Generate multiple options
                "max_tokens": 200,
                "stop": None,
                "description": "Diverse ideas for brainstorming"
            },
            "analysis": {
                "temperature": 0.3,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.2,
                "max_tokens": 800,
                "stop": None,
                "description": "Analytical, comprehensive coverage"
            },
            "chat": {
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.0,
                "max_tokens": 300,
                "stop": None,
                "description": "Natural, engaging conversation"
            }
        }
        
        return configs.get(use_case, configs["factual"])

# Usage
config = ParameterReference.get_config_for_use_case("code")
print(f"Code Generation Config: {config}")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a Python function to calculate factorial"}],
    **{k: v for k, v in config.items() if k != "description"}
)
```

---

## Part 2: Error Handling & Retries

### Exercise 2.1: Handling Rate Limits

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
import logging

logger = logging.getLogger(__name__)

class RobustAPIClient:
    """
    OpenAI API client with comprehensive error handling.
    """
    
    def __init__(self, max_retries: int = 3, initial_wait: int = 1, max_wait: int = 60):
        self.client = OpenAI()
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def completion_with_retry(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Make API call with automatic retry on transient errors.
        
        Args:
            messages: Chat messages
            **kwargs: Additional API parameters
        
        Returns:
            Response text
        
        Raises:
            Exception: If all retries fail
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
            
        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}. Retrying...")
            raise  # Let tenacity handle retry
            
        except APITimeoutError as e:
            logger.warning(f"API timeout: {e}. Retrying...")
            raise
            
        except APIConnectionError as e:
            logger.warning(f"Connection error: {e}. Retrying...")
            raise
            
        except APIError as e:
            logger.error(f"API error (not retrying): {e}")
            raise
    
    def safe_completion(
        self,
        messages: List[Dict[str, str]],
        fallback_response: str = "I'm experiencing technical difficulties. Please try again.",
        **kwargs
    ) -> Dict[str, any]:
        """
        Make API call with fallback on error.
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            response = self.completion_with_retry(messages, **kwargs)
            return {
                "success": True,
                "response": response,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"All retries failed: {e}")
            return {
                "success": False,
                "response": fallback_response,
                "error": str(e)
            }

# Usage
robust_client = RobustAPIClient(max_retries=3)

messages = [{"role": "user", "content": "What is machine learning?"}]
result = robust_client.safe_completion(
    messages=messages,
    model="gpt-3.5-turbo",
    temperature=0.7
)

if result["success"]:
    print(f"Response: {result['response']}")
else:
    print(f"Error occurred: {result['error']}")
    print(f"Fallback response: {result['response']}")
```

### Exercise 2.2: Timeout Handling

```python
import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError

class TimeoutAPIClient:
    """
    API client with timeout handling.
    """
    
    def __init__(self, default_timeout: int = 30):
        self.client = OpenAI()
        self.default_timeout = default_timeout
    
    async def completion_with_timeout(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Make API call with timeout.
        
        Args:
            messages: Chat messages
            timeout: Timeout in seconds (uses default if None)
            **kwargs: API parameters
        
        Returns:
            Response dictionary
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Create async client
            async_client = AsyncOpenAI()
            
            # Run with timeout
            response = await asyncio.wait_for(
                async_client.chat.completions.create(
                    messages=messages,
                    **kwargs
                ),
                timeout=timeout
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "timed_out": False,
                "usage": response.usage._asdict()
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "response": None,
                "timed_out": True,
                "error": f"Request exceeded {timeout}s timeout"
            }
        
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "timed_out": False,
                "error": str(e)
            }

# Usage
async def main():
    timeout_client = TimeoutAPIClient(default_timeout=10)
    
    messages = [{"role": "user", "content": "Explain quantum computing in detail"}]
    
    result = await timeout_client.completion_with_timeout(
        messages=messages,
        model="gpt-4",
        timeout=5  # 5 second timeout
    )
    
    if result["timed_out"]:
        print("Request timed out!")
    elif result["success"]:
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result['error']}")

# Run async
asyncio.run(main())
```

### Exercise 2.3: Validation & Fallback

```python
from pydantic import BaseModel, ValidationError
import json

class ResponseValidator:
    """
    Validate and sanitize API responses.
    """
    
    @staticmethod
    def validate_json_response(response: str, schema: BaseModel) -> Dict[str, any]:
        """
        Validate JSON response against schema.
        
        Args:
            response: API response text
            schema: Pydantic model for validation
        
        Returns:
            Validated data or error info
        """
        try:
            # Try to parse JSON
            data = json.loads(response)
            
            # Validate against schema
            validated = schema(**data)
            
            return {
                "valid": True,
                "data": validated.dict(),
                "error": None
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "data": None,
                "error": f"Invalid JSON: {e}"
            }
        
        except ValidationError as e:
            return {
                "valid": False,
                "data": None,
                "error": f"Schema validation failed: {e}"
            }
    
    @staticmethod
    def safe_json_completion(
        client: OpenAI,
        messages: List[Dict[str, str]],
        schema: BaseModel,
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, any]:
        """
        Get validated JSON response with retries.
        
        Returns:
            Validated response or error
        """
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    response_format={"type": "json_object"},  # Force JSON
                    **kwargs
                )
                
                text = response.choices[0].message.content
                
                # Validate
                validation_result = ResponseValidator.validate_json_response(text, schema)
                
                if validation_result["valid"]:
                    return validation_result
                
                # If invalid, add error to prompt and retry
                messages.append({
                    "role": "assistant",
                    "content": text
                })
                messages.append({
                    "role": "user",
                    "content": f"The response was invalid: {validation_result['error']}. Please provide a valid response."
                })
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "valid": False,
                        "data": None,
                        "error": f"All attempts failed: {e}"
                    }
        
        return {
            "valid": False,
            "data": None,
            "error": "Max retries exceeded"
        }

# Example usage
from pydantic import Field

class PersonInfo(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    occupation: str
    email: str

client = OpenAI()

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that outputs JSON."
    },
    {
        "role": "user",
        "content": "Generate information for a fictional person. Return as JSON with fields: name, age, occupation, email."
    }
]

result = ResponseValidator.safe_json_completion(
    client=client,
    messages=messages,
    schema=PersonInfo,
    model="gpt-3.5-turbo"
)

if result["valid"]:
    print(f"Valid data: {result['data']}")
else:
    print(f"Validation failed: {result['error']}")
```

---

## Part 3: Streaming Responses

### Exercise 3.1: Basic Streaming

```python
def stream_completion(prompt: str, model: str = "gpt-3.5-turbo"):
    """
    Stream a completion and print in real-time.
    
    Args:
        prompt: User prompt
        model: Model to use
    """
    print("Streaming response:", end=" ", flush=True)
    
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    full_response = ""
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print()  # New line
    return full_response

# Usage
response = stream_completion("Explain how recursion works in programming.")
```

**Output:**
```
Streaming response: Recursion is a programming technique where a function calls itself to solve a problem. It breaks down complex problems into smaller, manageable subproblems...
```

### Exercise 3.2: Streaming with Event Handlers

```python
from typing import Callable, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StreamEvent:
    """Event during streaming."""
    type: str  # 'start', 'chunk', 'end', 'error'
    content: Optional[str] = None
    timestamp: datetime = None
    metadata: Optional[Dict] = None

class StreamHandler:
    """
    Handle streaming responses with callbacks.
    """
    
    def __init__(
        self,
        on_start: Optional[Callable[[StreamEvent], None]] = None,
        on_chunk: Optional[Callable[[StreamEvent], None]] = None,
        on_end: Optional[Callable[[StreamEvent], None]] = None,
        on_error: Optional[Callable[[StreamEvent], None]] = None
    ):
        self.on_start = on_start or self._default_handler
        self.on_chunk = on_chunk or self._default_handler
        self.on_end = on_end or self._default_handler
        self.on_error = on_error or self._default_handler
    
    def _default_handler(self, event: StreamEvent):
        """Default event handler."""
        pass
    
    def stream_with_callbacks(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> str:
        """
        Stream completion with event callbacks.
        
        Returns:
            Complete response text
        """
        # Emit start event
        self.on_start(StreamEvent(
            type='start',
            timestamp=datetime.now(),
            metadata={'model': model}
        ))
        
        full_response = ""
        chunk_count = 0
        
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    chunk_count += 1
                    
                    # Emit chunk event
                    self.on_chunk(StreamEvent(
                        type='chunk',
                        content=content,
                        timestamp=datetime.now(),
                        metadata={'chunk_number': chunk_count}
                    ))
            
            # Emit end event
            self.on_end(StreamEvent(
                type='end',
                content=full_response,
                timestamp=datetime.now(),
                metadata={
                    'total_chunks': chunk_count,
                    'total_length': len(full_response)
                }
            ))
            
            return full_response
            
        except Exception as e:
            # Emit error event
            self.on_error(StreamEvent(
                type='error',
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            ))
            raise

# Example usage with custom handlers
def on_start_handler(event: StreamEvent):
    print(f"[{event.timestamp.strftime('%H:%M:%S')}] Starting stream with {event.metadata['model']}")

def on_chunk_handler(event: StreamEvent):
    print(event.content, end="", flush=True)

def on_end_handler(event: StreamEvent):
    print(f"\n\n[{event.timestamp.strftime('%H:%M:%S')}] Stream complete!")
    print(f"Total chunks: {event.metadata['total_chunks']}")
    print(f"Total length: {event.metadata['total_length']} characters")

def on_error_handler(event: StreamEvent):
    print(f"\n[ERROR] {event.metadata['error']}")

handler = StreamHandler(
    on_start=on_start_handler,
    on_chunk=on_chunk_handler,
    on_end=on_end_handler,
    on_error=on_error_handler
)

messages = [{"role": "user", "content": "Write a short story about a robot."}]
response = handler.stream_with_callbacks(messages)
```

### Exercise 3.3: Async Streaming

```python
async def async_stream_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    **kwargs
) -> AsyncIterator[str]:
    """
    Async generator for streaming completions.
    
    Yields:
        Content chunks as they arrive
    """
    async_client = AsyncOpenAI()
    
    stream = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **kwargs
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Usage
async def main():
    messages = [{"role": "user", "content": "Count from 1 to 10 slowly."}]
    
    print("Async streaming:", end=" ")
    async for chunk in async_stream_completion(messages):
        print(chunk, end="", flush=True)
        await asyncio.sleep(0.1)  # Simulate processing
    print()

asyncio.run(main())
```

---

## Part 4: Token Management

### Exercise 4.1: Token Counting

```python
import tiktoken

class TokenManager:
    """
    Manage and count tokens for API requests.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
        
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in message list.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            Total tokens (including formatting tokens)
        """
        tokens = 0
        
        for message in messages:
            # Every message has 4 tokens for formatting
            tokens += 4
            
            for key, value in message.items():
                tokens += self.count_tokens(value)
                
                # Role tokens
                if key == "name":
                    tokens += -1  # Role is omitted if name is present
        
        # Every reply is primed with 2 tokens
        tokens += 2
        
        return tokens
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-3.5-turbo"
    ) -> float:
        """
        Estimate API call cost.
        
        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model name
        
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (verify current pricing)
        pricing = {
            "gpt-3.5-turbo": {
                "input": 0.0005 / 1000,   # $0.0005 per 1K tokens
                "output": 0.0015 / 1000    # $0.0015 per 1K tokens
            },
            "gpt-4": {
                "input": 0.03 / 1000,      # $0.03 per 1K tokens
                "output": 0.06 / 1000       # $0.06 per 1K tokens
            },
            "gpt-4-turbo": {
                "input": 0.01 / 1000,      # $0.01 per 1K tokens
                "output": 0.03 / 1000       # $0.03 per 1K tokens
            }
        }
        
        if model not in pricing:
            model = "gpt-3.5-turbo"  # Default
        
        input_cost = input_tokens * pricing[model]["input"]
        output_cost = output_tokens * pricing[model]["output"]
        
        return input_cost + output_cost
    
    def truncate_to_limit(
        self,
        text: str,
        max_tokens: int,
        reserve_for_response: int = 500
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum total tokens (input + output)
            reserve_for_response: Tokens to reserve for response
        
        Returns:
            Truncated text
        """
        available_tokens = max_tokens - reserve_for_response
        
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= available_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:available_tokens]
        return self.encoding.decode(truncated_tokens)

# Usage
token_manager = TokenManager(model="gpt-3.5-turbo")

# Count tokens
text = "How many tokens is this sentence?"
token_count = token_manager.count_tokens(text)
print(f"Text: '{text}'")
print(f"Tokens: {token_count}")

# Count message tokens
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]
message_tokens = token_manager.count_message_tokens(messages)
print(f"\nMessage tokens: {message_tokens}")

# Estimate cost
input_tokens = 100
output_tokens = 200
cost = token_manager.estimate_cost(input_tokens, output_tokens, "gpt-3.5-turbo")
print(f"\nEstimated cost: ${cost:.6f}")

# Truncate long text
long_text = "Very long text..." * 1000
truncated = token_manager.truncate_to_limit(long_text, max_tokens=4096)
print(f"\nOriginal tokens: {token_manager.count_tokens(long_text)}")
print(f"Truncated tokens: {token_manager.count_tokens(truncated)}")
```

### Exercise 4.2: Conversation History Management

```python
class ConversationManager:
    """
    Manage conversation history within token limits.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 4096,
        reserve_for_response: int = 500
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self.token_manager = TokenManager(model)
        self.messages = []
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def get_messages_within_limit(self) -> List[Dict[str, str]]:
        """
        Get messages that fit within token limit.
        
        Returns:
            List of messages within limit
        """
        available_tokens = self.max_tokens - self.reserve_for_response
        
        # Always include system message if present
        system_messages = [m for m in self.messages if m["role"] == "system"]
        other_messages = [m for m in self.messages if m["role"] != "system"]
        
        # Count system message tokens
        system_tokens = self.token_manager.count_message_tokens(system_messages)
        remaining_tokens = available_tokens - system_tokens
        
        # Add messages from most recent until we hit limit
        included_messages = []
        current_tokens = 0
        
        for message in reversed(other_messages):
            message_tokens = self.token_manager.count_message_tokens([message])
            
            if current_tokens + message_tokens <= remaining_tokens:
                included_messages.insert(0, message)
                current_tokens += message_tokens
            else:
                break
        
        return system_messages + included_messages
    
    def summarize_history(self, keep_recent: int = 4) -> str:
        """
        Summarize old conversation history.
        
        Args:
            keep_recent: Number of recent messages to keep unsummarized
        
        Returns:
            Summary of old messages
        """
        if len(self.messages) <= keep_recent + 1:  # +1 for system message
            return ""
        
        # Separate system message and messages to summarize
        system_msg = next((m for m in self.messages if m["role"] == "system"), None)
        to_summarize = self.messages[1:-keep_recent]  # Skip system and recent
        
        if not to_summarize:
            return ""
        
        # Create summary prompt
        conversation_text = "\n".join([
            f"{m['role']}: {m['content']}" for m in to_summarize
        ])
        
        prompt = f"""Summarize this conversation history concisely:

{conversation_text}

Provide a brief summary of the key points and context."""
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        summary = response.choices[0].message.content
        
        # Replace old messages with summary
        recent_messages = self.messages[-keep_recent:]
        self.messages = ([system_msg] if system_msg else []) + [
            {"role": "system", "content": f"Previous conversation summary: {summary}"}
        ] + recent_messages
        
        return summary

# Usage
conv_manager = ConversationManager(max_tokens=1000)

# Add system message
conv_manager.add_message("system", "You are a helpful assistant.")

# Simulate conversation
conv_manager.add_message("user", "What is machine learning?")
conv_manager.add_message("assistant", "Machine learning is a subset of AI...")
conv_manager.add_message("user", "Can you explain supervised learning?")
conv_manager.add_message("assistant", "Supervised learning uses labeled data...")
# ... many more messages ...

# Get messages within limit
messages_to_send = conv_manager.get_messages_within_limit()
print(f"Sending {len(messages_to_send)} messages out of {len(conv_manager.messages)} total")

# Summarize if history is too long
if len(conv_manager.messages) > 20:
    summary = conv_manager.summarize_history(keep_recent=4)
    print(f"Summarized history: {summary}")
```

---

## Part 5: Rate Limiting & Optimization

### Exercise 5.1: Rate Limiter

```python
import time
from collections import deque
from threading import Lock

class RateLimiter:
    """
    Rate limiter for API calls.
    """
    
    def __init__(self, calls_per_minute: int = 60, tokens_per_minute: int = 90000):
        self.calls_per_minute = calls_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        self.call_timestamps = deque()
        self.token_usage = deque()
        
        self.lock = Lock()
    
    def acquire(self, estimated_tokens: int = 1000):
        """
        Acquire permission to make an API call.
        
        Args:
            estimated_tokens: Estimated tokens for this call
        
        Blocks until rate limit allows the call.
        """
        with self.lock:
            current_time = time.time()
            one_minute_ago = current_time - 60
            
            # Remove old timestamps
            while self.call_timestamps and self.call_timestamps[0] < one_minute_ago:
                self.call_timestamps.popleft()
            
            while self.token_usage and self.token_usage[0][0] < one_minute_ago:
                self.token_usage.popleft()
            
            # Check if we're at call limit
            while len(self.call_timestamps) >= self.calls_per_minute:
                sleep_time = self.call_timestamps[0] + 60 - current_time
                if sleep_time > 0:
                    print(f"Rate limit: sleeping {sleep_time:.2f}s (calls)")
                    time.sleep(sleep_time)
                    current_time = time.time()
                self.call_timestamps.popleft()
            
            # Check if we're at token limit
            total_tokens = sum(t[1] for t in self.token_usage)
            while total_tokens + estimated_tokens > self.tokens_per_minute and self.token_usage:
                sleep_time = self.token_usage[0][0] + 60 - current_time
                if sleep_time > 0:
                    print(f"Rate limit: sleeping {sleep_time:.2f}s (tokens)")
                    time.sleep(sleep_time)
                    current_time = time.time()
                _, tokens = self.token_usage.popleft()
                total_tokens -= tokens
            
            # Record this call
            self.call_timestamps.append(current_time)
            self.token_usage.append((current_time, estimated_tokens))
    
    def get_stats(self) -> Dict[str, any]:
        """Get current rate limit statistics."""
        with self.lock:
            current_time = time.time()
            one_minute_ago = current_time - 60
            
            recent_calls = sum(1 for ts in self.call_timestamps if ts > one_minute_ago)
            recent_tokens = sum(t[1] for t in self.token_usage if t[0] > one_minute_ago)
            
            return {
                "calls_in_last_minute": recent_calls,
                "calls_remaining": self.calls_per_minute - recent_calls,
                "tokens_in_last_minute": recent_tokens,
                "tokens_remaining": self.tokens_per_minute - recent_tokens
            }

# Usage
rate_limiter = RateLimiter(calls_per_minute=3, tokens_per_minute=5000)

token_manager = TokenManager()

for i in range(5):
    messages = [{"role": "user", "content": f"Tell me fact number {i+1} about Python."}]
    estimated_tokens = token_manager.count_message_tokens(messages) + 100
    
    print(f"\nRequest {i+1}:")
    print(f"Stats before: {rate_limiter.get_stats()}")
    
    rate_limiter.acquire(estimated_tokens)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    print(f"Response: {response.choices[0].message.content[:100]}...")
```

### Exercise 5.2: Request Batching

```python
class BatchProcessor:
    """
    Batch multiple requests for efficiency.
    """
    
    def __init__(
        self,
        max_batch_size: int = 10,
        max_wait_time: float = 1.0
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.lock = Lock()
    
    def add_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Add a request to the batch.
        
        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())
        
        with self.lock:
            self.pending_requests.append({
                "id": request_id,
                "messages": messages,
                "kwargs": kwargs,
                "result": None,
                "complete": False
            })
        
        return request_id
    
    def process_batch(self):
        """
        Process accumulated requests in batch.
        """
        with self.lock:
            if not self.pending_requests:
                return
            
            batch = self.pending_requests[:self.max_batch_size]
            self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        # Process each request (could be parallelized further)
        for request in batch:
            try:
                response = client.chat.completions.create(
                    messages=request["messages"],
                    **request["kwargs"]
                )
                request["result"] = response.choices[0].message.content
                request["complete"] = True
            except Exception as e:
                request["result"] = f"Error: {e}"
                request["complete"] = True
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[str]:
        """
        Get result for a request ID.
        
        Args:
            request_id: Request ID
            timeout: Max wait time
        
        Returns:
            Result or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if result is ready
            for request in self.pending_requests:
                if request["id"] == request_id and request["complete"]:
                    return request["result"]
            
            time.sleep(0.1)
        
        return None

# Usage would typically be in a background thread/process
```

---

## Part 6: Production API Wrapper

### Complete Production-Ready Client

```python
from dataclasses import dataclass
import logging
from typing import Optional, Dict, List, Callable
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for production API client."""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    max_retries: int = 3
    timeout: int = 30
    enable_caching: bool = True
    enable_rate_limiting: bool = True
    calls_per_minute: int = 60
    tokens_per_minute: int = 90000
    log_requests: bool = True

class ProductionOpenAIClient:
    """
    Production-ready OpenAI API client with all features.
    """
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        
        # Components
        self.token_manager = TokenManager(self.config.model)
        self.rate_limiter = RateLimiter(
            self.config.calls_per_minute,
            self.config.tokens_per_minute
        ) if self.config.enable_rate_limiting else None
        
        self.cache = {} if self.config.enable_caching else None
        self.request_log = []
    
    def completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, any]:
        """
        Make a completion request with all production features.
        
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Merge config with kwargs
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        
        # Check cache
        if self.cache is not None:
            cache_key = self._get_cache_key(messages, params)
            if cache_key in self.cache:
                logger.info("Cache hit")
                return self.cache[cache_key]
        
        # Count tokens and check rate limit
        estimated_tokens = self.token_manager.count_message_tokens(messages)
        if self.rate_limiter:
            self.rate_limiter.acquire(estimated_tokens)
        
        # Make request with retries
        try:
            response = self._make_request_with_retry(messages, params)
            
            result = {
                "success": True,
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "metadata": {
                    "duration": time.time() - start_time,
                    "cached": False
                }
            }
            
            # Cache result
            if self.cache is not None:
                self.cache[cache_key] = result
            
            # Log request
            if self.config.log_requests:
                self._log_request(messages, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {
                "success": False,
                "content": None,
                "error": str(e),
                "metadata": {
                    "duration": time.time() - start_time
                }
            }
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60)
    )
    def _make_request_with_retry(self, messages: List[Dict[str, str]], params: Dict):
        """Make API request with retry logic."""
        return self.client.chat.completions.create(
            messages=messages,
            **params
        )
    
    def _get_cache_key(self, messages: List[Dict[str, str]], params: Dict) -> str:
        """Generate cache key for request."""
        cache_str = json.dumps({
            "messages": messages,
            "params": {k: v for k, v in params.items() if k != "stream"}
        }, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _log_request(self, messages: List[Dict[str, str]], result: Dict):
        """Log request for monitoring."""
        self.request_log.append({
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "success": result["success"],
            "tokens": result.get("usage", {}).get("total_tokens", 0),
            "duration": result["metadata"]["duration"]
        })
    
    def get_statistics(self) -> Dict[str, any]:
        """Get usage statistics."""
        if not self.request_log:
            return {"message": "No requests logged"}
        
        total_requests = len(self.request_log)
        successful = sum(1 for r in self.request_log if r["success"])
        total_tokens = sum(r["tokens"] for r in self.request_log)
        avg_duration = sum(r["duration"] for r in self.request_log) / total_requests
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "success_rate": successful / total_requests,
            "total_tokens": total_tokens,
            "average_duration": avg_duration,
            "cache_size": len(self.cache) if self.cache else 0
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")

# Usage
config = APIConfig(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_retries=3,
    enable_caching=True,
    enable_rate_limiting=True,
    log_requests=True
)

prod_client = ProductionOpenAIClient(config)

# Make requests
messages = [{"role": "user", "content": "What is Python?"}]
result = prod_client.completion(messages)

if result["success"]:
    print(f"Response: {result['content']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")
    print(f"Duration: {result['metadata']['duration']:.2f}s")

# Get statistics
stats = prod_client.get_statistics()
print(f"\nStatistics: {json.dumps(stats, indent=2)}")
```

---

## Best Practices

### 1. Parameter Selection
- Start with defaults, then optimize based on use case
- Use temperature=0.0 for deterministic tasks
- Use frequency_penalty to reduce repetition
- Don't use temperature and top_p together at non-default values

### 2. Error Handling
- Always implement retry logic for transient errors
- Set appropriate timeouts
- Provide fallback responses
- Log errors for monitoring

### 3. Token Management
- Count tokens before making requests
- Truncate inputs to stay within limits
- Summarize long conversation histories
- Monitor token usage for cost control

### 4. Rate Limiting
- Implement client-side rate limiting
- Use exponential backoff for retries
- Batch requests when possible
- Monitor rate limit headers

### 5. Caching
- Cache identical requests
- Use cache invalidation strategies
- Monitor cache hit rates
- Clear cache periodically

### 6. Streaming
- Use streaming for long responses
- Implement proper event handlers
- Handle stream interruptions
- Monitor streaming performance

### 7. Production Readiness
- Comprehensive error handling
- Request logging and monitoring
- Cost tracking
- Performance metrics
- Health checks

---

**End of Lab 2 Solutions**
