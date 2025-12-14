# Lesson 3: Guardrails & Safety Systems

**Duration:** 120 minutes  
**Level:** Intermediate to Advanced  
**Prerequisites:** Understanding of LLM applications, Python async/await

## Table of Contents
- [Introduction to Guardrails](#introduction-to-guardrails)
- [Content Moderation](#content-moderation)
- [PII Detection and Redaction](#pii-detection-and-redaction)
- [Custom Guardrails Framework](#custom-guardrails-framework)
- [Output Validation](#output-validation)
- [Toxicity Detection](#toxicity-detection)
- [Compliance and Regulatory Requirements](#compliance-and-regulatory-requirements)
- [Guardrail Orchestration](#guardrail-orchestration)
- [Performance Optimization](#performance-optimization)
- [Production Patterns](#production-patterns)

---

## Introduction to Guardrails

**Guardrails** are safety mechanisms that validate, filter, and control inputs to and outputs from LLM systems. They act as checkpoints to ensure:

- 🛡️ **Safety**: Prevent harmful content
- 🔒 **Security**: Block malicious inputs
- ⚖️ **Compliance**: Meet regulatory requirements
- ✅ **Quality**: Ensure output correctness
- 🎯 **Reliability**: Enforce business rules

### Why Guardrails Matter

```python
# Without guardrails
user_input = "How do I make a bomb?"
response = llm.complete(user_input)
# ❌ Dangerous content generated
# ❌ Legal liability
# ❌ Reputational damage

# With guardrails
user_input = "How do I make a bomb?"
if safety_guardrail.is_unsafe(user_input):
    response = "I cannot provide information on dangerous activities."
# ✅ Safe, compliant response
```

### Guardrail Layers

```
User Input
    ↓
┌─────────────────────────────────┐
│  Input Guardrails               │
│  - Content moderation           │
│  - PII detection                │
│  - Prompt injection check       │
│  - Input validation             │
└─────────────────────────────────┘
    ↓
LLM Processing
    ↓
┌─────────────────────────────────┐
│  Output Guardrails              │
│  - Content filtering            │
│  - PII redaction                │
│  - Format validation            │
│  - Fact checking                │
└─────────────────────────────────┘
    ↓
User Response
```

---

## Content Moderation

### OpenAI Moderation API

OpenAI provides a free moderation endpoint that classifies content across categories.

```python
from openai import OpenAI

client = OpenAI()

def moderate_content(text: str) -> dict:
    """
    Check content for policy violations.
    
    Categories:
    - hate: Content promoting hate
    - hate/threatening: Hateful content with violence
    - harassment: Harassing language
    - harassment/threatening: Harassment with threats
    - self-harm: Self-harm content
    - self-harm/intent: Self-harm with intent
    - self-harm/instructions: How-to self-harm
    - sexual: Sexual content
    - sexual/minors: Sexual content involving minors
    - violence: Violent content
    - violence/graphic: Graphic violence
    """
    
    response = client.moderations.create(input=text)
    result = response.results[0]
    
    return {
        "flagged": result.flagged,
        "categories": {
            cat: score 
            for cat, score in result.category_scores.items()
            if score > 0.01  # Only show relevant scores
        },
        "highest_category": max(
            result.category_scores.items(),
            key=lambda x: x[1]
        ) if result.flagged else None
    }

# Usage
text = "I want to hurt someone"
moderation_result = moderate_content(text)

if moderation_result["flagged"]:
    print(f"⚠️ Content flagged: {moderation_result['highest_category']}")
    # Take action: block, log, alert
else:
    print("✅ Content is safe")
```

### Custom Content Filter

```python
import re
from typing import List, Dict
from enum import Enum

class ContentCategory(Enum):
    """Content categories for filtering."""
    SAFE = "safe"
    PROFANITY = "profanity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SPAM = "spam"

class ContentFilter:
    """Custom content filtering system."""
    
    def __init__(self):
        # Load blocklists
        self.profanity_patterns = self._load_profanity_patterns()
        self.hate_speech_patterns = self._load_hate_speech_patterns()
        self.violence_patterns = self._load_violence_patterns()
    
    def _load_profanity_patterns(self) -> List[re.Pattern]:
        """Load profanity regex patterns."""
        words = ["damn", "hell", "crap"]  # Extend as needed
        return [
            re.compile(rf'\b{word}\b', re.IGNORECASE) 
            for word in words
        ]
    
    def _load_hate_speech_patterns(self) -> List[re.Pattern]:
        """Load hate speech patterns."""
        # In production, use a comprehensive database
        patterns = [
            r'\b(racist|bigot|discriminat)\w*\b',
            # Add more patterns
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def _load_violence_patterns(self) -> List[re.Pattern]:
        """Load violence-related patterns."""
        patterns = [
            r'\b(kill|murder|attack|assault)\w*\b',
            r'\b(weapon|gun|knife|bomb)\w*\b',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def check(self, text: str) -> Dict:
        """
        Check text against all filters.
        
        Returns:
            {
                "category": ContentCategory,
                "flagged": bool,
                "matches": List[str],
                "confidence": float
            }
        """
        matches = {
            ContentCategory.PROFANITY: [],
            ContentCategory.HATE_SPEECH: [],
            ContentCategory.VIOLENCE: []
        }
        
        # Check profanity
        for pattern in self.profanity_patterns:
            found = pattern.findall(text)
            if found:
                matches[ContentCategory.PROFANITY].extend(found)
        
        # Check hate speech
        for pattern in self.hate_speech_patterns:
            found = pattern.findall(text)
            if found:
                matches[ContentCategory.HATE_SPEECH].extend(found)
        
        # Check violence
        for pattern in self.violence_patterns:
            found = pattern.findall(text)
            if found:
                matches[ContentCategory.VIOLENCE].extend(found)
        
        # Determine most severe category
        if matches[ContentCategory.HATE_SPEECH]:
            category = ContentCategory.HATE_SPEECH
            flagged = True
        elif matches[ContentCategory.VIOLENCE]:
            category = ContentCategory.VIOLENCE
            flagged = True
        elif matches[ContentCategory.PROFANITY]:
            category = ContentCategory.PROFANITY
            flagged = len(matches[ContentCategory.PROFANITY]) > 2  # Allow some
        else:
            category = ContentCategory.SAFE
            flagged = False
        
        return {
            "category": category,
            "flagged": flagged,
            "matches": matches[category] if flagged else [],
            "confidence": len(matches[category]) / max(len(text.split()), 1)
        }

# Usage
filter = ContentFilter()

text = "I want to attack that person"
result = filter.check(text)

if result["flagged"]:
    print(f"⚠️ Flagged as {result['category'].value}")
    print(f"Matches: {result['matches']}")
```

### Perspective API Integration

Google's Perspective API provides ML-based toxicity scores.

```python
from googleapiclient import discovery
import os

class PerspectiveAPI:
    """Google Perspective API for toxicity detection."""
    
    def __init__(self, api_key: str):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
        )
    
    def analyze(self, text: str, attributes: List[str] = None) -> Dict:
        """
        Analyze text for toxicity.
        
        Attributes:
        - TOXICITY: Overall toxicity
        - SEVERE_TOXICITY: Very toxic content
        - IDENTITY_ATTACK: Attacks on identity
        - INSULT: Insulting language
        - PROFANITY: Profane language
        - THREAT: Threatening language
        """
        
        if attributes is None:
            attributes = [
                "TOXICITY",
                "SEVERE_TOXICITY",
                "IDENTITY_ATTACK",
                "INSULT",
                "PROFANITY",
                "THREAT"
            ]
        
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {attr: {} for attr in attributes}
        }
        
        response = self.client.comments().analyze(body=analyze_request).execute()
        
        scores = {}
        for attr in attributes:
            score = response['attributeScores'][attr]['summaryScore']['value']
            scores[attr.lower()] = score
        
        return {
            "scores": scores,
            "flagged": scores.get("toxicity", 0) > 0.7,  # Threshold
            "max_score": max(scores.values()),
            "max_attribute": max(scores.items(), key=lambda x: x[1])[0]
        }

# Usage
perspective = PerspectiveAPI(api_key=os.getenv("PERSPECTIVE_API_KEY"))

text = "You're stupid and worthless"
result = perspective.analyze(text)

if result["flagged"]:
    print(f"⚠️ Toxic content detected")
    print(f"Max score: {result['max_score']:.2f} ({result['max_attribute']})")
```

---

## PII Detection and Redaction

### Microsoft Presidio

Presidio is an open-source PII detection and anonymization framework.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class PIIGuardrail:
    """PII detection and redaction using Presidio."""
    
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
    
    def detect(self, text: str) -> List[Dict]:
        """
        Detect PII in text.
        
        Returns list of detected PII entities:
        - PERSON: Person names
        - EMAIL_ADDRESS: Email addresses
        - PHONE_NUMBER: Phone numbers
        - CREDIT_CARD: Credit card numbers
        - IBAN_CODE: Bank account numbers
        - US_SSN: Social security numbers
        - LOCATION: Addresses, cities
        - DATE_TIME: Dates and times
        - IP_ADDRESS: IP addresses
        - URL: URLs
        """
        
        results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=None  # Detect all entity types
        )
        
        return [
            {
                "type": result.entity_type,
                "text": text[result.start:result.end],
                "start": result.start,
                "end": result.end,
                "score": result.score
            }
            for result in results
        ]
    
    def redact(self, text: str, replacement: str = "[REDACTED]") -> Dict:
        """
        Redact PII from text.
        
        Returns:
            {
                "text": str,           # Redacted text
                "pii_found": List[Dict],  # Detected PII
                "redacted_count": int
            }
        """
        
        # Detect PII
        pii_found = self.detect(text)
        
        if not pii_found:
            return {
                "text": text,
                "pii_found": [],
                "redacted_count": 0
            }
        
        # Convert to Presidio format
        analyzer_results = self.analyzer.analyze(text=text, language='en')
        
        # Anonymize
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": replacement})
            }
        )
        
        return {
            "text": anonymized.text,
            "pii_found": pii_found,
            "redacted_count": len(pii_found)
        }
    
    def redact_with_type_labels(self, text: str) -> Dict:
        """Redact PII with type-specific labels."""
        
        analyzer_results = self.analyzer.analyze(text=text, language='en')
        
        # Custom operators for each type
        operators = {}
        for result in analyzer_results:
            entity_type = result.entity_type
            operators[entity_type] = OperatorConfig(
                "replace",
                {"new_value": f"[{entity_type}]"}
            )
        
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators
        )
        
        return {
            "text": anonymized.text,
            "pii_detected": len(analyzer_results)
        }

# Usage
pii_guardrail = PIIGuardrail()

text = """
My name is John Doe and my email is john.doe@example.com.
You can reach me at 555-123-4567.
My SSN is 123-45-6789.
"""

# Detect PII
detected = pii_guardrail.detect(text)
print("PII Detected:")
for entity in detected:
    print(f"  {entity['type']}: {entity['text']} (confidence: {entity['score']:.2f})")

# Redact with generic label
redacted = pii_guardrail.redact(text)
print(f"\nRedacted: {redacted['text']}")

# Redact with type labels
redacted_typed = pii_guardrail.redact_with_type_labels(text)
print(f"\nRedacted (typed): {redacted_typed['text']}")
```

### Custom PII Patterns

```python
import re
from typing import List, Tuple

class CustomPIIDetector:
    """Custom PII detection for specific use cases."""
    
    def __init__(self):
        self.patterns = {
            "email": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "phone": re.compile(
                r'\b(\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
            ),
            "ssn": re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b'
            ),
            "credit_card": re.compile(
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            ),
            "ip_address": re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            "custom_id": re.compile(
                r'\b[A-Z]{2}\d{8}\b'  # e.g., "AB12345678"
            )
        }
    
    def detect_all(self, text: str) -> Dict[str, List[Tuple[str, int, int]]]:
        """Detect all PII types."""
        
        found = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = []
            for match in pattern.finditer(text):
                matches.append((
                    match.group(),
                    match.start(),
                    match.end()
                ))
            
            if matches:
                found[pii_type] = matches
        
        return found
    
    def redact_all(self, text: str, preserve_format: bool = False) -> str:
        """Redact all PII."""
        
        found = self.detect_all(text)
        redacted = text
        
        # Sort by position (reverse to maintain indices)
        all_matches = []
        for pii_type, matches in found.items():
            for match_text, start, end in matches:
                all_matches.append((start, end, pii_type, match_text))
        
        all_matches.sort(reverse=True)
        
        # Replace each match
        for start, end, pii_type, match_text in all_matches:
            if preserve_format:
                # Preserve length and format
                if pii_type == "email":
                    replacement = "x" * (match_text.index("@")) + "@" + "x" * (len(match_text) - match_text.index("@") - 1)
                elif pii_type == "phone":
                    replacement = re.sub(r'\d', 'X', match_text)
                elif pii_type == "credit_card":
                    parts = match_text.split()
                    replacement = "XXXX " * 3 + match_text.split()[-1]
                else:
                    replacement = "X" * len(match_text)
            else:
                replacement = f"[{pii_type.upper()}]"
            
            redacted = redacted[:start] + replacement + redacted[end:]
        
        return redacted

# Usage
detector = CustomPIIDetector()

text = """
Contact me at john@example.com or call 555-123-4567.
My employee ID is AB12345678.
"""

found = detector.detect_all(text)
print("Found PII:")
for pii_type, matches in found.items():
    print(f"  {pii_type}: {len(matches)} match(es)")

redacted = detector.redact_all(text)
print(f"\nRedacted: {redacted}")
```

---

## Custom Guardrails Framework

### Guardrails AI Library

```python
from guardrails import Guard
from guardrails.validators import ValidLength, ValidRange, TwoWords
from pydantic import BaseModel, Field

# Define output structure with validators
class CustomerResponse(BaseModel):
    """Customer service response schema."""
    
    summary: str = Field(
        description="Brief summary of the response",
        validators=[ValidLength(min=10, max=100)]
    )
    
    sentiment_score: float = Field(
        description="Sentiment score",
        validators=[ValidRange(min=0.0, max=1.0)]
    )
    
    action_items: List[str] = Field(
        description="List of action items",
        validators=[ValidLength(min=1, max=5)]
    )

# Create guard
guard = Guard.from_pydantic(
    output_class=CustomerResponse,
    prompt="Generate a customer service response..."
)

# Use with LLM
raw_output = llm.complete(prompt)

# Validate and parse
validated_output = guard.parse(
    raw_output,
    llm_api=llm.complete,  # For reasks if validation fails
    num_reasks=2
)

if validated_output.validation_passed:
    print("✅ Output validated")
    print(validated_output.validated_output)
else:
    print("❌ Validation failed")
    print(validated_output.error)
```

### Custom Validator Framework

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from enum import Enum

class ValidationResult(Enum):
    """Validation result."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

class Validator(ABC):
    """Base validator class."""
    
    @abstractmethod
    def validate(self, value: Any) -> Dict:
        """
        Validate value.
        
        Returns:
            {
                "result": ValidationResult,
                "message": str,
                "details": Dict
            }
        """
        pass

class LengthValidator(Validator):
    """Validate text length."""
    
    def __init__(self, min_length: int = None, max_length: int = None):
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: str) -> Dict:
        length = len(value)
        
        if self.min_length and length < self.min_length:
            return {
                "result": ValidationResult.FAIL,
                "message": f"Text too short: {length} < {self.min_length}",
                "details": {"length": length, "min": self.min_length}
            }
        
        if self.max_length and length > self.max_length:
            return {
                "result": ValidationResult.FAIL,
                "message": f"Text too long: {length} > {self.max_length}",
                "details": {"length": length, "max": self.max_length}
            }
        
        return {
            "result": ValidationResult.PASS,
            "message": "Length valid",
            "details": {"length": length}
        }

class FormatValidator(Validator):
    """Validate text format."""
    
    def __init__(self, expected_format: str):
        """
        expected_format can be:
        - "json": Valid JSON
        - "email": Valid email
        - "url": Valid URL
        - "regex:<pattern>": Match regex pattern
        """
        self.expected_format = expected_format
    
    def validate(self, value: str) -> Dict:
        if self.expected_format == "json":
            try:
                import json
                json.loads(value)
                return {
                    "result": ValidationResult.PASS,
                    "message": "Valid JSON",
                    "details": {}
                }
            except json.JSONDecodeError as e:
                return {
                    "result": ValidationResult.FAIL,
                    "message": f"Invalid JSON: {e}",
                    "details": {"error": str(e)}
                }
        
        elif self.expected_format == "email":
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(pattern, value):
                return {
                    "result": ValidationResult.PASS,
                    "message": "Valid email",
                    "details": {}
                }
            else:
                return {
                    "result": ValidationResult.FAIL,
                    "message": "Invalid email format",
                    "details": {}
                }
        
        # Add more formats as needed
        
        return {
            "result": ValidationResult.FAIL,
            "message": f"Unknown format: {self.expected_format}",
            "details": {}
        }

class ContentValidator(Validator):
    """Validate content safety."""
    
    def __init__(self, content_filter: ContentFilter):
        self.content_filter = content_filter
    
    def validate(self, value: str) -> Dict:
        result = self.content_filter.check(value)
        
        if result["flagged"]:
            return {
                "result": ValidationResult.FAIL,
                "message": f"Content flagged: {result['category'].value}",
                "details": result
            }
        
        return {
            "result": ValidationResult.PASS,
            "message": "Content safe",
            "details": result
        }

class GuardrailChain:
    """Chain multiple validators."""
    
    def __init__(self, validators: List[Validator]):
        self.validators = validators
    
    def validate(self, value: Any) -> Dict:
        """Run all validators."""
        
        results = []
        
        for validator in self.validators:
            result = validator.validate(value)
            results.append(result)
            
            # Stop on first failure
            if result["result"] == ValidationResult.FAIL:
                return {
                    "passed": False,
                    "failed_at": type(validator).__name__,
                    "result": result,
                    "all_results": results
                }
        
        return {
            "passed": True,
            "all_results": results
        }

# Usage
chain = GuardrailChain([
    LengthValidator(min_length=10, max_length=500),
    FormatValidator("json"),
    ContentValidator(ContentFilter())
])

text = '{"response": "This is a safe, valid response"}'
validation_result = chain.validate(text)

if validation_result["passed"]:
    print("✅ All guardrails passed")
else:
    print(f"❌ Failed at: {validation_result['failed_at']}")
    print(f"Reason: {validation_result['result']['message']}")
```

---

## Output Validation

### Structured Output Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json

class ValidatedResponse(BaseModel):
    """Validated LLM response structure."""
    
    answer: str = Field(..., min_length=1, max_length=1000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)
    metadata: Optional[Dict] = None
    
    @validator('answer')
    def answer_must_not_be_empty(cls, v):
        """Ensure answer is not just whitespace."""
        if not v.strip():
            raise ValueError('Answer cannot be empty')
        return v
    
    @validator('sources')
    def sources_must_be_valid_urls(cls, v):
        """Validate source URLs."""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        for url in v:
            if not url_pattern.match(url):
                raise ValueError(f'Invalid URL: {url}')
        
        return v

class OutputValidator:
    """Validate LLM outputs."""
    
    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema
    
    def validate(self, output: str) -> Dict:
        """
        Validate output against schema.
        
        Returns:
            {
                "valid": bool,
                "parsed": BaseModel or None,
                "errors": List[str]
            }
        """
        try:
            # Try to parse as JSON
            if isinstance(output, str):
                try:
                    data = json.loads(output)
                except json.JSONDecodeError:
                    # Not JSON, treat as plain text
                    data = {"answer": output}
            else:
                data = output
            
            # Validate with Pydantic
            parsed = self.schema(**data)
            
            return {
                "valid": True,
                "parsed": parsed,
                "errors": []
            }
        
        except Exception as e:
            return {
                "valid": False,
                "parsed": None,
                "errors": [str(e)]
            }

# Usage
validator = OutputValidator(ValidatedResponse)

# Valid output
output = json.dumps({
    "answer": "RAG stands for Retrieval-Augmented Generation",
    "confidence": 0.95,
    "sources": ["https://example.com/rag-guide"]
})

result = validator.validate(output)
if result["valid"]:
    print("✅ Output validated")
    print(f"Answer: {result['parsed'].answer}")
else:
    print(f"❌ Validation errors: {result['errors']}")
```

### Fact-Checking Guardrail

```python
class FactCheckGuardrail:
    """Check factual consistency of LLM outputs."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def check_consistency(
        self,
        claim: str,
        context: List[str]
    ) -> Dict:
        """
        Check if claim is consistent with context.
        
        Uses LLM to verify factual consistency.
        """
        
        prompt = f"""
Given the following context, determine if the claim is factually consistent.

Context:
{chr(10).join(f"- {c}" for c in context)}

Claim:
{claim}

Is this claim consistent with the context? Answer with:
- "consistent": Claim is fully supported by context
- "inconsistent": Claim contradicts context
- "unverifiable": Insufficient information

Respond in JSON format:
{{
    "verdict": "consistent|inconsistent|unverifiable",
    "reasoning": "explanation",
    "confidence": 0.0-1.0
}}
"""
        
        response = await self.llm.complete(prompt)
        
        try:
            result = json.loads(response)
            return {
                "is_consistent": result["verdict"] == "consistent",
                "verdict": result["verdict"],
                "reasoning": result["reasoning"],
                "confidence": result["confidence"]
            }
        except:
            return {
                "is_consistent": False,
                "verdict": "error",
                "reasoning": "Failed to parse response",
                "confidence": 0.0
            }
    
    async def verify_claims(
        self,
        text: str,
        context: List[str]
    ) -> Dict:
        """Extract and verify all claims in text."""
        
        # Extract claims
        extract_prompt = f"""
Extract all factual claims from the following text.
Return as JSON array of claims.

Text: {text}

Format: {{"claims": ["claim1", "claim2", ...]}}
"""
        
        response = await self.llm.complete(extract_prompt)
        claims_data = json.loads(response)
        claims = claims_data.get("claims", [])
        
        # Verify each claim
        results = []
        for claim in claims:
            verification = await self.check_consistency(claim, context)
            results.append({
                "claim": claim,
                **verification
            })
        
        # Calculate overall consistency score
        consistent_count = sum(1 for r in results if r["is_consistent"])
        consistency_score = consistent_count / len(results) if results else 1.0
        
        return {
            "claims": results,
            "consistency_score": consistency_score,
            "all_consistent": consistency_score == 1.0
        }

# Usage
fact_checker = FactCheckGuardrail(llm_client)

context = [
    "RAG combines retrieval and generation",
    "It was introduced in 2020",
    "It improves factual accuracy"
]

claim = "RAG is a technique from 2020 that improves accuracy"

result = await fact_checker.check_consistency(claim, context)

if result["is_consistent"]:
    print(f"✅ Claim is consistent (confidence: {result['confidence']:.2f})")
else:
    print(f"❌ Claim is {result['verdict']}")
    print(f"Reasoning: {result['reasoning']}")
```

---

## Toxicity Detection

### Transformer-Based Toxicity Detection

```python
from transformers import pipeline

class ToxicityDetector:
    """Detect toxic content using HuggingFace models."""
    
    def __init__(self, model_name: str = "unitary/toxic-bert"):
        """
        Initialize toxicity detector.
        
        Popular models:
        - unitary/toxic-bert
        - martin-ha/toxic-comment-model
        - s-nlp/roberta_toxicity_classifier
        """
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None  # Return all labels
        )
    
    def detect(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Detect toxicity in text.
        
        Returns:
            {
                "is_toxic": bool,
                "scores": Dict[str, float],
                "highest_category": str,
                "highest_score": float
            }
        """
        
        results = self.classifier(text)[0]
        
        # Parse results
        scores = {r["label"]: r["score"] for r in results}
        
        # Find highest scoring category
        highest = max(results, key=lambda x: x["score"])
        
        return {
            "is_toxic": highest["score"] > threshold,
            "scores": scores,
            "highest_category": highest["label"],
            "highest_score": highest["score"]
        }
    
    def detect_batch(
        self,
        texts: List[str],
        threshold: float = 0.5
    ) -> List[Dict]:
        """Detect toxicity in batch."""
        
        results = self.classifier(texts)
        
        outputs = []
        for text_results in results:
            scores = {r["label"]: r["score"] for r in text_results}
            highest = max(text_results, key=lambda x: x["score"])
            
            outputs.append({
                "is_toxic": highest["score"] > threshold,
                "scores": scores,
                "highest_category": highest["label"],
                "highest_score": highest["score"]
            })
        
        return outputs

# Usage
detector = ToxicityDetector()

texts = [
    "You're a wonderful person!",
    "I hate you and want to hurt you",
    "This product is terrible"
]

for text in texts:
    result = detector.detect(text)
    status = "🚫 Toxic" if result["is_toxic"] else "✅ Safe"
    print(f"{status}: {text}")
    print(f"  Highest: {result['highest_category']} ({result['highest_score']:.2f})")
```

---

## Compliance and Regulatory Requirements

### GDPR Compliance

```python
class GDPRGuardrail:
    """GDPR compliance guardrail."""
    
    def __init__(self):
        self.pii_detector = PIIGuardrail()
        self.consent_registry = {}  # In production, use database
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for data processing."""
        
        user_consents = self.consent_registry.get(user_id, {})
        return user_consents.get(purpose, False)
    
    def record_consent(
        self,
        user_id: str,
        purpose: str,
        consented: bool
    ):
        """Record user consent."""
        
        if user_id not in self.consent_registry:
            self.consent_registry[user_id] = {}
        
        self.consent_registry[user_id][purpose] = consented
    
    def validate_request(
        self,
        user_id: str,
        text: str,
        purpose: str
    ) -> Dict:
        """
        Validate GDPR compliance for request.
        
        Checks:
        1. User has given consent
        2. PII is detected and can be processed
        3. Purpose is valid
        """
        
        # Check consent
        has_consent = self.check_consent(user_id, purpose)
        
        if not has_consent:
            return {
                "compliant": False,
                "reason": "no_consent",
                "message": f"User has not consented to {purpose}"
            }
        
        # Detect PII
        pii_found = self.pii_detector.detect(text)
        
        if pii_found:
            # Check if PII processing is allowed
            pii_consent = self.check_consent(user_id, "pii_processing")
            
            if not pii_consent:
                return {
                    "compliant": False,
                    "reason": "pii_no_consent",
                    "message": "PII detected but no consent for processing",
                    "pii_found": pii_found
                }
        
        return {
            "compliant": True,
            "pii_found": pii_found,
            "consents": self.consent_registry.get(user_id, {})
        }

# Usage
gdpr = GDPRGuardrail()

# Record consent
gdpr.record_consent("user_123", "chatbot", True)
gdpr.record_consent("user_123", "pii_processing", True)

# Validate request
text = "My email is john@example.com"
result = gdpr.validate_request("user_123", text, "chatbot")

if result["compliant"]:
    print("✅ GDPR compliant")
else:
    print(f"❌ Not compliant: {result['message']}")
```

### Data Retention Policy

```python
from datetime import datetime, timedelta

class RetentionPolicy:
    """Implement data retention policies."""
    
    def __init__(self):
        self.policies = {
            "chat_logs": timedelta(days=90),
            "pii_data": timedelta(days=30),
            "analytics": timedelta(days=365),
            "audit_logs": timedelta(days=2555)  # 7 years
        }
        self.storage = {}  # In production, use database
    
    def store_data(
        self,
        data_type: str,
        data_id: str,
        data: Any
    ):
        """Store data with retention policy."""
        
        retention_period = self.policies.get(
            data_type,
            timedelta(days=30)  # Default: 30 days
        )
        
        expiry_date = datetime.utcnow() + retention_period
        
        if data_type not in self.storage:
            self.storage[data_type] = {}
        
        self.storage[data_type][data_id] = {
            "data": data,
            "stored_at": datetime.utcnow(),
            "expires_at": expiry_date
        }
    
    def cleanup_expired(self):
        """Remove expired data."""
        
        now = datetime.utcnow()
        deleted_count = 0
        
        for data_type in self.storage:
            expired_ids = [
                data_id
                for data_id, entry in self.storage[data_type].items()
                if entry["expires_at"] < now
            ]
            
            for data_id in expired_ids:
                del self.storage[data_type][data_id]
                deleted_count += 1
        
        return deleted_count
    
    def get_expiry_report(self) -> Dict:
        """Generate expiry report."""
        
        now = datetime.utcnow()
        report = {}
        
        for data_type in self.storage:
            total = len(self.storage[data_type])
            expiring_soon = sum(
                1 for entry in self.storage[data_type].values()
                if entry["expires_at"] < now + timedelta(days=7)
            )
            
            report[data_type] = {
                "total_records": total,
                "expiring_within_7_days": expiring_soon
            }
        
        return report
```

---

## Guardrail Orchestration

### Complete Guardrail Pipeline

```python
class GuardrailPipeline:
    """Orchestrate multiple guardrails."""
    
    def __init__(self):
        self.input_guardrails = []
        self.output_guardrails = []
        self.audit_log = []
    
    def add_input_guardrail(self, guardrail: Validator):
        """Add input guardrail."""
        self.input_guardrails.append(guardrail)
    
    def add_output_guardrail(self, guardrail: Validator):
        """Add output guardrail."""
        self.output_guardrails.append(guardrail)
    
    async def process_input(self, user_input: str, user_id: str) -> Dict:
        """Process and validate user input."""
        
        results = {
            "passed": True,
            "original_input": user_input,
            "processed_input": user_input,
            "guardrail_results": [],
            "blocked": False,
            "block_reason": None
        }
        
        for guardrail in self.input_guardrails:
            result = guardrail.validate(results["processed_input"])
            results["guardrail_results"].append({
                "guardrail": type(guardrail).__name__,
                "result": result
            })
            
            # Check if blocked
            if result["result"] == ValidationResult.FAIL:
                results["passed"] = False
                results["blocked"] = True
                results["block_reason"] = result["message"]
                break
            
            # Apply transformations (e.g., PII redaction)
            if hasattr(guardrail, "transform"):
                results["processed_input"] = guardrail.transform(
                    results["processed_input"]
                )
        
        # Audit log
        self.audit_log.append({
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "type": "input",
            "blocked": results["blocked"],
            "reason": results["block_reason"]
        })
        
        return results
    
    async def process_output(self, llm_output: str, user_id: str) -> Dict:
        """Process and validate LLM output."""
        
        results = {
            "passed": True,
            "original_output": llm_output,
            "processed_output": llm_output,
            "guardrail_results": [],
            "blocked": False,
            "block_reason": None
        }
        
        for guardrail in self.output_guardrails:
            result = guardrail.validate(results["processed_output"])
            results["guardrail_results"].append({
                "guardrail": type(guardrail).__name__,
                "result": result
            })
            
            if result["result"] == ValidationResult.FAIL:
                results["passed"] = False
                results["blocked"] = True
                results["block_reason"] = result["message"]
                break
            
            if hasattr(guardrail, "transform"):
                results["processed_output"] = guardrail.transform(
                    results["processed_output"]
                )
        
        # Audit log
        self.audit_log.append({
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "type": "output",
            "blocked": results["blocked"],
            "reason": results["block_reason"]
        })
        
        return results
    
    async def execute_with_guardrails(
        self,
        user_input: str,
        user_id: str,
        llm_function: Callable
    ) -> Dict:
        """Execute complete pipeline with guardrails."""
        
        # Input guardrails
        input_result = await self.process_input(user_input, user_id)
        
        if input_result["blocked"]:
            return {
                "success": False,
                "stage": "input",
                "message": input_result["block_reason"],
                "output": None
            }
        
        # Execute LLM
        try:
            llm_output = await llm_function(input_result["processed_input"])
        except Exception as e:
            return {
                "success": False,
                "stage": "llm",
                "message": str(e),
                "output": None
            }
        
        # Output guardrails
        output_result = await self.process_output(llm_output, user_id)
        
        if output_result["blocked"]:
            return {
                "success": False,
                "stage": "output",
                "message": output_result["block_reason"],
                "output": None
            }
        
        return {
            "success": True,
            "output": output_result["processed_output"],
            "input_guardrails": input_result["guardrail_results"],
            "output_guardrails": output_result["guardrail_results"]
        }

# Usage
pipeline = GuardrailPipeline()

# Add input guardrails
pipeline.add_input_guardrail(ContentValidator(ContentFilter()))
pipeline.add_input_guardrail(LengthValidator(max_length=1000))

# Add output guardrails
pipeline.add_output_guardrail(PIIRedactionGuardrail())
pipeline.add_output_guardrail(ToxicityGuardrail())

# Execute
async def my_llm_function(prompt):
    return await llm.complete(prompt)

result = await pipeline.execute_with_guardrails(
    user_input="What is machine learning?",
    user_id="user_123",
    llm_function=my_llm_function
)

if result["success"]:
    print(f"✅ Response: {result['output']}")
else:
    print(f"❌ Blocked at {result['stage']}: {result['message']}")
```

---

## Performance Optimization

### Caching Guardrail Results

```python
import hashlib
from functools import lru_cache

class CachedGuardrail:
    """Cache guardrail results for performance."""
    
    def __init__(self, guardrail: Validator, cache_size: int = 1000):
        self.guardrail = guardrail
        self.cache = {}
        self.cache_size = cache_size
    
    def _hash(self, text: str) -> str:
        """Generate cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def validate(self, text: str) -> Dict:
        """Validate with caching."""
        
        cache_key = self._hash(text)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Run validation
        result = self.guardrail.validate(text)
        
        # Store in cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = result
        
        return result
```

### Parallel Guardrail Execution

```python
import asyncio

class ParallelGuardrailPipeline:
    """Execute guardrails in parallel."""
    
    def __init__(self, guardrails: List[Validator]):
        self.guardrails = guardrails
    
    async def validate_parallel(self, text: str) -> Dict:
        """Run all guardrails in parallel."""
        
        # Create tasks
        tasks = [
            asyncio.create_task(self._async_validate(g, text))
            for g in self.guardrails
        ]
        
        # Wait for all
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for failures
        failed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({
                    "guardrail": type(self.guardrails[i]).__name__,
                    "error": str(result)
                })
            elif result["result"] == ValidationResult.FAIL:
                failed.append(result)
        
        return {
            "passed": len(failed) == 0,
            "results": results,
            "failed": failed
        }
    
    async def _async_validate(self, guardrail: Validator, text: str) -> Dict:
        """Async wrapper for validation."""
        return guardrail.validate(text)
```

---

## Production Patterns

### Circuit Breaker for External Services

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker for guardrail services."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        recovery_timeout: int = 30
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.opened_at = None
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker."""
        
        # Check if circuit should transition
        self._check_state_transition()
        
        # If open, reject immediately
        if self.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _check_state_transition(self):
        """Check if circuit should change state."""
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.opened_at:
                elapsed = (datetime.now() - self.opened_at).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
    
    def _on_success(self):
        """Handle successful call."""
        
        if self.state == CircuitState.HALF_OPEN:
            # Recovered, close circuit
            self.state = CircuitState.CLOSED
        
        self.failure_count = 0
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed call."""
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = datetime.now()

# Usage
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

def call_external_guardrail(text):
    try:
        return breaker.call(external_api.moderate, text)
    except Exception as e:
        # Fallback to local guardrail
        return local_filter.check(text)
```

---

## Summary

### Key Takeaways

1. **Guardrails** are essential for safe, compliant LLM systems
2. **Multi-layer approach**: Input and output guardrails
3. **PII protection**: Detect and redact sensitive information
4. **Content moderation**: Block harmful content
5. **Compliance**: Meet regulatory requirements (GDPR, etc.)
6. **Performance**: Cache, parallelize, use circuit breakers

### Checklist

- [ ] Implement content moderation (OpenAI Moderation API)
- [ ] Add PII detection and redaction (Presidio)
- [ ] Create custom guardrails for business rules
- [ ] Set up output validation
- [ ] Add toxicity detection
- [ ] Implement GDPR compliance
- [ ] Build guardrail pipeline
- [ ] Optimize performance (caching, parallel)
- [ ] Add audit logging
- [ ] Test with edge cases

### Next Steps

In Lesson 4, we'll cover **Prompt Security & Attack Mitigation**, including:
- Prompt injection attacks
- Jailbreaking techniques
- Defense strategies
- Red team exercises
- Security testing frameworks

---

## Additional Resources

- [Guardrails AI Documentation](https://docs.guardrailsai.com/)
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
- [GDPR Compliance Guide](https://gdpr.eu/)
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
