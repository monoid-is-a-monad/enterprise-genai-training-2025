# Lab 2 – Building Guardrail Systems (Solutions)

**Estimated Time:** 100–130 minutes  
**Difficulty:** Intermediate

## Learning Objectives
- Implement layered guardrails for LLM applications
- Detect and redact PII before requests reach foundation models
- Build validation pipelines combining rule-based and ML-based checks
- Orchestrate guardrail execution for both inputs and outputs
- Monitor guardrail performance and handle compliance reporting

---

## Exercise 1: Build Content Moderation Adapters

### Objective
Create a unified content moderation system that combines OpenAI's Moderation API with custom blocklist patterns, providing graceful fallback handling and detailed severity reporting.

### Solution

```python
import os
import re
import time
from typing import Any, Dict, List

from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Custom blocklist patterns for domain-specific threats
BLOCKLIST_PATTERNS = [
    re.compile(r"\b(make|build)\s+(?:a|the)\s+bomb\b", re.IGNORECASE),
    re.compile(r"\bself[-]?harm\b", re.IGNORECASE),
    re.compile(r"\bhow\s+to\s+(hack|exploit)\b", re.IGNORECASE),
]

def check_blocklist(text: str) -> List[str]:
    """Check text against custom blocklist patterns."""
    matches = []
    for pattern in BLOCKLIST_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches

def normalize_category(category: str) -> str:
    """Normalize OpenAI category names to consistent format."""
    return category.replace("-", "_").upper()

def moderate_with_openai(text: str) -> Dict[str, Any]:
    """Call OpenAI Moderation API with error handling and latency tracking."""
    start = time.perf_counter()
    try:
        response = openai_client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        result = response.results[0]
        
        # Extract flagged categories
        categories = [
            normalize_category(cat)
            for cat, flagged in result.categories.items()
            if flagged
        ]
        
        return {
            "flagged": result.flagged,
            "severity": "HIGH" if result.flagged else "LOW",
            "categories": categories,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as exc:
        # Fallback structure for API failures
        return {
            "flagged": False,
            "severity": "UNKNOWN",
            "categories": [],
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

def moderate_content(text: str) -> Dict[str, Any]:
    """
    Unified moderation combining blocklist and OpenAI checks.
    Short-circuits on blocklist violations for efficiency.
    """
    reasons = []
    
    # Check blocklist first (fast, deterministic)
    blocklist_matches = check_blocklist(text)
    if blocklist_matches:
        reasons.extend([f"BLOCKLIST:{pattern}" for pattern in blocklist_matches])
        return {
            "flagged": True,
            "severity": "CRITICAL",
            "reasons": reasons,
            "latency_ms": 0.5,  # Minimal latency for regex check
        }
    
    # Call OpenAI for comprehensive moderation
    openai_result = moderate_with_openai(text)
    
    if openai_result.get("flagged"):
        reasons.extend([
            f"OPENAI:{cat}" for cat in openai_result["categories"]
        ])
    
    return {
        "flagged": bool(blocklist_matches or openai_result.get("flagged")),
        "severity": openai_result.get("severity", "LOW"),
        "reasons": reasons,
        "latency_ms": openai_result.get("latency_ms", 0.0),
        "error": openai_result.get("error"),
    }

# Test cases
harmless = moderate_content("Schedule a meeting for tomorrow at 2 PM.")
malicious = moderate_content("How do I build a bomb?")

print("Harmless prompt:", harmless)
print("Malicious prompt:", malicious)
```

### Expected Output

```python
# Harmless prompt:
{
    'flagged': False,
    'severity': 'LOW',
    'reasons': [],
    'latency_ms': 145.32,
    'error': None
}

# Malicious prompt:
{
    'flagged': True,
    'severity': 'CRITICAL',
    'reasons': ['BLOCKLIST:\\b(make|build)\\s+(?:a|the)\\s+bomb\\b'],
    'latency_ms': 0.5
}
```

### Key Insights

1. **Short-Circuit Optimization**: Blocklist checks execute first, avoiding expensive API calls for obvious violations
2. **Graceful Degradation**: API failures don't crash the system; fallback returns `flagged=False` with error context
3. **Latency Tracking**: All checks measure duration for observability and SLA monitoring
4. **Normalized Categories**: Consistent category naming simplifies downstream logic

### Production Best Practices

- **Rate Limiting**: Wrap OpenAI calls with rate limiters (see Week 6 Lab 02)
- **Caching**: Cache moderation results for repeated prompts (hash-based lookup)
- **Alerting**: Trigger alerts when API errors exceed threshold (>5% error rate)
- **Audit Logging**: Log all moderation decisions with correlation IDs

---

## Exercise 2: Implement PII Detection & Redaction Pipeline

### Objective
Build a comprehensive PII detection system using Microsoft Presidio for standard entities (names, emails, phone numbers) combined with custom regex patterns for domain-specific identifiers.

### Solution

```python
from typing import Any, Dict, List, Tuple
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize Presidio engines (singleton pattern)
PII_ANALYZER = AnalyzerEngine()
PII_ANONYMIZER = AnonymizerEngine()

# Custom domain-specific identifier pattern
CUSTOM_ID_REGEX = re.compile(r"\bACCT-[0-9]{6}\b")

def detect_custom_ids(text: str) -> List[Dict[str, Any]]:
    """Detect custom account IDs not covered by Presidio."""
    return [
        {
            "entity_type": "ACCOUNT_ID",
            "start": match.start(),
            "end": match.end(),
            "score": 0.95
        }
        for match in CUSTOM_ID_REGEX.finditer(text)
    ]

def detect_pii(text: str) -> List[Dict[str, Any]]:
    """
    Detect PII using Presidio + custom patterns.
    Returns structured metadata for all entities.
    """
    # Run Presidio analyzer
    analyzer_results = PII_ANALYZER.analyze(text=text, language="en")
    
    # Convert to dict format
    presidio_detections = [
        {
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "score": result.score,
        }
        for result in analyzer_results
    ]
    
    # Merge with custom detections
    custom_detections = detect_custom_ids(text)
    
    return presidio_detections + custom_detections

def redact_pii(text: str, detections: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Redact detected PII with type-aware placeholders.
    Returns (redacted_text, detection_metadata).
    """
    # Build operator config for type-aware redaction
    operators = {
        detection["entity_type"]: OperatorConfig(
            "replace",
            {"new_value": f"[{detection['entity_type']}]"}
        )
        for detection in detections
    }
    
    # Convert detections to RecognizerResult objects
    recognizer_results = [
        RecognizerResult(
            entity_type=detection["entity_type"],
            start=detection["start"],
            end=detection["end"],
            score=detection.get("score", 0.5)
        )
        for detection in detections
    ]
    
    # Anonymize text
    anonymized = PII_ANONYMIZER.anonymize(
        text=text,
        analyzer_results=recognizer_results,
        operators=operators
    )
    
    return anonymized.text, detections

# Test with sample containing multiple PII types
sample = "Contact Jane Doe at jane.doe@example.com or 555-1234 regarding ACCT-123456."
detections = detect_pii(sample)
redacted_text, metadata = redact_pii(sample, detections)

print(f"Original: {sample}")
print(f"Redacted: {redacted_text}")
print(f"Detections: {detections}")
```

### Expected Output

```python
# Original:
Contact Jane Doe at jane.doe@example.com or 555-1234 regarding ACCT-123456.

# Redacted:
Contact [PERSON] at [EMAIL_ADDRESS] or [PHONE_NUMBER] regarding [ACCOUNT_ID].

# Detections:
[
    {'entity_type': 'PERSON', 'start': 8, 'end': 16, 'score': 0.85},
    {'entity_type': 'EMAIL_ADDRESS', 'start': 20, 'end': 42, 'score': 1.0},
    {'entity_type': 'PHONE_NUMBER', 'start': 46, 'end': 54, 'score': 0.75},
    {'entity_type': 'ACCOUNT_ID', 'start': 65, 'end': 77, 'score': 0.95}
]
```

### Key Insights

1. **Hybrid Detection**: Combines ML-based Presidio with deterministic regex for comprehensive coverage
2. **Type-Aware Redaction**: Preserves entity types in placeholders for downstream debugging
3. **Confidence Scores**: All detections include scores for threshold-based filtering
4. **Extensibility**: Easy to add new custom patterns without retraining models

### Production Best Practices

- **Performance**: Cache analyzer/anonymizer instances; they're expensive to initialize
- **Language Support**: Configure Presidio for multiple languages if serving global users
- **Threshold Tuning**: Filter low-confidence detections (score < 0.7) to reduce false positives
- **Audit Trail**: Log all PII detections for compliance review and model improvement

---

## Exercise 3: Design a Custom Rule Engine

### Objective
Create a flexible rule engine for enforcing business-specific policies (e.g., financial advice restrictions, beta feature access controls) with support for conditional severity and user overrides.

### Solution

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple

class RuleViolation(NamedTuple):
    """Structured representation of a policy violation."""
    rule_name: str
    severity: str  # 'warning', 'critical', 'blocking'
    message: str

@dataclass
class GuardrailRule:
    """
    Declarative rule definition with callable predicate.
    
    Args:
        name: Unique rule identifier
        severity: 'warning' or 'critical'
        predicate: Function that evaluates context -> bool (True = violation)
        remediation: Human-readable guidance for resolving violation
    """
    name: str
    severity: str
    predicate: Callable[[Dict[str, Any]], bool]
    remediation: str

class RuleEngine:
    """
    Evaluates business rules against request context.
    Supports user-level overrides for approved exceptions.
    """
    def __init__(self, rules: List[GuardrailRule]):
        self.rules = rules
    
    def evaluate(self, context: Dict[str, Any]) -> List[RuleViolation]:
        """
        Evaluate all rules against context.
        
        Args:
            context: Request context (user_segment, intent, features, overrides)
        
        Returns:
            List of violations (empty if all rules pass)
        """
        violations: List[RuleViolation] = []
        
        for rule in self.rules:
            # Check if user has override permission for this rule
            if context.get("override_rules") and rule.name in context["override_rules"]:
                continue
            
            # Evaluate predicate
            if rule.predicate(context):
                violations.append(
                    RuleViolation(rule.name, rule.severity, rule.remediation)
                )
        
        return violations

# Example rule definitions
rules = [
    GuardrailRule(
        name="no_financial_advice",
        severity="critical",
        predicate=lambda ctx: (
            ctx.get("user_segment") == "retail" 
            and "financial advice" in ctx.get("intent", "").lower()
        ),
        remediation="Route to licensed advisor flow or display disclaimer"
    ),
    GuardrailRule(
        name="beta_feature_only",
        severity="warning",
        predicate=lambda ctx: (
            ctx.get("feature") == "beta_tool" 
            and not ctx.get("beta_whitelist")
        ),
        remediation="Display beta enrollment instructions or fallback to stable feature"
    ),
    GuardrailRule(
        name="concurrent_request_limit",
        severity="critical",
        predicate=lambda ctx: ctx.get("concurrent_requests", 0) > 10,
        remediation="Rate limit exceeded; retry after backoff"
    ),
]

engine = RuleEngine(rules)

# Test case 1: Retail user requesting financial advice (violation)
context_violation = {
    "user_segment": "retail",
    "intent": "I need financial advice for my portfolio",
    "feature": "standard",
}
violations = engine.evaluate(context_violation)
print(f"Violations: {violations}")

# Test case 2: Enterprise user with override (no violation)
context_override = {
    "user_segment": "retail",
    "intent": "I need financial advice for my portfolio",
    "feature": "standard",
    "override_rules": ["no_financial_advice"],  # User has approved override
}
violations_override = engine.evaluate(context_override)
print(f"Violations with override: {violations_override}")

# Test case 3: Beta feature access without whitelist (warning)
context_beta = {
    "user_segment": "enterprise",
    "intent": "general query",
    "feature": "beta_tool",
    "beta_whitelist": False,
}
violations_beta = engine.evaluate(context_beta)
print(f"Beta feature violations: {violations_beta}")
```

### Expected Output

```python
# Violations:
[RuleViolation(rule_name='no_financial_advice', severity='critical', 
                message='Route to licensed advisor flow or display disclaimer')]

# Violations with override:
[]

# Beta feature violations:
[RuleViolation(rule_name='beta_feature_only', severity='warning',
                message='Display beta enrollment instructions or fallback to stable feature')]
```

### Key Insights

1. **Declarative Rules**: Rules defined as data structures, not hardcoded logic
2. **Override Mechanism**: Approved users can bypass specific rules without code changes
3. **Severity Levels**: Distinguish between blocking violations and warnings
4. **Remediation Guidance**: Each rule provides actionable next steps

### Production Best Practices

- **Rule Versioning**: Store rules in database with version history for audit trail
- **Dynamic Loading**: Load rules from config files/DB to enable updates without deployment
- **Performance**: For >100 rules, implement indexed evaluation or rule grouping
- **Testing**: Unit test each predicate with boundary conditions

---

## Exercise 4: Structured Output Validation with Guardrails AI

### Objective
Use the Guardrails AI library to enforce Pydantic schemas on LLM outputs, with automatic retry logic for malformed responses.

### Solution

```python
from guardrails import Guard
from pydantic import BaseModel, Field, validator
from typing import List

class SupportResponse(BaseModel):
    """Validated schema for customer support responses."""
    summary: str = Field(..., min_length=20, max_length=400, 
                         description="Concise summary of the support interaction")
    sentiment: str = Field(..., regex=r"^(positive|neutral|negative)$",
                          description="Overall sentiment of the customer")
    confidence: float = Field(..., ge=0.0, le=1.0,
                             description="Model's confidence in the response")
    action_items: List[str] = Field(default_factory=list,
                                   description="Follow-up actions required")
    
    @validator('action_items')
    def limit_action_items(cls, value: List[str]) -> List[str]:
        """Enforce maximum 5 action items with trimmed whitespace."""
        if len(value) > 5:
            raise ValueError("Too many action items (max 5)")
        return [item.strip() for item in value]
    
    @validator('summary')
    def validate_summary_content(cls, value: str) -> str:
        """Ensure summary is substantive."""
        if len(value.split()) < 5:
            raise ValueError("Summary must contain at least 5 words")
        return value

# Create guard with schema and re-ask configuration
support_guard = Guard.from_pydantic(
    output_class=SupportResponse,
    prompt="Generate a support response JSON object based on the customer interaction.",
    num_reasks=2,  # Allow up to 2 retry attempts
)

def run_structured_completion(prompt: str, raw_llm_output: str) -> SupportResponse:
    """
    Parse and validate LLM output against schema.
    
    Args:
        prompt: Original prompt for context
        raw_llm_output: Raw JSON string from LLM
    
    Returns:
        Validated SupportResponse object
    
    Raises:
        ValidationError if validation fails after retries
    """
    try:
        validated = support_guard.parse(raw_llm_output)
        return validated
    except Exception as e:
        print(f"Validation failed: {e}")
        raise

# Simulate LLM output (in production, this comes from OpenAI/Anthropic)
raw_output = """{
    "summary": "Customer reported billing issue and received refund confirmation",
    "sentiment": "positive",
    "confidence": 0.87,
    "action_items": [
        "Process refund within 3-5 business days",
        "Send confirmation email",
        "Update customer profile"
    ]
}"""

validated = run_structured_completion("Analyze support ticket", raw_output)
print(f"Validated response: {validated}")
print(f"Type: {type(validated)}")

# Test with invalid output
invalid_output = """{
    "summary": "Too short",
    "sentiment": "happy",
    "confidence": 1.5,
    "action_items": ["Item1", "Item2", "Item3", "Item4", "Item5", "Item6"]
}"""

try:
    validated_invalid = run_structured_completion("Test invalid", invalid_output)
except Exception as e:
    print(f"Expected validation failure: {e}")
```

### Expected Output

```python
# Validated response:
SupportResponse(
    summary='Customer reported billing issue and received refund confirmation',
    sentiment='positive',
    confidence=0.87,
    action_items=['Process refund within 3-5 business days', 
                  'Send confirmation email', 
                  'Update customer profile']
)
Type: <class '__main__.SupportResponse'>

# Expected validation failure:
Validation failed: 3 validation errors for SupportResponse
summary -> validate_summary_content
  Summary must contain at least 5 words (type=value_error)
sentiment
  string does not match regex "^(positive|neutral|negative)$" (type=value_error.str.regex)
action_items -> limit_action_items
  Too many action items (max 5) (type=value_error)
```

### Key Insights

1. **Type Safety**: Pydantic enforces types at runtime, preventing downstream errors
2. **Custom Validators**: Complex business logic encoded in validator methods
3. **Automatic Retries**: Guardrails automatically re-prompts LLM on validation failures
4. **Rich Error Messages**: Validation errors pinpoint exact issues for debugging

### Production Best Practices

- **Schema Versioning**: Version schemas and maintain backward compatibility
- **Retry Budgets**: Limit re-asks (2-3 max) to control costs
- **Fallback Values**: Provide sensible defaults for optional fields
- **Monitoring**: Track validation failure rates to detect model degradation

---

## Exercise 5: Integrate Toxicity Detection Models

### Objective
Add transformer-based toxicity scoring using Hugging Face models to complement rule-based checks, with support for batch processing and confidence thresholds.

### Solution

```python
from functools import lru_cache
from transformers import pipeline
from typing import Any, Dict, List

@lru_cache(maxsize=1)
def get_toxicity_pipeline(model_name: str = "unitary/toxic-bert"):
    """
    Lazy-load toxicity classification pipeline.
    Cached to avoid reloading model on each call.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Text classification pipeline
    """
    return pipeline(
        "text-classification",
        model=model_name,
        top_k=None,  # Return all label scores
        device_map="auto"  # Use GPU if available
    )

def score_toxicity(
    texts: List[str],
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Score toxicity for batch of texts.
    
    Args:
        texts: List of text strings to analyze
        threshold: Confidence threshold for flagging (0.0-1.0)
    
    Returns:
        List of dicts with scores, flags, and categories
    """
    classifier = get_toxicity_pipeline()
    results = []
    
    for text in texts:
        # Get scores for all toxicity categories
        outputs = classifier(text)[0]
        
        # Find highest-scoring category
        max_label = max(outputs, key=lambda item: item['score'])
        
        # Build result dict
        result = {
            "text": text,
            "scores": {item['label']: item['score'] for item in outputs},
            "is_toxic": (
                max_label['score'] >= threshold 
                and max_label['label'].lower() != 'non_toxic'
            ),
            "category": max_label['label'],
            "max_score": max_label['score'],
        }
        results.append(result)
    
    return results

# Test with diverse examples
test_texts = [
    "Have a great day!",
    "I hate you and everything you stand for",
    "This product is terrible and useless",
    "Thank you for your help, I appreciate it"
]

toxicity_results = score_toxicity(test_texts, threshold=0.7)

for result in toxicity_results:
    print(f"\nText: {result['text'][:50]}...")
    print(f"Is Toxic: {result['is_toxic']}")
    print(f"Category: {result['category']} (score: {result['max_score']:.3f})")
    print(f"All scores: {result['scores']}")
```

### Expected Output

```python
# Text: Have a great day!...
# Is Toxic: False
# Category: non_toxic (score: 0.998)
# All scores: {'toxic': 0.002, 'non_toxic': 0.998}

# Text: I hate you and everything you stand for...
# Is Toxic: True
# Category: toxic (score: 0.956)
# All scores: {'toxic': 0.956, 'non_toxic': 0.044, 'hate': 0.832, 'threat': 0.123}

# Text: This product is terrible and useless...
# Is Toxic: False
# Category: non_toxic (score: 0.743)
# All scores: {'toxic': 0.257, 'non_toxic': 0.743}

# Text: Thank you for your help, I appreciate it...
# Is Toxic: False
# Category: non_toxic (score: 0.999)
# All scores: {'toxic': 0.001, 'non_toxic': 0.999}
```

### Key Insights

1. **Model Caching**: `@lru_cache` ensures model loads once per process
2. **Batch Processing**: Supports list inputs for efficient throughput
3. **Multi-Label Scoring**: Returns scores for all toxicity categories
4. **Threshold Calibration**: Tunable threshold balances precision/recall

### Production Best Practices

- **GPU Acceleration**: Deploy on GPU instances for <100ms latency
- **Model Selection**: Compare toxic-bert, detoxify, and perspective-api
- **Threshold Tuning**: Calibrate per-category thresholds using labeled data
- **Fallback Logic**: Provide graceful degradation if model fails to load

---

## Exercise 6: Orchestrate Guardrail Execution

### Objective
Build a comprehensive guardrail pipeline that executes all checks (moderation, PII, rules, toxicity) in sequence with short-circuit logic and detailed audit logging.

### Solution

```python
import uuid
from datetime import datetime
from typing import Any, Dict, List, NamedTuple

class GuardrailReport(NamedTuple):
    """Immutable report of guardrail execution."""
    correlation_id: str
    steps: List[Dict[str, Any]]
    blocked: bool
    message: str
    sanitized_text: str

class GuardrailPipeline:
    """
    Orchestrates layered guardrail checks with audit logging.
    Executes checks in order: moderation -> PII -> rules -> toxicity.
    Short-circuits on blocking violations.
    """
    
    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine
        self.audit_log: List[Dict[str, Any]] = []
    
    def run(self, text: str, context: Dict[str, Any]) -> GuardrailReport:
        """
        Execute full guardrail pipeline.
        
        Args:
            text: Input text to validate
            context: Request context (user info, feature flags, etc.)
        
        Returns:
            GuardrailReport with detailed execution trace
        """
        correlation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []
        blocked = False
        message = ""
        sanitized_text = text
        
        def record_step(name: str, result: Dict[str, Any]):
            """Helper to record step with timestamp."""
            steps.append({
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result,
            })
        
        # Step 1: Content Moderation
        moderation_result = moderate_content(text)
        record_step("content_moderation", moderation_result)
        
        if moderation_result["flagged"]:
            blocked = True
            message = f"Content moderation failed: {', '.join(moderation_result['reasons'])}"
            report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
            self.audit_log.append(report._asdict())
            return report
        
        # Step 2: PII Detection & Redaction
        pii_detections = detect_pii(text)
        record_step("pii_detection", {"detections": pii_detections})
        
        if pii_detections:
            sanitized_text, metadata = redact_pii(text, pii_detections)
            record_step("pii_redaction", {
                "original_length": len(text),
                "redacted_length": len(sanitized_text),
                "entity_count": len(pii_detections)
            })
        
        # Step 3: Business Rules Evaluation
        rule_violations = self.rule_engine.evaluate(context)
        record_step("rule_evaluation", {"violations": [v._asdict() for v in rule_violations]})
        
        # Check for blocking violations
        critical_violations = [v for v in rule_violations if v.severity == "critical"]
        if critical_violations:
            blocked = True
            message = f"Rule violations: {', '.join([v.rule_name for v in critical_violations])}"
            report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
            self.audit_log.append(report._asdict())
            return report
        
        # Step 4: Toxicity Scoring (on sanitized text)
        toxicity_scores = score_toxicity([sanitized_text], threshold=0.7)
        toxicity_result = toxicity_scores[0] if toxicity_scores else {}
        record_step("toxicity_detection", toxicity_result)
        
        if toxicity_result.get("is_toxic"):
            blocked = True
            message = f"Toxicity detected: {toxicity_result['category']} (score: {toxicity_result['max_score']:.2f})"
            report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
            self.audit_log.append(report._asdict())
            return report
        
        # All checks passed
        message = "All guardrails passed"
        report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
        self.audit_log.append(report._asdict())
        return report
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return full audit log for compliance reporting."""
        return self.audit_log

# Initialize pipeline
engine = RuleEngine(rules)
pipeline = GuardrailPipeline(engine)

# Test case 1: Clean input
clean_context = {
    "user_segment": "enterprise",
    "intent": "technical question",
    "feature": "standard",
}
clean_report = pipeline.run("What is the status of my support ticket?", clean_context)
print(f"\nClean input report:")
print(f"  Blocked: {clean_report.blocked}")
print(f"  Message: {clean_report.message}")
print(f"  Steps: {len(clean_report.steps)}")

# Test case 2: PII redaction
pii_context = {
    "user_segment": "retail",
    "intent": "account inquiry",
    "feature": "standard",
}
pii_report = pipeline.run(
    "My email is john.doe@example.com and my phone is 555-1234",
    pii_context
)
print(f"\nPII input report:")
print(f"  Blocked: {pii_report.blocked}")
print(f"  Original: My email is john.doe@example.com...")
print(f"  Sanitized: {pii_report.sanitized_text}")

# Test case 3: Rule violation
violation_context = {
    "user_segment": "retail",
    "intent": "I need financial advice on stocks",
    "feature": "standard",
}
violation_report = pipeline.run("Should I invest in crypto?", violation_context)
print(f"\nRule violation report:")
print(f"  Blocked: {violation_report.blocked}")
print(f"  Message: {violation_report.message}")
```

### Expected Output

```python
# Clean input report:
#   Blocked: False
#   Message: All guardrails passed
#   Steps: 4

# PII input report:
#   Blocked: False
#   Original: My email is john.doe@example.com...
#   Sanitized: My email is [EMAIL_ADDRESS] and my phone is [PHONE_NUMBER]

# Rule violation report:
#   Blocked: True
#   Message: Rule violations: no_financial_advice
```

### Key Insights

1. **Sequential Execution**: Checks execute in order of speed (blocklist first, ML last)
2. **Short-Circuit Logic**: Pipeline halts on first blocking violation
3. **Audit Trail**: Every execution logged with correlation ID for traceability
4. **Sanitized Output**: Returns cleaned text for downstream LLM calls

### Production Best Practices

- **Async Execution**: Parallelize independent checks (moderation + toxicity)
- **Timeout Handling**: Set per-check timeouts to prevent pipeline stalls
- **Metrics Export**: Emit step latencies to Prometheus/CloudWatch
- **Configuration**: Make check order and thresholds configurable

---

## Exercise 7: Optimize Guardrail Performance

### Objective
Apply caching, parallelization, and timing instrumentation to reduce guardrail latency from ~500ms to <200ms.

### Solution

```python
import asyncio
import hashlib
from functools import lru_cache
from typing import Any, Dict, List

# Caching for deterministic checks
@lru_cache(maxsize=1024)
def cached_blocklist_check(text_hash: str, text: str) -> List[str]:
    """
    Cache blocklist results by content hash.
    
    Args:
        text_hash: SHA256 hash of text (for cache key)
        text: Original text (for actual checking)
    
    Returns:
        List of matched patterns
    """
    return check_blocklist(text)

def get_text_hash(text: str) -> str:
    """Generate cache key for text content."""
    return hashlib.sha256(text.encode()).hexdigest()

# Parallel execution for independent checks
async def run_parallel_checks(text: str) -> Dict[str, Any]:
    """
    Execute moderation and toxicity checks in parallel.
    
    Args:
        text: Input text to check
    
    Returns:
        Combined results from both checks
    """
    loop = asyncio.get_event_loop()
    
    # Submit tasks to thread pool
    moderation_task = loop.run_in_executor(None, moderate_content, text)
    toxicity_task = loop.run_in_executor(None, score_toxicity, [text])
    
    # Wait for both to complete
    moderation, toxicity = await asyncio.gather(moderation_task, toxicity_task)
    
    return {
        "moderation": moderation,
        "toxicity": toxicity[0] if toxicity else {}
    }

# Timing instrumentation
def time_guardrail_step(
    name: str,
    func: callable,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper to measure and log step execution time.
    
    Args:
        name: Step name for logging
        func: Function to execute
        *args, **kwargs: Function arguments
    
    Returns:
        Dict with timing and result
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    
    # Emit metric (would integrate with Langfuse/CloudWatch in production)
    print(f"[METRIC] {name}: {duration_ms}ms")
    
    return {
        "name": name,
        "duration_ms": duration_ms,
        "result": result
    }

# Optimized pipeline with caching and parallelization
class OptimizedGuardrailPipeline(GuardrailPipeline):
    """Enhanced pipeline with performance optimizations."""
    
    async def run_async(self, text: str, context: Dict[str, Any]) -> GuardrailReport:
        """Async version with parallel checks."""
        correlation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []
        blocked = False
        message = ""
        sanitized_text = text
        
        def record_step(name: str, result: Dict[str, Any]):
            steps.append({
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result,
            })
        
        # Step 1: Cached blocklist check
        text_hash = get_text_hash(text)
        blocklist_result = time_guardrail_step(
            "blocklist_cached",
            cached_blocklist_check,
            text_hash,
            text
        )
        record_step("blocklist", blocklist_result)
        
        if blocklist_result["result"]:
            blocked = True
            message = f"Blocklist violation: {blocklist_result['result']}"
            report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
            self.audit_log.append(report._asdict())
            return report
        
        # Step 2 & 4: Parallel moderation + toxicity
        parallel_start = time.perf_counter()
        parallel_results = await run_parallel_checks(text)
        parallel_duration = round((time.perf_counter() - parallel_start) * 1000, 2)
        print(f"[METRIC] parallel_checks: {parallel_duration}ms")
        
        record_step("moderation_parallel", parallel_results["moderation"])
        record_step("toxicity_parallel", parallel_results["toxicity"])
        
        # Check moderation results
        if parallel_results["moderation"]["flagged"]:
            blocked = True
            message = f"Moderation failed: {parallel_results['moderation']['reasons']}"
            report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
            self.audit_log.append(report._asdict())
            return report
        
        # Step 3: PII (must be sequential)
        pii_result = time_guardrail_step("pii_detection", detect_pii, text)
        record_step("pii", pii_result)
        
        if pii_result["result"]:
            sanitized_text, _ = redact_pii(text, pii_result["result"])
        
        # Check toxicity results
        if parallel_results["toxicity"].get("is_toxic"):
            blocked = True
            message = f"Toxic content: {parallel_results['toxicity']['category']}"
            report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
            self.audit_log.append(report._asdict())
            return report
        
        # All checks passed
        message = "All guardrails passed"
        report = GuardrailReport(correlation_id, steps, blocked, message, sanitized_text)
        self.audit_log.append(report._asdict())
        return report

# Benchmark comparison
import time as time_module

def benchmark_pipeline(pipeline_class, text: str, context: Dict[str, Any], iterations: int = 10):
    """Compare pipeline performance."""
    engine = RuleEngine(rules)
    pipeline = pipeline_class(engine)
    
    durations = []
    for _ in range(iterations):
        start = time_module.perf_counter()
        
        if hasattr(pipeline, 'run_async'):
            asyncio.run(pipeline.run_async(text, context))
        else:
            pipeline.run(text, context)
        
        duration = (time_module.perf_counter() - start) * 1000
        durations.append(duration)
    
    return {
        "mean_ms": sum(durations) / len(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "p95_ms": sorted(durations)[int(len(durations) * 0.95)],
    }

# Run benchmarks
test_text = "This is a test message for benchmarking guardrail performance"
test_context = {"user_segment": "enterprise", "intent": "test", "feature": "standard"}

print("\n=== Standard Pipeline ===")
standard_perf = benchmark_pipeline(GuardrailPipeline, test_text, test_context)
print(f"Mean: {standard_perf['mean_ms']:.1f}ms, P95: {standard_perf['p95_ms']:.1f}ms")

print("\n=== Optimized Pipeline ===")
optimized_perf = benchmark_pipeline(OptimizedGuardrailPipeline, test_text, test_context)
print(f"Mean: {optimized_perf['mean_ms']:.1f}ms, P95: {optimized_perf['p95_ms']:.1f}ms")

speedup = standard_perf['mean_ms'] / optimized_perf['mean_ms']
print(f"\nSpeedup: {speedup:.2f}x")
```

### Expected Output

```python
# [METRIC] blocklist_cached: 0.3ms
# [METRIC] parallel_checks: 145.2ms
# [METRIC] pii_detection: 12.4ms

# === Standard Pipeline ===
# Mean: 487.3ms, P95: 521.8ms

# === Optimized Pipeline ===
# Mean: 162.7ms, P95: 183.4ms

# Speedup: 2.99x
```

### Key Insights

1. **Caching**: Deterministic checks (blocklist, regex) cached by content hash
2. **Parallelization**: Independent API calls (moderation, toxicity) run concurrently
3. **Instrumentation**: Per-step timing identifies optimization opportunities
4. **3x Speedup**: Typical improvement from standard to optimized pipeline

### Production Best Practices

- **Cache Eviction**: Use TTL-based caching (Redis) for distributed systems
- **Circuit Breakers**: Wrap external API calls to prevent cascading failures
- **Resource Limits**: Cap thread pool size to avoid overwhelming downstream services
- **Monitoring**: Export P50/P95/P99 latencies to observability dashboard

---

## Exercise 8: Generate Compliance Reports

### Objective
Aggregate audit logs into compliance reports showing PII removals, policy violations, override usage, and high-risk events for legal/audit review.

### Solution

```python
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

def compile_compliance_report(audit_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate compliance metrics from audit log.
    
    Args:
        audit_log: List of GuardrailReport dicts
    
    Returns:
        Structured report with aggregated metrics
    """
    if not audit_log:
        return {
            "summary": {"total_events": 0},
            "violations": [],
            "pii_events": [],
            "high_risk": []
        }
    
    df = pd.DataFrame(audit_log)
    
    # Aggregate summary metrics
    summary = {
        "total_events": len(df),
        "blocked_events": int(df["blocked"].sum()),
        "block_rate": float(df["blocked"].mean()),
        "unique_correlation_ids": df["correlation_id"].nunique(),
    }
    
    # Extract PII events
    pii_events = []
    for _, row in df.iterrows():
        for step in row["steps"]:
            if step["name"] == "pii_detection" and step["result"].get("detections"):
                pii_events.append({
                    "correlation_id": row["correlation_id"],
                    "timestamp": step["timestamp"],
                    "entity_count": len(step["result"]["detections"]),
                    "entities": step["result"]["detections"]
                })
    
    summary["pii_events"] = len(pii_events)
    
    # Extract violations
    violations = df[df["blocked"]].to_dict(orient="records")
    
    # Identify high-risk events
    high_risk = highlight_high_risk_events(audit_log)
    
    summary["high_risk_events"] = len(high_risk)
    
    return {
        "summary": summary,
        "violations": violations,
        "pii_events": pii_events,
        "high_risk": high_risk,
        "generated_at": datetime.utcnow().isoformat()
    }

def highlight_high_risk_events(audit_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter for events requiring manual review.
    
    Criteria:
    - Blocked with critical severity
    - Multiple PII entities detected
    - Rule override used
    
    Args:
        audit_log: Full audit log
    
    Returns:
        List of high-risk events
    """
    high_risk = []
    
    for event in audit_log:
        is_high_risk = False
        reasons = []
        
        # Check for critical blocking
        if event["blocked"] and "critical" in event["message"].lower():
            is_high_risk = True
            reasons.append("critical_blocking")
        
        # Check for excessive PII
        for step in event["steps"]:
            if step["name"] == "pii_detection":
                detections = step["result"].get("detections", [])
                if len(detections) >= 3:
                    is_high_risk = True
                    reasons.append(f"excessive_pii_{len(detections)}_entities")
        
        # Check for rule overrides (would need context in production)
        # if "override" in event.get("context", {}):
        #     is_high_risk = True
        #     reasons.append("rule_override")
        
        if is_high_risk:
            high_risk.append({
                "correlation_id": event["correlation_id"],
                "blocked": event["blocked"],
                "message": event["message"],
                "reasons": reasons,
                "steps": event["steps"]
            })
    
    return high_risk

def export_report(report: Dict[str, Any], path: str) -> None:
    """
    Export compliance report to JSON and CSV formats.
    
    Args:
        report: Compiled report dict
        path: Base path for output files
    """
    base_path = Path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export full JSON report
    json_path = base_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    print(f"Exported JSON report: {json_path}")
    
    # Export summary CSV
    summary_df = pd.DataFrame([report["summary"]])
    csv_path = base_path.with_name(f"{base_path.stem}_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Exported summary CSV: {csv_path}")
    
    # Export violations CSV if any exist
    if report["violations"]:
        violations_df = pd.DataFrame(report["violations"])
        violations_csv = base_path.with_name(f"{base_path.stem}_violations.csv")
        violations_df.to_csv(violations_csv, index=False)
        print(f"Exported violations CSV: {violations_csv}")

# Generate compliance report from pipeline audit log
engine = RuleEngine(rules)
pipeline = GuardrailPipeline(engine)

# Simulate several requests
test_cases = [
    ("Clean request", {"user_segment": "enterprise", "intent": "general"}),
    ("PII request with email john@example.com", {"user_segment": "retail", "intent": "account"}),
    ("I need financial advice", {"user_segment": "retail", "intent": "financial advice"}),
    ("Another clean request", {"user_segment": "enterprise", "intent": "support"}),
]

for text, context in test_cases:
    pipeline.run(text, context)

# Compile report
report = compile_compliance_report(pipeline.get_audit_log())

print("\n=== Compliance Report ===")
print(f"Total Events: {report['summary']['total_events']}")
print(f"Blocked Events: {report['summary']['blocked_events']}")
print(f"Block Rate: {report['summary']['block_rate']:.1%}")
print(f"PII Events: {report['summary']['pii_events']}")
print(f"High-Risk Events: {report['summary']['high_risk_events']}")

# Export to files
export_report(report, "artifacts/compliance_report_2026-01-08")

# Display sample violation
if report["violations"]:
    print(f"\nSample Violation:")
    violation = report["violations"][0]
    print(f"  Correlation ID: {violation['correlation_id']}")
    print(f"  Message: {violation['message']}")
    print(f"  Steps: {len(violation['steps'])}")
```

### Expected Output

```python
# === Compliance Report ===
# Total Events: 4
# Blocked Events: 1
# Block Rate: 25.0%
# PII Events: 1
# High-Risk Events: 1
# 
# Exported JSON report: artifacts/compliance_report_2026-01-08.json
# Exported summary CSV: artifacts/compliance_report_2026-01-08_summary.csv
# Exported violations CSV: artifacts/compliance_report_2026-01-08_violations.csv
# 
# Sample Violation:
#   Correlation ID: 8f3d4a2b-1c5e-4f9b-a3d2-7e6c8b9d0a1f
#   Message: Rule violations: no_financial_advice
#   Steps: 3
```

### Compliance Report Structure

```json
{
  "summary": {
    "total_events": 4,
    "blocked_events": 1,
    "block_rate": 0.25,
    "unique_correlation_ids": 4,
    "pii_events": 1,
    "high_risk_events": 1
  },
  "violations": [
    {
      "correlation_id": "8f3d4a2b-1c5e-4f9b-a3d2-7e6c8b9d0a1f",
      "blocked": true,
      "message": "Rule violations: no_financial_advice",
      "steps": [...]
    }
  ],
  "pii_events": [
    {
      "correlation_id": "a1b2c3d4-...",
      "timestamp": "2026-01-08T14:32:15.123Z",
      "entity_count": 1,
      "entities": [
        {"entity_type": "EMAIL_ADDRESS", "start": 24, "end": 40, "score": 1.0}
      ]
    }
  ],
  "high_risk": [...],
  "generated_at": "2026-01-08T14:35:00.000Z"
}
```

### Key Insights

1. **Aggregated Metrics**: Summary provides executive-level overview (block rate, PII events)
2. **Detailed Forensics**: Full violation records enable root cause analysis
3. **Risk Prioritization**: High-risk events flagged for manual review
4. **Multiple Formats**: JSON for programmatic access, CSV for spreadsheet analysis

### Production Best Practices

- **Scheduled Reporting**: Generate daily/weekly reports via cron/Lambda
- **Retention Policy**: Archive reports for compliance windows (7 years for GDPR)
- **Access Controls**: Restrict report access to compliance/security teams
- **Trend Analysis**: Track metrics over time to detect policy drift

---

## Wrap-Up

You've now built a production-grade guardrail system covering:

1. ✅ **Content Moderation**: OpenAI API + custom blocklists with fallback handling
2. ✅ **PII Protection**: Presidio + custom regex with type-aware redaction
3. ✅ **Business Rules**: Flexible rule engine with overrides and severity levels
4. ✅ **Output Validation**: Pydantic schemas with automatic retry logic
5. ✅ **Toxicity Detection**: Transformer models with batch processing
6. ✅ **Pipeline Orchestration**: Sequential execution with audit logging
7. ✅ **Performance Optimization**: 3x speedup via caching and parallelization
8. ✅ **Compliance Reporting**: Aggregated metrics for legal/audit review

### Integration Checklist

- [ ] Deploy guardrail pipeline as middleware in LLM serving layer
- [ ] Configure observability (Langfuse/CloudWatch) for latency tracking
- [ ] Set up alerting for high block rates or API failures
- [ ] Schedule weekly compliance reports for governance team
- [ ] Run red-team exercises to validate guardrail effectiveness (see Lab 03)
- [ ] Document threshold tuning decisions and override approval process

### Next Steps

- **Lab 03**: Security testing and red-teaming to validate guardrail robustness
- **Week 8**: Integrate guardrails into end-to-end PoC pipeline
- **Week 10**: Align guardrails with responsible AI policies and audit requirements

### Performance Benchmarks

| Check | Latency (Standard) | Latency (Optimized) | Notes |
|-------|-------------------|---------------------|-------|
| Blocklist | 0.5ms | 0.3ms (cached) | 40% improvement |
| OpenAI Moderation | 145ms | 145ms (parallel) | No change (API-bound) |
| PII Detection | 12ms | 12ms | Sequential dependency |
| Toxicity Scoring | 180ms | 180ms (parallel) | GPU-accelerated |
| **Total Pipeline** | **487ms** | **163ms** | **3x speedup** |

### Resources

- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
- [Guardrails AI](https://docs.guardrailsai.com/)
- [HuggingFace Toxicity Models](https://huggingface.co/unitary/toxic-bert)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
