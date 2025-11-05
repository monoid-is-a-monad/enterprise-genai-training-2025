# Structured Outputs Cheatsheet

## JSON Mode Prompt Skeleton
```
You MUST respond with valid JSON only. Do not include prose.
Use this exact schema:
{
  "answer": "string",
  "confidence": "low|medium|high",
  "citations": ["string"]
}
```

## Pydantic v2 Parsing
```python
from pydantic import BaseModel, Field

class QA(BaseModel):
    answer: str
    confidence: str = Field(pattern=r"^(low|medium|high)$")
    citations: list[str] = []

obj = QA.model_validate_json(model_output_json)
```

## Validation Flow
1. Ask model for JSON (low temperature)
2. Validate with JSON Schema/Pydantic
3. If invalid → repair prompt (show errors) → retry with backoff

## Repair Prompt
```
The previous JSON failed validation with errors:
- {errors}
Return corrected JSON only, no prose.
```

## Tips
- Keep schemas compact; avoid huge descriptions
- Use enums/regex to constrain values
- Prefer nested objects over long strings for complex outputs
- Log malformed outputs for analysis
