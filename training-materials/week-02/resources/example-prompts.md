# Example Prompts (Copy & Adapt)

Use these as starting points. Always adapt to your domain and data.

## 1. Summarization (Enterprise Memo)
```
You are an executive assistant. Summarize the memo below for directors.
- Max 7 bullet points
- Preserve numbers, dates, and decisions
- Include risks and owners

MEMO:
```text
{{PASTE_MEMO}}
```
```

## 2. Extract Structured Data (JSON)
```
Extract fields from the text. Respond with valid JSON matching the schema.
Schema:
{
  "title": "string",
  "date": "YYYY-MM-DD",
  "owner": "string",
  "actions": ["string"]
}

TEXT:
```text
{{PASTE_TEXT}}
```
```

## 3. Classification (Multi-Label)
```
Classify the ticket into one or more categories:
[CLOUD, SECURITY, BILLING, NETWORK, DEVTOOLS]
Return JSON: {"labels": ["..."]}

TICKET:
```text
{{PASTE_TICKET}}
```
```

## 4. Style Transfer
```
Rewrite the email in a concise, professional tone.
- Keep technical details
- Remove colloquialisms
- 120 words max

EMAIL:
```text
{{PASTE_EMAIL}}
```
```

## 5. SQL Drafting (Assisted)
```
You are a SQL assistant. Given the question and schema, draft a SQL query.
If fields are ambiguous, ask clarifying questions.

QUESTION: {{QUESTION}}
SCHEMA:
- customers(id, name, country, created_at)
- orders(id, customer_id, amount_usd, created_at)
```

## 6. RAG Answering with Citations
```
Answer using only the provided context. If unknown, say you don't know.
Cite sources using [n].

CONTEXT:
```text
{{TOP_K_SNIPPETS}}
```

QUESTION: {{QUESTION}}
```
