# Week 1 Quiz - Solutions

**Time Limit:** 30 minutes  
**Total Points:** 25 points (1 point per question)

---

## Section 1: GenAI Fundamentals & Concepts (10 questions)

### 1. What is the primary difference between traditional programming and Large Language Model (LLM) applications?

**Answer: C) LLMs learn patterns from data and generate probabilistic outputs rather than following explicit rules**

**Explanation:** Traditional programming follows deterministic, rule-based logic, while LLMs use statistical patterns learned from training data to generate probabilistic outputs. This is a fundamental distinction in how these systems operate.

---

### 2. Which of the following best describes "prompt engineering"?

**Answer: B) The practice of crafting effective input instructions to guide LLM behavior and outputs**

**Explanation:** Prompt engineering is the art and science of designing prompts that elicit desired behaviors and outputs from LLMs. It's a critical skill for working effectively with these models.

---

### 3. What is a "token" in the context of LLMs?

**Answer: D) A unit of text (word fragment, word, or character) that the model processes**

**Explanation:** Tokens are the fundamental units that LLMs process. They can represent whole words, parts of words, or single characters, depending on the tokenization scheme used.

---

### 4. In the OpenAI API, what does the "temperature" parameter control?

**Answer: C) The randomness/creativity of the model's responses**

**Explanation:** Temperature controls the randomness of token selection. Lower temperatures (e.g., 0-0.3) make outputs more deterministic and focused, while higher temperatures (e.g., 0.7-2.0) increase randomness and creativity.

---

### 5. What is "few-shot learning" in the context of prompt engineering?

**Answer: B) Providing several examples in the prompt to guide the model's response format**

**Explanation:** Few-shot learning involves including examples in the prompt to demonstrate the desired output format or behavior. This is a powerful technique for improving model performance without fine-tuning.

---

### 6. Which model architecture family does GPT-4 belong to?

**Answer: A) Transformer-based decoder-only architecture**

**Explanation:** GPT models use a decoder-only transformer architecture, which is designed for autoregressive text generation.

---

### 7. What is the purpose of "system messages" in chat-based LLM interactions?

**Answer: B) To set the behavior, tone, and context for the model throughout the conversation**

**Explanation:** System messages establish the overall context and behavior expectations for the model. They persist throughout the conversation and guide how the model interprets and responds to user messages.

---

### 8. What does "context window" refer to in LLMs?

**Answer: D) The maximum amount of text (in tokens) the model can process at once**

**Explanation:** The context window is the maximum number of tokens the model can "see" and process in a single request, including both input and output. For GPT-4, this can range from 8K to 128K tokens depending on the variant.

---

### 9. What is "hallucination" in the context of LLMs?

**Answer: B) When the model generates plausible-sounding but incorrect or fabricated information**

**Explanation:** Hallucination occurs when LLMs confidently generate false information that sounds plausible. This is a critical limitation to be aware of when working with these models.

---

### 10. Which of the following is NOT a recommended strategy for reducing hallucinations?

**Answer: D) Using the highest possible temperature value**

**Explanation:** High temperature increases randomness and can actually increase the likelihood of hallucinations. Better strategies include using lower temperatures, providing clear context, asking for citations, and using retrieval-augmented generation (RAG).

---

## Section 2: OpenAI API & Practical Application (10 questions)

### 11. In the OpenAI Python SDK, which method is used to create a chat completion?

**Answer: A) `client.chat.completions.create()`**

**Explanation:** This is the correct method in the OpenAI Python SDK (version 1.0+) for creating chat completions with models like GPT-4 and GPT-3.5-turbo.

---

### 12. What is the correct structure of a message in the OpenAI Chat API?

**Answer: C) `{"role": "user", "content": "Hello"}`**

**Explanation:** Chat messages require both a `role` field (system, user, or assistant) and a `content` field with the message text.

---

### 13. Which parameter limits the length of the model's response?

**Answer: C) `max_tokens`**

**Explanation:** The `max_tokens` parameter sets the maximum number of tokens the model can generate in its response.

---

### 14. If you want consistent, deterministic responses from an LLM, which temperature value should you use?

**Answer: A) 0**

**Explanation:** Temperature=0 makes the model completely deterministic, always selecting the most likely next token. This is ideal when you need consistent, reproducible outputs.

---

### 15. What is the purpose of the `top_p` parameter (nucleus sampling)?

**Answer: C) To limit token selection to the smallest set whose cumulative probability exceeds p**

**Explanation:** Top_p (nucleus sampling) is an alternative to temperature for controlling randomness. It considers only the top tokens whose cumulative probability mass exceeds the threshold p.

---

### 16. How can you handle longer conversations that exceed the model's context window?

**Answer: D) All of the above**

**Explanation:** All three strategies are valid approaches: summarizing older messages, truncating the conversation, or implementing a sliding window approach. The best choice depends on your specific use case.

---

### 17. What information is included in the `usage` object returned by the API?

**Answer: D) All of the above**

**Explanation:** The usage object provides comprehensive token usage information, including prompt tokens, completion tokens, and the total count. This is essential for cost tracking and optimization.

---

### 18. Which of the following is a valid role in OpenAI chat messages?

**Answer: D) All of the above**

**Explanation:** The OpenAI Chat API supports three roles: "system" (for instructions), "user" (for user inputs), and "assistant" (for model responses).

---

### 19. What is the recommended approach for storing API keys in Python applications?

**Answer: C) Store in a `.env` file and load with `python-dotenv`**

**Explanation:** Using environment variables loaded from a .env file (excluded from version control) is the security best practice for managing sensitive credentials.

---

### 20. What happens if you set `max_tokens` to a value higher than the model's limit?

**Answer: B) The API will use the model's maximum limit instead**

**Explanation:** The API automatically caps max_tokens at the model's maximum value. The request won't fail, but it will be limited to the model's constraints.

---

## Section 3: Best Practices & Ethics (5 questions)

### 21. Which of the following is a best practice for production LLM applications?

**Answer: D) All of the above**

**Explanation:** All three practices are essential for production systems: implementing retry logic for reliability, monitoring costs for budget control, and validating outputs for quality assurance.

---

### 22. What is the primary concern with using LLMs for medical or legal advice?

**Answer: C) Risk of hallucinations and lack of accountability**

**Explanation:** While LLMs can be helpful tools, they can generate incorrect information confidently (hallucinations) and don't provide the accountability needed for high-stakes domains like medicine and law.

---

### 23. When building customer-facing LLM applications, which safety measure is most important?

**Answer: D) All of the above**

**Explanation:** Comprehensive safety requires content filtering, monitoring for harmful outputs, and clear communication about AI limitations. No single measure is sufficient.

---

### 24. What is "data poisoning" in the context of LLMs?

**Answer: B) Malicious manipulation of training data to influence model behavior**

**Explanation:** Data poisoning refers to intentionally corrupting training data to cause undesired model behaviors. This is a significant security concern for ML systems.

---

### 25. Which practice helps ensure responsible AI development?

**Answer: D) All of the above**

**Explanation:** Responsible AI development requires diverse teams to avoid blind spots, transparency about capabilities and limitations, and regular audits for bias and fairness issues.

---

## Grading Scale

- **23-25 correct (92-100%)**: Excellent understanding
- **20-22 correct (80-88%)**: Good understanding, review missed topics
- **17-19 correct (68-76%)**: Adequate understanding, additional study recommended
- **Below 17 (< 68%)**: Review Week 1 materials before proceeding

---

## Key Takeaways

### Section 1 - Fundamentals:
- LLMs work through probabilistic pattern matching, not rule-based logic
- Tokens are the fundamental processing units for LLMs
- Temperature controls randomness/creativity in outputs
- System messages set persistent behavior and context
- Context windows limit the amount of text that can be processed at once

### Section 2 - API Usage:
- `client.chat.completions.create()` is the core method for chat completions
- Messages require both `role` and `content` fields
- `max_tokens` limits response length, not total tokens
- Temperature=0 provides deterministic outputs
- Always use environment variables for API keys

### Section 3 - Ethics & Best Practices:
- Implement retry logic and error handling in production
- Monitor costs and usage metrics
- Be cautious with LLMs in high-stakes domains (medical, legal)
- Implement comprehensive safety measures (content filtering, monitoring, transparency)
- Ensure diverse teams and regular bias audits

---

## Common Misconceptions Addressed

1. **"Higher temperature always means better outputs"**
   - False: Temperature should match the task. Factual tasks need low temperature; creative tasks can use higher values.

2. **"max_tokens limits the total conversation length"**
   - False: max_tokens only limits the response length. Total conversation is limited by the context window.

3. **"LLMs can reliably provide expert advice in specialized domains"**
   - False: LLMs can hallucinate and should not replace human experts in critical domains.

4. **"API keys can be safely hardcoded in applications"**
   - False: Always use environment variables or secure secret management systems.

---

## Further Reading

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Best Practices for Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [OpenAI Usage Policies](https://openai.com/policies/usage-policies)
- [Responsible AI Practices](https://www.microsoft.com/en-us/ai/responsible-ai)
