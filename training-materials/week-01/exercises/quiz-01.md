# Exercise 1: GenAI Concepts Quiz

**Time:** 30 minutes  
**Type:** Knowledge Check  
**Difficulty:** Beginner

---

## Instructions

This quiz tests your understanding of the fundamental concepts covered in Week 1. Answer all questions to the best of your ability. Some questions may have multiple correct answers.

---

## Section 1: Generative AI Fundamentals (10 questions)

### Question 1
What is the primary difference between generative AI and discriminative AI?

**A)** Generative AI classifies data, while discriminative AI creates new data  
**B)** Generative AI creates new data, while discriminative AI classifies existing data  
**C)** Generative AI is faster than discriminative AI  
**D)** There is no meaningful difference

---

### Question 2
Which of the following are examples of generative AI models? (Select all that apply)

**A)** GPT-4  
**B)** Random Forest Classifier  
**C)** DALL-E  
**D)** Support Vector Machine  
**E)** Stable Diffusion

---

### Question 3
What is a "token" in the context of Large Language Models?

**A)** A security credential for API access  
**B)** A unit of text (word, subword, or character) that the model processes  
**C)** A type of neural network architecture  
**D)** A payment method for API usage

---

### Question 4
True or False: The Transformer architecture was specifically designed for language models and cannot be used for other types of data.

**A)** True  
**B)** False

---

### Question 5
What is the purpose of the "attention mechanism" in Transformer models?

**A)** To make the model train faster  
**B)** To allow the model to focus on relevant parts of the input when generating output  
**C)** To reduce memory usage  
**D)** To encrypt sensitive data

---

### Question 6
Which of the following statements about embeddings is correct?

**A)** Embeddings are human-readable text representations  
**B)** Embeddings are high-dimensional vector representations of text  
**C)** Embeddings can only represent individual words, not sentences  
**D)** Embeddings are only used for classification tasks

---

### Question 7
What is "fine-tuning" in the context of LLMs?

**A)** Adjusting API parameters like temperature  
**B)** Training a pre-trained model on specific data for a particular task  
**C)** Optimizing prompt wording  
**D)** Reducing model size for deployment

---

### Question 8
Which of the following are common applications of generative AI? (Select all that apply)

**A)** Code generation and completion  
**B)** Spam email detection  
**C)** Content summarization  
**D)** Image classification  
**E)** Text-to-image generation

---

### Question 9
What is the typical relationship between model size (parameters) and capability?

**A)** Smaller models are always better  
**B)** Larger models generally have better performance but higher costs  
**C)** Model size has no impact on performance  
**D)** Larger models are always worse due to overfitting

---

### Question 10
True or False: LLMs can only work with English language text.

**A)** True  
**B)** False

---

## Section 2: OpenAI API & Practical Usage (10 questions)

### Question 11
What is the purpose of the `temperature` parameter in the OpenAI API?

**A)** To control the physical temperature of the server  
**B)** To control the randomness/creativity of model outputs  
**C)** To set the maximum response length  
**D)** To specify the model version

---

### Question 12
What happens when you set `temperature=0`?

**A)** The API request fails  
**B)** The model produces more random/creative outputs  
**C)** The model produces more deterministic/consistent outputs  
**D)** The model becomes faster

---

### Question 13
What is the purpose of the `max_tokens` parameter?

**A)** To set the maximum cost per request  
**B)** To limit the length of the model's response  
**C)** To specify how many API calls you can make  
**D)** To control the randomness of outputs

---

### Question 14
In the OpenAI Chat API, what are the three main message roles?

**A)** admin, user, assistant  
**B)** system, user, assistant  
**C)** input, output, error  
**D)** prompt, response, feedback

---

### Question 15
What is the purpose of the `system` message in a chat completion?

**A)** To provide error messages  
**B)** To set high-level instructions and behavior for the assistant  
**C)** To store conversation history  
**D)** To specify the API version

---

### Question 16
True or False: Token usage includes both input (prompt) and output (completion) tokens.

**A)** True  
**B)** False

---

### Question 17
What happens if your prompt exceeds the model's context window?

**A)** The model automatically summarizes it  
**B)** The API returns an error  
**C)** Only the first part of the prompt is processed  
**D)** The request costs more money

---

### Question 18
Which parameter would you adjust to get more varied responses to the same prompt?

**A)** max_tokens  
**B)** temperature  
**C)** model  
**D)** top_p

---

### Question 19
What is "streaming" in the context of API responses?

**A)** Using video instead of text  
**B)** Receiving the response in chunks as it's generated, rather than waiting for completion  
**C)** Processing multiple requests simultaneously  
**D)** Caching responses for faster retrieval

---

### Question 20
True or False: You should hardcode your API key directly in your Python scripts for convenience.

**A)** True  
**B)** False

---

## Section 3: Best Practices & Ethics (5 questions)

### Question 21
Which of the following are recommended practices when working with LLMs? (Select all that apply)

**A)** Store API keys in environment variables  
**B)** Implement error handling and retries  
**C)** Monitor token usage and costs  
**D)** Share your API key with teammates via Slack  
**E)** Validate and sanitize model outputs

---

### Question 22
What is a "hallucination" in the context of LLMs?

**A)** A visual effect in the model's training  
**B)** When the model generates plausible but incorrect or fabricated information  
**C)** A type of model architecture  
**D)** An API error code

---

### Question 23
Why is it important to consider ethical implications when deploying GenAI applications?

**A)** To comply with regulations only  
**B)** To avoid negative impacts on users and society, ensure fairness, and maintain trust  
**C)** It's not important, only technical performance matters  
**D)** Only to reduce costs

---

### Question 24
What should you do before deploying a GenAI application to production? (Select all that apply)

**A)** Test with diverse inputs  
**B)** Implement content filtering/moderation  
**C)** Set up monitoring and logging  
**D)** Skip testing to deploy faster  
**E)** Document limitations and expected behavior

---

### Question 25
True or False: LLMs are completely unbiased and always provide factually accurate information.

**A)** True  
**B)** False

---

## Submission Guidelines

1. Write your answers in a separate document (e.g., `quiz-01-answers.md`)
2. Format: `Question X: [Your Answer]`
3. For multiple-choice questions, list all selected options
4. Include brief explanations for questions you found challenging
5. Compare your answers with the solution file after completing

---

## Grading Criteria

- **23-25 correct:** Excellent understanding ✅
- **20-22 correct:** Good grasp of concepts 👍
- **17-19 correct:** Adequate understanding, review weak areas 📚
- **Below 17:** Review Week 1 materials thoroughly and retake 🔄

---

## Self-Assessment

After completing the quiz:
1. Identify topics where you struggled
2. Review corresponding lesson materials
3. Discuss challenging concepts with peers or instructors
4. Retake quiz if needed to solidify understanding

---

**Time to Complete:** ~30 minutes  
**Solutions Available:** `solutions/quiz-01-solutions.md`
