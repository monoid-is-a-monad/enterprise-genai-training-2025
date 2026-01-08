# Lab 2: Few-Shot Learning Experiments - Solutions

**Duration:** 90 minutes  
**Difficulty:** Intermediate

---

## Overview

This solution guide demonstrates effective few-shot learning techniques, showing how to select good examples, structure prompts, and handle various classification and generation tasks. Compare your implementations with these solutions to improve your few-shot prompting skills.

---

## Part 1: Few-Shot Classification

### Exercise 1.1: Sentiment Classification with Examples

**Task:** Build a few-shot classifier for customer review sentiment.

**Solution:**

```python
from openai import OpenAI

client = OpenAI()

def few_shot_sentiment_classifier(review_text):
    """Classify sentiment using few-shot examples."""
    
    prompt = """Classify the sentiment of customer reviews as Positive, Negative, or Neutral.

Examples:

Review: "This product exceeded all my expectations! The quality is outstanding and it arrived ahead of schedule."
Sentiment: Positive

Review: "Terrible experience. The item broke after two days and customer service was unhelpful."
Sentiment: Negative

Review: "The product works as described. Nothing special but does the job."
Sentiment: Neutral

Review: "Amazing! Best purchase I've made this year. Highly recommend to everyone."
Sentiment: Positive

Review: "Disappointed with the quality. Not worth the price at all."
Sentiment: Negative

Now classify this review:

Review: "{review}"
Sentiment:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(review=review_text)}],
        temperature=0.1,  # Low temperature for consistent classification
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip()

# Test cases
test_reviews = [
    "Love it! Works perfectly and the design is sleek.",
    "Complete waste of money. Broke within a week.",
    "It's okay. Does what it's supposed to do.",
    "Absolutely fantastic! Can't believe how well this works.",
    "Not impressed. Expected better quality for this price."
]

print("Few-Shot Sentiment Classification Results:")
print("=" * 60)

for i, review in enumerate(test_reviews, 1):
    sentiment = few_shot_sentiment_classifier(review)
    print(f"\n{i}. Review: {review}")
    print(f"   Sentiment: {sentiment}")
```

**Expected Output:**
```
Few-Shot Sentiment Classification Results:
============================================================

1. Review: Love it! Works perfectly and the design is sleek.
   Sentiment: Positive

2. Review: Complete waste of money. Broke within a week.
   Sentiment: Negative

3. Review: It's okay. Does what it's supposed to do.
   Sentiment: Neutral

4. Review: Absolutely fantastic! Can't believe how well this works.
   Sentiment: Positive

5. Review: Not impressed. Expected better quality for this price.
   Sentiment: Negative
```

**Why This Works:**
- **5 examples** provide solid pattern coverage (2 positive, 2 negative, 1 neutral)
- **Diverse language** shows different ways to express each sentiment
- **Clear format** makes it easy for model to follow pattern
- **Low temperature** ensures consistent classification
- **Balanced examples** prevent bias toward any sentiment

---

### Exercise 1.2: Intent Classification for Customer Support

**Task:** Classify customer support requests into categories.

**Solution:**

```python
def few_shot_intent_classifier(customer_message):
    """Classify customer support intent using few-shot learning."""
    
    prompt = """Classify customer support messages into one of these categories:
- Technical Support
- Billing Question
- Product Information
- Complaint
- Feature Request

Examples:

Message: "I can't log in to my account. It keeps saying my password is incorrect."
Category: Technical Support

Message: "When will my credit card be charged for this month's subscription?"
Category: Billing Question

Message: "Does this plan include unlimited storage?"
Category: Product Information

Message: "I've been waiting 3 days for a response to my ticket. This is unacceptable."
Category: Complaint

Message: "It would be great if you could add dark mode to the mobile app."
Category: Feature Request

Message: "How do I export my data to CSV format?"
Category: Technical Support

Message: "I was charged twice for my last order. Can you help?"
Category: Billing Question

Now classify this message:

Message: "{message}"
Category:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(message=customer_message)}],
        temperature=0.1,
        max_tokens=20
    )
    
    return response.choices[0].message.content.strip()

# Test cases
test_messages = [
    "The app crashes every time I try to upload a file.",
    "Can you show me the pricing for the enterprise plan?",
    "This service is terrible! I want a refund immediately.",
    "Please add support for Google Drive integration.",
    "Why was I charged $99 instead of $79 this month?"
]

print("\nIntent Classification Results:")
print("=" * 60)

for i, message in enumerate(test_messages, 1):
    intent = few_shot_intent_classifier(message)
    print(f"\n{i}. Message: {message}")
    print(f"   Intent: {intent}")
```

**Expected Output:**
```
Intent Classification Results:
============================================================

1. Message: The app crashes every time I try to upload a file.
   Intent: Technical Support

2. Message: Can you show me the pricing for the enterprise plan?
   Intent: Product Information

3. Message: This service is terrible! I want a refund immediately.
   Intent: Complaint

4. Message: Please add support for Google Drive integration.
   Intent: Feature Request

5. Message: Why was I charged $99 instead of $79 this month?
   Intent: Billing Question
```

**Key Improvements:**
1. **7 examples** - More complex task needs more examples
2. **Even distribution** - At least one example per category, two for common ones
3. **Realistic messages** - Examples match actual customer language
4. **Clear category list** - Listed upfront for reference

---

## Part 2: Few-Shot Generation

### Exercise 2.1: Product Description Generator

**Task:** Generate product descriptions in a consistent style.

**Solution:**

```python
def few_shot_product_description(product_name, features):
    """Generate product descriptions using few-shot learning."""
    
    features_text = "\n".join(f"- {feature}" for feature in features)
    
    prompt = """Generate compelling product descriptions for tech products. Follow the style and structure shown in the examples.

Example 1:
Product: Wireless Ergonomic Mouse
Features:
- Vertical design reduces wrist strain
- 2400 DPI precision sensor
- 6 programmable buttons
- 60-day battery life

Description: Say goodbye to wrist pain with our Wireless Ergonomic Mouse. Its innovative vertical design promotes a natural hand position, reducing strain during long work sessions. The precision 2400 DPI sensor ensures smooth, accurate cursor control, while 6 programmable buttons put your favorite commands at your fingertips. With an impressive 60-day battery life, you'll spend more time working and less time charging.

Example 2:
Product: USB-C Hub Multi-Port Adapter
Features:
- 7-in-1 connectivity
- 4K HDMI output
- 100W power delivery
- Aluminum body

Description: Transform your laptop into a productivity powerhouse with our 7-in-1 USB-C Hub. Connect to dual 4K monitors, transfer files at lightning speed, and charge your device—all through a single port. The sleek aluminum body isn't just beautiful; it efficiently dissipates heat for reliable performance. Whether you're presenting in the boardroom or working from your home office, this hub delivers the connections you need.

Example 3:
Product: Mechanical Keyboard
Features:
- Hot-swappable switches
- RGB per-key lighting
- Detachable USB-C cable
- PBT keycaps

Description: Experience typing perfection with our premium Mechanical Keyboard. Hot-swappable switches let you customize your feel without soldering, while durable PBT keycaps ensure years of pristine legends. Illuminate your workspace with stunning per-key RGB lighting that adapts to your mood or setup. The detachable USB-C cable makes transport effortless—take your ultimate typing experience wherever you go.

Now generate a description for:

Product: {product}
Features:
{features}

Description:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(
            product=product_name,
            features=features_text
        )}],
        temperature=0.7,  # Moderate creativity for engaging copy
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()

# Test cases
products = [
    ("Noise-Cancelling Headphones", [
        "Active noise cancellation",
        "30-hour battery life",
        "Premium leather ear cups",
        "Multi-device connectivity"
    ]),
    ("Portable SSD", [
        "1TB storage capacity",
        "Read speeds up to 1050 MB/s",
        "Shock-resistant design",
        "USB-C and USB-A compatible"
    ])
]

print("\nGenerated Product Descriptions:")
print("=" * 60)

for product_name, features in products:
    description = few_shot_product_description(product_name, features)
    print(f"\n{product_name}")
    print(f"\nFeatures: {', '.join(features)}")
    print(f"\nDescription:\n{description}")
    print("-" * 60)
```

**Expected Output Style:**
```
Generated Product Descriptions:
============================================================

Noise-Cancelling Headphones

Features: Active noise cancellation, 30-hour battery life, Premium leather ear cups, Multi-device connectivity

Description:
Immerse yourself in pure audio bliss with our Noise-Cancelling Headphones. Advanced active noise cancellation technology silences the world around you, letting you focus on what matters—your music, podcasts, or calls. Premium leather ear cups provide luxurious comfort for all-day wear, while an extraordinary 30-hour battery life keeps you listening through the longest flights or workdays. Seamlessly switch between devices with multi-device connectivity, staying connected to your laptop, phone, and tablet simultaneously.
------------------------------------------------------------

Portable SSD

Features: 1TB storage capacity, Read speeds up to 1050 MB/s, Shock-resistant design, USB-C and USB-A compatible

Description:
Take your data anywhere with confidence using our Portable SSD. With a massive 1TB capacity, store thousands of photos, hours of 4K video, or your entire project library in your pocket. Blazing-fast read speeds up to 1050 MB/s mean no more waiting—transfer large files in seconds, not minutes. The rugged, shock-resistant design protects your precious data from life's bumps and drops. Universal compatibility with both USB-C and USB-A ensures you're ready to connect to any device, anywhere.
------------------------------------------------------------
```

**Pattern Analysis:**
- **Hook opening** - "Say goodbye to...", "Transform...", "Experience..."
- **Feature integration** - Each feature woven into benefits
- **Value focus** - Emphasizes what user gains
- **Action words** - Dynamic verbs (transform, deliver, illuminate)
- **Closing impact** - Ends with compelling benefit or use case

---

### Exercise 2.2: Email Response Generator

**Task:** Generate professional email responses in consistent style.

**Solution:**

```python
def few_shot_email_response(customer_email, context=""):
    """Generate email responses using few-shot examples."""
    
    prompt = """Generate professional customer service email responses. Match the tone and structure of these examples.

Example 1:
Customer Email: "I ordered a keyboard 5 days ago and it still hasn't shipped. Can you tell me what's going on?"

Response:
Hi [Customer Name],

Thank you for reaching out, and I apologize for the delay with your keyboard order.

I've checked on your order #12345 and can see it's currently being prepared in our warehouse. Due to higher than expected demand, we're experiencing a 2-3 day delay in shipping. Your order is scheduled to ship tomorrow, and you'll receive a tracking number via email as soon as it's on its way.

To make up for the inconvenience, I've applied a 15% discount to your next purchase. The code is THANKYOU15 and it's valid for the next 30 days.

Is there anything else I can help you with?

Best regards,
[Agent Name]
Customer Support Team

---

Example 2:
Customer Email: "The software keeps crashing when I try to export files. I've tried restarting but it doesn't help."

Response:
Hi [Customer Name],

Thank you for contacting us about the export issue you're experiencing. I'm sorry you're running into this problem—let's get it resolved for you.

This type of crash is often related to file size or format. Could you help me narrow down the issue by providing:

• What file format are you trying to export? (PDF, CSV, etc.)
• Approximately how large is the file?
• Are you getting any error messages?

In the meantime, here are two quick fixes that have worked for similar cases:

1. Try exporting in smaller batches (e.g., 100 records at a time)
2. Ensure you're using the latest version of the software (check Help > About)

Once I have those details, I'll be able to provide more specific guidance. If the issue persists, we can schedule a screen-share session to troubleshoot together.

Looking forward to getting this sorted out for you!

Best regards,
[Agent Name]
Customer Support Team

---

Example 3:
Customer Email: "I love your product! Is there any way to get a bulk discount for my company?"

Response:
Hi [Customer Name],

Thank you so much for the kind words about our product—we're thrilled to hear you're enjoying it!

Absolutely, we do offer volume discounts for business customers. Our team licenses start at 10 users and include several additional benefits:

• 20-30% discount based on number of licenses
• Dedicated account manager
• Priority support
• Centralized billing and user management
• Custom onboarding sessions

I'd love to discuss your company's needs in more detail. Could you share:

• How many licenses you're interested in?
• Any specific features or integrations you require?
• Your preferred timeline for implementation?

I'll then prepare a custom quote tailored to your situation. We can also schedule a call if you'd like to discuss further.

Thank you for considering us for your company's needs!

Best regards,
[Agent Name]
Customer Support Team

---

Now generate a response to this email:

Customer Email: "{email}"
{context}

Response:"""
    
    context_text = f"Additional Context: {context}" if context else ""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(
            email=customer_email,
            context=context_text
        )}],
        temperature=0.6,  # Balanced creativity and consistency
        max_tokens=400
    )
    
    return response.choices[0].message.content.strip()

# Test cases
test_emails = [
    ("I received the wrong item. I ordered a blue mouse but got a red one.", 
     "Order #45678, customer has been with us for 2 years"),
    
    ("Do you offer a student discount?", 
     ""),
    
    ("The product arrived damaged. The box looked fine but the item inside was broken.",
     "Order #78901, delivered yesterday")
]

print("\nGenerated Email Responses:")
print("=" * 80)

for email, context in test_emails:
    response = few_shot_email_response(email, context)
    print(f"\nCustomer Email:\n{email}")
    if context:
        print(f"\nContext: {context}")
    print(f"\nGenerated Response:\n{response}")
    print("-" * 80)
```

**Pattern Observed:**
1. **Greeting** - Personalized, friendly
2. **Acknowledgment** - Validates customer's concern
3. **Core response** - Addresses the issue directly
4. **Action items** - Clear next steps or questions
5. **Compensation** (if applicable) - Discount, expedited shipping
6. **Closing** - Offers further help, positive tone
7. **Signature** - Professional sign-off

---

## Part 3: Few-Shot Data Formatting

### Exercise 3.1: Structured Data Extraction

**Task:** Extract structured data from unstructured text.

**Solution:**

```python
import json

def few_shot_data_extraction(text):
    """Extract structured data using few-shot examples."""
    
    prompt = """Extract structured information from event announcements and return as JSON.

Example 1:
Text: "Join us for our Annual Tech Conference on March 15-17, 2025 at the San Francisco Convention Center. Tickets are $499 for general admission or $899 for VIP access. Register at techconf2025.com"

JSON:
{
  "event_name": "Annual Tech Conference",
  "start_date": "2025-03-15",
  "end_date": "2025-03-17",
  "location": "San Francisco Convention Center",
  "ticket_prices": {
    "general": 499,
    "vip": 899
  },
  "website": "techconf2025.com"
}

---

Example 2:
Text: "Don't miss the Summer Music Festival happening June 20-22 at Central Park, NYC! Early bird tickets are just $75, or pay $120 at the door. Visit summerfest.org for the lineup."

JSON:
{
  "event_name": "Summer Music Festival",
  "start_date": "2025-06-20",
  "end_date": "2025-06-22",
  "location": "Central Park, NYC",
  "ticket_prices": {
    "early_bird": 75,
    "door": 120
  },
  "website": "summerfest.org"
}

---

Example 3:
Text: "Startup Networking Mixer on Thursday, April 10th, 2025, 6-9 PM at The Hub in Austin, Texas. Free for members, $25 for non-members. RSVP at startups-atx.com"

JSON:
{
  "event_name": "Startup Networking Mixer",
  "start_date": "2025-04-10",
  "end_date": "2025-04-10",
  "location": "The Hub, Austin, Texas",
  "ticket_prices": {
    "members": 0,
    "non_members": 25
  },
  "website": "startups-atx.com"
}

---

Now extract from:

Text: "{text}"

JSON:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(text=text)}],
        temperature=0.1,  # Low temperature for structured output
        max_tokens=300
    )
    
    # Parse JSON response
    json_text = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_response": json_text}

# Test cases
test_texts = [
    "Photography Workshop - Learn from the pros! Saturday, May 5, 2025 at Golden Gate Park. Workshop fee: $150 per person. Limited to 20 participants. Sign up at photopros.net",
    
    "Food & Wine Expo coming to Chicago on September 12-14, 2025 at McCormick Place. Weekend pass: $85, Single day: $45. Info: foodwinechi.com"
]

print("\nStructured Data Extraction Results:")
print("=" * 80)

for i, text in enumerate(test_texts, 1):
    print(f"\n{i}. Input Text:")
    print(f"   {text}")
    
    extracted = few_shot_data_extraction(text)
    print(f"\n   Extracted JSON:")
    print(f"   {json.dumps(extracted, indent=2)}")
    print("-" * 80)
```

---

### Exercise 3.2: Format Conversion

**Task:** Convert between different data formats using few-shot examples.

**Solution:**

```python
def few_shot_format_conversion(data, from_format, to_format):
    """Convert data between formats using few-shot learning."""
    
    prompt = """Convert data between different formats as shown in these examples.

Example 1 - CSV to Markdown Table:
Input (CSV):
Name,Department,Years
Alice Johnson,Engineering,5
Bob Smith,Marketing,3
Carol Lee,Sales,7

Output (Markdown Table):
| Name | Department | Years |
|------|------------|-------|
| Alice Johnson | Engineering | 5 |
| Bob Smith | Marketing | 3 |
| Carol Lee | Sales | 7 |

---

Example 2 - Bullet List to Numbered List:
Input (Bullet List):
• Learn Python basics
• Build a simple project
• Practice daily coding
• Join a coding community

Output (Numbered List):
1. Learn Python basics
2. Build a simple project
3. Practice daily coding
4. Join a coding community

---

Example 3 - Unstructured to YAML:
Input (Unstructured):
Product: Laptop. Price: $999. Stock: 15 units. Rating: 4.5 stars

Output (YAML):
product: Laptop
price: 999
stock: 15
rating: 4.5

---

Now convert:

Input ({from_fmt}):
{data}

Output ({to_fmt}):"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.format(
            from_fmt=from_format,
            data=data,
            to_fmt=to_format
        )}],
        temperature=0.1,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

# Test conversions
conversions = [
    ("""ID,Product,Price
101,Keyboard,$79
102,Mouse,$45
103,Monitor,$299""", "CSV", "Markdown Table"),
    
    ("""1. Review pull requests
2. Update documentation
3. Fix critical bugs
4. Deploy to staging""", "Numbered List", "JSON Array"),
    
    ("Title: GenAI Workshop | Date: 2025-03-20 | Duration: 4 hours | Instructor: Dr. Smith", 
     "Unstructured", "YAML")
]

print("\nFormat Conversion Results:")
print("=" * 80)

for data, from_fmt, to_fmt in conversions:
    converted = few_shot_format_conversion(data, from_fmt, to_fmt)
    print(f"\nFrom: {from_fmt}")
    print(f"To: {to_fmt}")
    print(f"\nInput:\n{data}")
    print(f"\nOutput:\n{converted}")
    print("-" * 80)
```

---

## Part 4: Example Selection Strategies

### Exercise 4.1: Dynamic Example Selection

**Task:** Select the most relevant examples for a given input.

**Solution:**

```python
from typing import List, Tuple
import numpy as np

class FewShotExampleSelector:
    """Intelligently select few-shot examples based on similarity."""
    
    def __init__(self, examples: List[Tuple[str, str]]):
        """
        Initialize with a pool of examples.
        
        Args:
            examples: List of (input, output) tuples
        """
        self.examples = examples
    
    def select_by_length(self, query: str, n: int = 3) -> List[Tuple[str, str]]:
        """Select examples with similar length to query."""
        query_len = len(query.split())
        
        # Calculate length difference for each example
        examples_with_diff = [
            (ex, abs(len(ex[0].split()) - query_len))
            for ex in self.examples
        ]
        
        # Sort by difference and take top n
        examples_with_diff.sort(key=lambda x: x[1])
        return [ex[0] for ex in examples_with_diff[:n]]
    
    def select_by_keywords(self, query: str, n: int = 3) -> List[Tuple[str, str]]:
        """Select examples with overlapping keywords."""
        query_words = set(query.lower().split())
        
        # Calculate keyword overlap for each example
        examples_with_score = []
        for ex in self.examples:
            ex_words = set(ex[0].lower().split())
            overlap = len(query_words & ex_words)
            examples_with_score.append((ex, overlap))
        
        # Sort by overlap and take top n
        examples_with_score.sort(key=lambda x: x[1], reverse=True)
        return [ex[0] for ex in examples_with_score[:n]]
    
    def select_diverse(self, n: int = 3) -> List[Tuple[str, str]]:
        """Select diverse examples from the pool."""
        # Simple diversity: spread examples evenly across the pool
        if len(self.examples) <= n:
            return self.examples
        
        step = len(self.examples) / n
        indices = [int(i * step) for i in range(n)]
        return [self.examples[i] for i in indices]

# Example usage
sentiment_examples = [
    ("This product is amazing! Best purchase ever.", "Positive"),
    ("Terrible quality. Waste of money.", "Negative"),
    ("It's okay, nothing special.", "Neutral"),
    ("Love it! Exceeded my expectations.", "Positive"),
    ("Very disappointed. Does not work as advertised.", "Negative"),
    ("Average product. Does the job.", "Neutral"),
    ("Fantastic! Would definitely buy again.", "Positive"),
    ("Complete garbage. Broke after one use.", "Negative"),
]

selector = FewShotExampleSelector(sentiment_examples)

# Test query
query = "Really happy with this purchase! Works perfectly."

print("Dynamic Example Selection:")
print("=" * 60)
print(f"\nQuery: {query}")

# Select by length
length_examples = selector.select_by_length(query, n=3)
print("\nExamples selected by length similarity:")
for i, (input_ex, output_ex) in enumerate(length_examples, 1):
    print(f"{i}. Input: {input_ex}")
    print(f"   Output: {output_ex}")

# Select by keywords
keyword_examples = selector.select_by_keywords(query, n=3)
print("\nExamples selected by keyword overlap:")
for i, (input_ex, output_ex) in enumerate(keyword_examples, 1):
    print(f"{i}. Input: {input_ex}")
    print(f"   Output: {output_ex}")

# Select diverse
diverse_examples = selector.select_diverse(n=3)
print("\nDiverse example selection:")
for i, (input_ex, output_ex) in enumerate(diverse_examples, 1):
    print(f"{i}. Input: {input_ex}")
    print(f"   Output: {output_ex}")
```

---

## Part 5: Few-Shot vs Zero-Shot Comparison

### Exercise 5.1: Performance Comparison

**Task:** Compare accuracy of few-shot vs zero-shot approaches.

**Solution:**

```python
def zero_shot_classifier(text):
    """Classify text without examples."""
    prompt = f"""Classify the following text as Positive, Negative, or Neutral.

Text: "{text}"

Classification:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

def few_shot_classifier(text):
    """Classify text with examples."""
    prompt = f"""Classify text as Positive, Negative, or Neutral.

Examples:
"Amazing product! Love it!" → Positive
"Terrible experience." → Negative
"It's okay, nothing special." → Neutral
"Best purchase ever!" → Positive
"Complete waste of money." → Negative

Text: "{text}"
Classification:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

# Test dataset with ground truth
test_cases = [
    ("Absolutely love this! Best thing I've bought.", "Positive"),
    ("Don't waste your money on this.", "Negative"),
    ("It's fine. Does what it says.", "Neutral"),
    ("Exceeded all my expectations!", "Positive"),
    ("Very disappointed with the quality.", "Negative"),
    ("Pretty good overall, happy with it.", "Positive"),
    ("Not bad, but not great either.", "Neutral"),
    ("This is garbage. Broke immediately.", "Negative"),
]

print("Zero-Shot vs Few-Shot Comparison:")
print("=" * 80)

zero_shot_correct = 0
few_shot_correct = 0

for text, true_label in test_cases:
    zero_pred = zero_shot_classifier(text)
    few_pred = few_shot_classifier(text)
    
    zero_correct = zero_pred.lower() == true_label.lower()
    few_correct = few_pred.lower() == true_label.lower()
    
    zero_shot_correct += zero_correct
    few_shot_correct += few_correct
    
    print(f"\nText: {text}")
    print(f"True Label: {true_label}")
    print(f"Zero-Shot: {zero_pred} {'✓' if zero_correct else '✗'}")
    print(f"Few-Shot: {few_pred} {'✓' if few_correct else '✗'}")

print("\n" + "=" * 80)
print(f"Zero-Shot Accuracy: {zero_shot_correct}/{len(test_cases)} ({zero_shot_correct/len(test_cases)*100:.1f}%)")
print(f"Few-Shot Accuracy: {few_shot_correct}/{len(test_cases)} ({few_shot_correct/len(test_cases)*100:.1f}%)")
```

**Expected Results:**
- Few-shot typically achieves 90-100% accuracy on these clear cases
- Zero-shot achieves 70-90% accuracy, may struggle with nuanced cases
- Few-shot is more consistent across similar test cases
- Zero-shot may interpret edge cases differently

---

## Key Takeaways

### When to Use Few-Shot Learning ✅

1. **Classification tasks** - Especially with multiple categories
2. **Style consistency** - Need outputs in specific format/tone
3. **Domain-specific tasks** - Industry jargon or specialized formats
4. **Edge case handling** - Show how to handle unusual inputs
5. **Format specification** - Complex output structures (JSON, tables, etc.)

### How Many Examples to Use 📊

- **Simple tasks:** 2-3 examples
- **Classification (few categories):** 3-5 examples
- **Complex generation:** 3-4 examples
- **Many categories:** 1-2 per category minimum
- **Rule of thumb:** More examples = more consistency, but diminishing returns after 5-7

### Example Selection Best Practices 🎯

1. **Representative** - Cover common variations
2. **Diverse** - Show different aspects of the task
3. **Balanced** - Equal representation of categories
4. **Clear** - Unambiguous inputs and outputs
5. **Realistic** - Match actual use case data
6. **Quality** - Perfect examples, not edge cases
7. **Concise** - Short enough to fit in context window

---

## Common Pitfalls to Avoid ⚠️

1. **Too many examples** - Wastes tokens, may confuse model
2. **Biased examples** - All positive, all one category, etc.
3. **Inconsistent format** - Examples don't follow same pattern
4. **Poor quality examples** - Contains errors model might copy
5. **Irrelevant examples** - Don't match the actual task
6. **Too similar** - All examples too close, not diverse enough
7. **Missing edge cases** - Don't show how to handle unusual inputs

---

## Next Steps

1. **Complete Lab 3** - Chain-of-Thought Prompting
2. **Build example libraries** - Create reusable example sets
3. **Experiment with hybrid approaches** - Combine few-shot with instructions
4. **Test on your data** - Apply to real-world tasks
5. **Measure performance** - Compare few-shot vs zero-shot on your use cases

---

## Additional Resources

- Research: "Language Models are Few-Shot Learners" (GPT-3 paper)
- [Few-Shot Learning Best Practices](../resources/prompt-cheatsheet.md)
- [Example Libraries](../resources/example-prompts.md)
- Practice datasets for classification and generation tasks
