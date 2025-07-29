# Prompt Engineering Advanced Techniques
*AI Summer Camp - Advanced Session*

## Welcome! 👋

This guide covers advanced prompt engineering techniques that will help you get better, more reliable results from AI models. Whether you're working with OpenAI's GPT models, Anthropic's Claude, or other LLMs, these techniques will make you a more effective AI user.

## What You'll Learn

By the end of this session, you'll understand how to make AI models work better for you through smart prompting techniques, parameter tuning, and structured outputs. Think of this as learning to "speak AI" more fluently!

## 🎯 Getting Started

Before diving in, make sure you have:
- Access to an AI model API (OpenAI, Anthropic, etc.)
- Basic Python knowledge (we'll show you the code, don't worry!)
- Curiosity and willingness to experiment

---

## 🎛️ Understanding Model Parameters

Before we dive into prompting, let's understand the "knobs and dials" that control how AI models behave.

### Temperature (0-1)
- **Low (0-0.3)**: Predictable, focused responses
- **High (0.7-1.0)**: Creative, varied responses
- **Example**: Ask "What's a good pizza topping?" at temp 0.1 vs 0.9

### Top P (Nucleus Sampling)
Controls which words the model considers based on probability.
- **Example**: "Today I went to the [park: 50%, store: 30%, moon: 10%, library: 10%]"
- **Top P = 0.8**: Only considers "park" and "store" (80% combined)
- **Lower values = more focused responses**

### Top K
Simply limits to the K most likely next words.
- **Top K = 5**: Only considers the 5 most probable words
- **Use either Top P OR Top K, not both**

### Other Useful Parameters
- **Max Length**: Stop after X tokens
- **Stop Sequences**: Stop when model outputs specific text (like "END" or "---")

---

## 📝 Elements of a Good Prompt

Every effective prompt has four key components:

### 1. Instruction
What you want the model to do.
```
"Summarize the following text in 2 sentences."
```

### 2. Context
Background information that helps the model understand.
```
"You are a financial advisor explaining to a college student."
```

### 3. Input Data
The actual content to work with.
```
"Text: [your article here]"
```

### 4. Output Indicator
Format you want back.
```
"Format: Bullet points with key takeaways"
```

### Pro Tips for Better Prompts
- **Be specific and clear** - vague prompts = vague results
- **Use separators** like `---` or `<input>` to organize sections
- **Put instructions first** - models pay more attention to the beginning
- **Focus on what TO DO** instead of what not to do
- **One task per prompt** - break complex requests into steps

---

## 🚀 Advanced Prompting Techniques

### Few-Shot Prompting
Give the model examples to learn from:

```
Classify these movie reviews as positive or negative:

Review: "Amazing cinematography and great acting!" 
Sentiment: Positive

Review: "Boring plot, terrible dialogue."
Sentiment: Negative

Review: "The special effects were incredible but the story dragged."
Sentiment: [model completes this]
```

### Chain of Thought Prompting
Ask the model to "think step by step":

```
Problem: If a shirt costs $25 and is on sale for 20% off, what's the final price?

Think through this step by step:
1. Calculate the discount amount
2. Subtract from original price
3. State the final answer
```

### Prompt Chaining
Break complex tasks into smaller steps, where each prompt builds on the previous output.

**Step 1**: "List the main characters in this story"
**Step 2**: "For each character from the previous list, describe their motivation"
**Step 3**: "Identify conflicts between these characters based on their motivations"

### Tree of Thought
Let the model explore multiple approaches and pick the best one:

```
You need to solve this problem. Consider 3 different approaches:

Approach A: [method 1]
Approach B: [method 2] 
Approach C: [method 3]

Evaluate each approach and choose the best one, explaining why.
```

### RAG (Retrieval-Augmented Generation)
Combine external knowledge with the model's built-in knowledge. You retrieve relevant information first, then include it in your prompt as context.

---

## 🎯 Model Choice

Choosing the right model is crucial for balancing performance, cost, and speed:

### Model Categories
- **Fast & Cheap**: GPT-3.5 Turbo, Claude Haiku, Llama 3.1 8B
  - Great for: Simple tasks, classification, data extraction
  - Use when: Processing large volumes, tight budgets
  
- **Balanced**: GPT-4, Claude Sonnet
  - Great for: Most complex tasks, analysis, creative writing
  - Use when: Quality matters more than speed
  
- **Premium**: GPT-4 Turbo, Claude Opus, O1-Preview
  - Great for: Complex reasoning, research, high-stakes tasks
  - Use when: You need the absolute best results

### Specialized Models
- **Code**: GitHub Copilot, CodeLlama, GPT-4 (code-optimized)
- **Embeddings**: text-embedding-ada-002, sentence-transformers
- **Image**: DALL-E, Midjourney, Stable Diffusion

## 📊 Classification Techniques

Classification is one of the most practical applications of prompt engineering:

### Basic Classification
```
Task: Classify customer feedback sentiment

Examples:
"Amazing product, love it!" → Positive
"Terrible quality, waste of money" → Negative
"It's okay, nothing special" → Neutral

Your turn:
"The delivery was fast but the product broke immediately" → ?
```

### Multi-Label Classification
```
Classify this email into categories (can be multiple):
[Urgent, Technical Issue, Billing, Feature Request, Complaint]

Email: "Hi, my subscription was charged twice this month and now the app won't load. This is really frustrating as I need this for work tomorrow."

Categories: Urgent, Technical Issue, Billing, Complaint
```

### Advanced Classification with Confidence
```
Classify the following with confidence scores (0-100):

Text: "I think the new feature might be useful, though I haven't tried it yet."

Classification:
- Positive: 60%
- Negative: 10% 
- Neutral: 30%
```

## 🚩 Using Flags for Better Control

Flags help you control model behavior and get more consistent outputs:

### Content Flags
```python
prompt = """
[FLAG: FAMILY_FRIENDLY] 
[FLAG: FORMAL_TONE]
[FLAG: TECHNICAL_LEVEL_BEGINNER]

Explain how machine learning works.
"""
```

### Processing Flags
```python
prompt = """
[FLAG: STRICT_JSON_OUTPUT]
[FLAG: NO_EXPLANATIONS]
[FLAG: VALIDATE_BEFORE_RESPONSE]

Extract person data: "John Smith, age 25, works as a teacher in Boston"
"""
```

### Custom Behavior Flags
```python
system_prompt = """
You are an AI assistant. Follow these flags:
- [CONCISE]: Keep responses under 100 words
- [EXAMPLES]: Always provide examples
- [CITE_SOURCES]: Include source references when possible
- [STEP_BY_STEP]: Break down complex topics
"""
```

## 🎲 Synthetic Data Generation

Use AI to create training data, test cases, and examples:

### Generating Training Data
```python
prompt = """
Generate 10 customer service conversations about billing issues.
Include realistic customer concerns and helpful agent responses.

Format each as:
Customer: [complaint]
Agent: [helpful response]
Outcome: [Resolved/Escalated]
"""
```

### Creating Test Cases
```python
prompt = """
Generate edge cases for testing a username validation system.
Include: special characters, different lengths, Unicode, spaces, numbers

Format:
Username: [test_case]
Expected: [Valid/Invalid]
Reason: [explanation]
"""
```

### Synthetic Dataset with Structure
```python
from pydantic import BaseModel
from typing import List

class SyntheticUser(BaseModel):
    name: str
    age: int
    occupation: str
    interests: List[str]
    location: str

# Generate diverse, realistic user profiles
prompt = """
Create a realistic user profile for a fitness app.
Include diverse demographics, occupations, and interests.
"""
```

## ⚙️ Parsing and ETL (Extract, Transform, Load)

Transform messy, unstructured data into clean, usable formats:

### Email Parsing
```python
class EmailData(BaseModel):
    sender: str
    subject: str
    priority: str  # High, Medium, Low
    action_required: bool
    deadline: Optional[str]
    categories: List[str]

# Extract structured data from email text
prompt = """
Parse this email and extract key information:
[email content here]
"""
```

### Document Processing Pipeline
```python
# Step 1: Extract text sections
extract_prompt = """
Extract these sections from the document:
- Executive Summary
- Key Findings  
- Recommendations
- Budget Information
"""

# Step 2: Transform format
transform_prompt = """
Convert this text into structured bullet points:
- Each point should be actionable
- Include relevant numbers/dates
- Prioritize by importance
"""

# Step 3: Load into database format
load_prompt = """
Format this data for database insertion:
Table: project_reports
Columns: title, summary, findings, budget, priority
"""
```

### Web Scraping Data Cleanup
```python
class ProductInfo(BaseModel):
    name: str
    price: float
    rating: float
    features: List[str]
    availability: str

# Clean messy scraped product data
prompt = """
Clean and structure this scraped product data:
[messy HTML/text content]

Extract: name, price, rating, key features, stock status
Handle: missing values, price variations, rating formats
"""
```

### Real-World ETL Example
```python
# Processing customer feedback from multiple sources
class FeedbackRecord(BaseModel):
    source: str  # email, survey, social_media
    sentiment: str
    topic: str
    urgency: int  # 1-5 scale
    customer_tier: str
    resolution_needed: bool

# Pipeline for processing hundreds of feedback items
def process_feedback_batch(raw_feedback_list):
    processed_records = []
    
    for raw_text in raw_feedback_list:
        structured = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=FeedbackRecord,
            messages=[{
                "role": "user", 
                "content": f"Process this customer feedback: {raw_text}"
            }]
        )
        processed_records.append(structured)
    
    return processed_records
```

---

## 🐍 Structured Output with Python's Instructor Library

The `instructor` library helps you get reliable, structured data from AI models instead of just text.

### Basic Setup
```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Wrap your OpenAI client
client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Get structured output
person = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Person,
    messages=[
        {"role": "user", "content": "Extract info: John is 25 and works as a teacher"}
    ]
)

print(person.name)  # "John"
print(person.age)   # 25
```

### Why This Matters
- **Reliable data extraction** from messy text
- **Perfect for ETL pipelines** (Extract, Transform, Load)
- **Synthetic data generation** with guaranteed structure
- **No more parsing headaches** with inconsistent text formats

### Real-World Example: Processing Survey Responses
```python
class SurveyResponse(BaseModel):
    satisfaction_score: int  # 1-5 scale
    main_complaint: str
    would_recommend: bool

# Process hundreds of text responses into clean data
responses = []
for text_response in survey_texts:
    structured = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=SurveyResponse,
        messages=[{"role": "user", "content": f"Analyze: {text_response}"}]
    )
    responses.append(structured)
```

---

## 💡 Key Takeaways

1. **Experiment with parameters** - small changes can dramatically improve results
2. **Structure your prompts** with clear instructions, context, input, and output format
3. **Use examples** - few-shot prompting is incredibly powerful
4. **Break down complex tasks** - prompt chaining prevents confusion
5. **Get structured data** - libraries like `instructor` turn AI into reliable data processors

## 🛠️ Try It Yourself

Your challenge: Create a prompt that:
1. Uses few-shot examples
2. Includes clear output formatting
3. Solves a real problem you care about

Remember: The best prompt engineers are made through experimentation, not perfection on the first try!

## 📚 Additional Resources

Want to dive deeper? Check out these resources:
- [Instructor Documentation](https://python.useinstructor.com/) - Complete guide to structured outputs
- [OpenAI API Documentation](https://platform.openai.com/docs) - Official API reference
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) - Comprehensive prompting strategies

## 🤝 Contributing

Found an error or have a suggestion? Feel free to open an issue or submit a pull request to improve this guide for future students!