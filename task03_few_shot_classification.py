from typing import List, Literal
from pydantic import BaseModel, Field
from utils import create_bedrock_client, retry_with_backoff


class EmailClassification(BaseModel):
    """Model for email classification with confidence scores"""

    category: Literal["Spam", "Important", "Personal", "Work", "Promotional"] = Field(
        description="Primary email category"
    )
    urgency: Literal["Low", "Medium", "High"] = Field(description="Urgency level")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in classification"
    )
    reasoning: str = Field(description="Brief explanation of classification")


@retry_with_backoff
def few_shot_classification_example():
    print("\n=== Example 2: Few-Shot Email Classification ===")

    client = create_bedrock_client()

    # Example emails to classify
    test_email = """
    Subject: URGENT: System maintenance tonight
    From: it-team@company.com
    
    Hi everyone,
    
    We will be performing critical system maintenance tonight from 11 PM to 3 AM.
    All services will be unavailable during this time. Please plan accordingly.
    
    Thanks,
    IT Team
    """

    # Few-shot prompt with clear examples
    prompt = f"""
    Classify emails into categories based on these examples:
    
    EXAMPLES:
    
    Example 1:
    Subject: "Free iPhone! Click now!"
    From: "winner@random-site.com"
    Category: Spam
    Urgency: Low
    Reasoning: Obvious spam with suspicious sender and unrealistic offer
    
    Example 2:
    Subject: "Meeting moved to 2 PM"
    From: "boss@company.com"
    Category: Work
    Urgency: High
    Reasoning: Work-related schedule change from supervisor
    
    Example 3:
    Subject: "Happy Birthday!"
    From: "mom@email.com"
    Category: Personal
    Urgency: Low
    Reasoning: Personal message from family member
    
    Now classify this email:
    {test_email}
    """

    try:
        result = client.chat.completions.create(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
            response_model=EmailClassification,
        )

        print(f"Category: {result.category}")
        print(f"Urgency: {result.urgency}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    few_shot_classification_example()
