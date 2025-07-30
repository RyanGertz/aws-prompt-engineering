from typing import List, Literal
from pydantic import BaseModel, Field
from utils import create_bedrock_client, retry_with_backoff


class MathSolution(BaseModel):
    """Model for step-by-step math problem solving"""

    problem: str = Field(description="The original problem")
    steps: List[str] = Field(description="Step-by-step solution process")
    final_answer: str = Field(description="Final numerical answer with units")
    confidence: Literal["Low", "Medium", "High"] = Field(
        description="Confidence in solution"
    )


@retry_with_backoff
def chain_of_thought_example():
    """
    Example 3: Chain of thought reasoning for problem solving
    Shows how to get step-by-step reasoning from the model
    """
    print("\n=== Example 3: Chain of Thought Reasoning ===")

    client = create_bedrock_client()

    math_problem = """
    A company's revenue increased by 15% in Q1, then decreased by 8% in Q2, 
    and finally increased by 12% in Q3. If their starting revenue was $500,000, 
    what was their revenue at the end of Q3?
    """

    # Prompt explicitly asking for step-by-step thinking
    prompt = f"""
    Solve this business math problem step by step. Show your reasoning clearly:
    
    Problem: {math_problem}
    
    Think through this systematically:
    1. Identify what we know
    2. Calculate each quarter's change
    3. Show the math for each step
    4. Arrive at the final answer
    
    Be precise with calculations and show intermediate results.
    """

    try:
        result = client.chat.completions.create(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.1,  # Low temperature for mathematical accuracy
            response_model=MathSolution,
        )

        print(f"Problem: {result.problem}")
        print("\nSolution Steps:")
        for i, step in enumerate(result.steps, 1):
            print(f"{i}. {step}")
        print(f"\nFinal Answer: {result.final_answer}")
        print(f"Confidence: {result.confidence}")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    chain_of_thought_example()
