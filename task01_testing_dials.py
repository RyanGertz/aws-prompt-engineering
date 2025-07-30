import boto3
from utils import retry_with_backoff


@retry_with_backoff
def test_dials_and_parameters():
    """
    Test different dials and parameters for the Bedrock model
    This is a simple test to ensure the model can handle various settings
    Please feel free to modify the prompt and parameters!!
    """
    print("\n=== Testing Dials and Parameters ===")

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    prompt = "Give me a short response to this question: What is the meaning of life?"

    try:
        result = client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "temperature": 0.7,  # temperature for creativity
                "topP": 1,  # Use top-p sampling
                "stopSequences": ["<END>"],
            },
        )

        print(f"Response: {result['output']['message']['content'][0]['text']}")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    test_dials_and_parameters()
