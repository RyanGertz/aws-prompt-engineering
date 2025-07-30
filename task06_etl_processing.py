from typing import List, Optional
from pydantic import BaseModel, Field
from utils import create_bedrock_client, retry_with_backoff, extract_text_from_pdf


class Supervisor(BaseModel):
    name: str = Field(description="Name of the supervisor")
    district: str = Field(description="District the supervisor represents")


class SocialServicesItem(BaseModel):
    item_number: str = Field(description="The item number on the agenda")
    title_or_summary: str = Field(description="Short summary or title of the item")
    districts: Optional[List[str]] = Field(
        description="List of districts mentioned, if specified"
    )
    type: str = Field(
        description="Type of agenda item, like 'Consent', 'Public Hearing', etc."
    )


class ProcessedAgenda(BaseModel):
    meeting_title: str = Field(description="The official meeting title")
    date: str = Field(
        description="Date of the meeting, in YYYY-MM-DD format if possible"
    )
    location: str = Field(description="Where the meeting was held")
    supervisors: List[Supervisor] = Field(
        description="List of supervisors with name and district"
    )
    all_section_titles: List[str] = Field(
        description="List of all section titles in the agenda"
    )
    social_services_items: List[SocialServicesItem] = Field(
        description="Detailed agenda items related to social services"
    )


@retry_with_backoff
def etl_processing_example():
    """
    Example 5: ETL (Extract, Transform, Load) processing
    Shows how to process messy customer feedback into structured data
    """
    print("\n=== Example 5: ETL Data Processing ===")

    client = create_bedrock_client()

    # Raw, unstructured customer feedback (simulate real messy data)
    pdf_text = extract_text_from_pdf("Board-of-Supervisors-Agenda.pdf")

    # ETL prompt for batch processing
    prompt = f"""
    Please analyze the following document text and extract key information into a structured JSON format.

    Return an object with the following fields:

    - meeting_title
    - date
    - location
    - supervisors (list of names and districts)

    - all_section_titles: a list of all agenda section titles in the document

    - social_services_items: a list of detailed items specifically related to Social Services. These may appear in a section titled "Social Services" or be items that address social services topics such as welfare, benefits, homelessness, housing support, child services, etc.

    Each item in social_services_items should strictly include:
      - item_number
      - title or summary
      - districts (if specified)
      - type (e.g. "Consent", "Public Hearing", "Presentation", etc.)
    do not include anything else for an item. 

    Return only valid JSON. Do not include any explanatory text.

    Document text:
    {pdf_text}
    """

    try:
        result = client.chat.completions.create(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.5,  # Balanced temperature for processing
            response_model=ProcessedAgenda,
        )

        print("-" * 60)

        with open("processed_etl_agenda.json", "w") as f:
            f.write(result.model_dump_json(indent=2))

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    etl_processing_example()
