import os
import json
from typing import List
from pydantic import BaseModel, ValidationError
from openai import AzureOpenAI



from langfuse.callback import CallbackHandler
callback_handler = CallbackHandler()



client = AzureOpenAI(
    deployment_name="myllm", 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
    callbacks=[callback_handler])


class MeetingNotes(BaseModel):
    summary: str
    action_items: List[str]


PROMPT_TEMPLATE = """You are a meeting assistant.
1. Summarize the meeting transcript below in exactly two sentences.
2. Then list all action items mentioned, each as a separate bullet
beginning with a dash.
Return the result strictly as JSON with keys "summary" and "action_items".
Transcript:
{transcript}
"""

def gpt_model(prompt: str, retry=False) -> str:

    messages = [
        {"role": "system", "content": "Please output valid JSON only."} if retry else {},
        {"role": "user", "content": prompt}
    ]

    messages = [msg for msg in messages if msg]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-4o-mini"),
        temperature=0,
        messages=messages
    )

    return response.choices[0].message.content

def extract_meeting_notes(transcript: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(transcript=transcript)
    for attempt in range(2):
        reponse_content = gpt_model(prompt, retry=(attempt == 1))
        try:
            data = json.loads(reponse_content)
            validated = MeetingNotes(**data)
            return validated.dict()
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == 1:
                raise ValueError(f"Failed after retry. Last response:\n{reponse_content}") from e
    return {}


if __name__ == "__main__":
    test_transcript = """Alice: Welcome everyone. Today we need to finalize the Q3
        roadmap.
        Bob: I've emailed the updated feature list—please review by
        Friday.
        Carol: I'll set up the user-testing sessions next week.
        Dan: Let's push the new UI mockups to staging on Wednesday.
        Alice: Great. Also, can someone compile the stakeholder
        feedback into a slide deck?
        Bob: I can handle the slide deck by Monday.
        Alice: Thanks, team. Meeting adjourned."""
    
    result = extract_meeting_notes(test_transcript)
    print(json.dumps(result, indent=2))
    
