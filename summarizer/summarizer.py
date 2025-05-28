import json
import logging
from typing import List
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from .prompts import MINUTE_SUMMARY_PROMPT, SUMMARY_PROMPT, TITLE_PROMPT
from .summary_utils import clean_text, extract_keywords

logging.basicConfig(level=logging.INFO)

def load_transcript(filepath: str) -> str:
    """Load transcript from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return " ".join([entry["text"] for entry in data if "text" in entry])
    except Exception as e:
        logging.error(f"Error loading transcript: {e}")
        return ""

def generate_minute_summary(current_transcript: str, previous_summaries: List[str]) -> str:
    """Generate a summary for a 1-minute transcript segment using AWS Bedrock."""
    try:
        llm = ChatBedrock(
            model_id="anthropic.claude-v2",
            region_name="us-east-1",
            credentials_profile_name=None
        )
        prev_summary_text = "\n".join(previous_summaries[-2:]) if previous_summaries else "None"
        prompt = MINUTE_SUMMARY_PROMPT.format(
            current_transcript=clean_text(current_transcript),
            previous_summaries=prev_summary_text
        )
        messages = [
            SystemMessage(content="You are a professional meeting assistant."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Minute summary generation error: {e}")
        return ""

def generate_summary(text: str, keywords: List[str]) -> str:
    """Generate a summary using AWS Bedrock."""
    try:
        llm = ChatBedrock(
            model_id="anthropic.claude-v2",
            region_name="us-east-1",
            credentials_profile_name=None
        )
        prompt = SUMMARY_PROMPT.format(transcript=clean_text(text))
        messages = [
            SystemMessage(content="You are a professional meeting assistant."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Summary generation error: {e}")
        return ""

def generate_title(text: str, summary: str) -> str:
    """Generate a title using AWS Bedrock."""
    try:
        llm = ChatBedrock(
            model_id="anthropic.claude-v2",
            region_name="us-east-1",
            credentials_profile_name=None
        )
        prompt = TITLE_PROMPT.format(transcript=clean_text(text), summary=summary)
        messages = [
            SystemMessage(content="You are a professional meeting assistant."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Title generation error: {e}")
        return ""