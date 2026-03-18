import os
import logging
import traceback

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def call_gemini_llm(text, model="gemini-2.5-flash-lite"):
    try:
        response = client.chat.completions.create(
            model=model,
            reasoning_effort="low",
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(traceback.format_exc())
        return ""

def call_openai_llm(text, model="gpt-5.1"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(traceback.format_exc())
        return ""