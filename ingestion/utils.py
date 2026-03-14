import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def call_llm(text):
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash-lite",
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
