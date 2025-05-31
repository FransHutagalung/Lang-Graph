from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import requests

class OpenRouterChat(ChatOpenAI):
    @property
    def _llm_type(self):
        return "openrouter-chat"

    def _generate(self, messages, stop=None, **kwargs):
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "http://localhost",  # optional
            "X-Title": "LangGraph Test",         # optional
        }

        payload = {
            "model": "qwen/qwen3-30b-a3b",
            "messages": [m.dict() for m in messages],
            "temperature": 0.7,
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"OpenRouter error: {response.status_code} - {response.text}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content
