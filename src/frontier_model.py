from __future__ import annotations
import json
import os
import time

from openai import OpenAI

from src.memory import ConversationMemory
from src.tools import web_search
from src.logger import log_request

_MODEL = "gpt-4.1"
_MAX_TOKENS = 512
_TEMPERATURE = 0.7

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
    if not api_key or api_key.strip() in ("", "sk-..."):
        raise ValueError("OPENAI_API_KEY is not set. Add it in Space Settings → Variables and secrets.")
    return OpenAI(api_key=api_key.strip())


def generate(memory: ConversationMemory) -> tuple[str, ConversationMemory]:
    client = _get_client()
    start = time.time()
    messages = memory.to_list()

    response = client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        tools=[_WEB_SEARCH_TOOL],
        tool_choice="auto",
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
    )

    tool_used = False
    message = response.choices[0].message

    if message.tool_calls:
        tool_used = True
        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        search_result = web_search(args["query"])
        snippets = " | ".join(r["snippet"] for r in search_result["results"][:3])

        messages_with_tool = messages + [
            message.model_dump(exclude_unset=True),
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Search results: {snippets}",
            },
        ]
        response = client.chat.completions.create(
            model=_MODEL,
            messages=messages_with_tool,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        message = response.choices[0].message

    text = message.content or ""
    latency_ms = int((time.time() - start) * 1000)
    token_count = response.usage.completion_tokens if response.usage else 0
    log_request("frontier", messages[-1]["content"], text, latency_ms, token_count, tool_used=tool_used)

    return text, memory.add("assistant", text)
