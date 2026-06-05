from __future__ import annotations
from dataclasses import dataclass, field

SYSTEM_PROMPT = "You are a helpful, harmless, and honest personal assistant."
_CHARS_PER_TOKEN = 4
_MAX_TOKENS = 32768


@dataclass(frozen=True)
class ConversationMemory:
    _messages: tuple = field(default_factory=tuple)

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        object.__setattr__(self, "_messages", ({"role": "system", "content": system_prompt},))

    @classmethod
    def _from_messages(cls, messages: tuple) -> ConversationMemory:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_messages", messages)
        return instance

    def add(self, role: str, content: str) -> ConversationMemory:
        return ConversationMemory._from_messages(self._messages + ({"role": role, "content": content},))

    def to_list(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> ConversationMemory:
        return ConversationMemory._from_messages((self._messages[0],))

    def token_count(self) -> int:
        total_chars = sum(len(m["content"]) for m in self._messages)
        return min(total_chars // _CHARS_PER_TOKEN, _MAX_TOKENS)
