import pytest
from src.memory import ConversationMemory

SYSTEM_PROMPT = "You are a helpful assistant."


def test_initial_memory_has_system_prompt():
    mem = ConversationMemory(SYSTEM_PROMPT)
    messages = mem.to_list()
    assert len(messages) == 1
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_add_returns_new_instance():
    mem = ConversationMemory(SYSTEM_PROMPT)
    mem2 = mem.add("user", "Hello")
    assert mem is not mem2


def test_add_does_not_mutate_original():
    mem = ConversationMemory(SYSTEM_PROMPT)
    mem.add("user", "Hello")
    assert len(mem.to_list()) == 1


def test_add_appends_message():
    mem = ConversationMemory(SYSTEM_PROMPT)
    mem2 = mem.add("user", "Hello")
    assert len(mem2.to_list()) == 2
    assert mem2.to_list()[-1] == {"role": "user", "content": "Hello"}


def test_clear_returns_empty_memory_with_system_prompt():
    mem = ConversationMemory(SYSTEM_PROMPT)
    mem2 = mem.add("user", "Hello").add("assistant", "Hi")
    cleared = mem2.clear()
    assert len(cleared.to_list()) == 1
    assert cleared.to_list()[0]["role"] == "system"


def test_token_count_is_int():
    mem = ConversationMemory(SYSTEM_PROMPT)
    mem2 = mem.add("user", "Hello world")
    assert isinstance(mem2.token_count(), int)
    assert mem2.token_count() > 0


def test_token_count_caps_at_limit():
    long_content = "word " * 200000
    mem = ConversationMemory(SYSTEM_PROMPT)
    mem2 = mem.add("user", long_content)
    assert mem2.token_count() <= 32768
