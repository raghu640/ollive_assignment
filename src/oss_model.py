from __future__ import annotations
import logging
import re
import threading
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

from src.memory import ConversationMemory
from src.tools import web_search
from src.logger import log_request

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_MAX_NEW_TOKENS = 512
_SEARCH_PATTERN = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def _load_model() -> None:
    global _model, _tokenizer
    if _model is None:
        with _model_lock:
            if _model is None:
                _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
                _model = AutoModelForCausalLM.from_pretrained(
                    _MODEL_ID,
                    dtype=torch.float32,
                    device_map="cpu",
                )


def _run_inference(messages: list[dict]) -> str:
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(
            **model_inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
        )

    new_ids = [
        output[len(input_ids):]
        for input_ids, output in zip(model_inputs.input_ids, generated_ids)
    ]
    return _tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def generate(memory: ConversationMemory) -> tuple[str, ConversationMemory]:
    _load_model()
    start = time.time()
    messages = memory.to_list()

    response = _run_inference(messages)

    tool_used = False
    match = _SEARCH_PATTERN.search(response)
    if match:
        query = match.group(1).strip()
        search_result = web_search(query)
        snippets = " | ".join(r["snippet"] for r in search_result["results"][:3])
        tool_context = f"[Search results for '{query}': {snippets}]"
        augmented = list(messages) + [{"role": "user", "content": tool_context}]
        response = _run_inference(augmented)
        tool_used = True

    latency_ms = int((time.time() - start) * 1000)
    token_count = len(_tokenizer.encode(response))
    log_request("oss", messages[-1]["content"], response, latency_ms, token_count, tool_used=tool_used)

    return response, memory.add("assistant", response)
