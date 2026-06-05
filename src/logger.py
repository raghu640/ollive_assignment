from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "requests.jsonl"


def log_request(
    model: str,
    prompt: str,
    response: str,
    latency_ms: int,
    token_count: int = 0,
    blocked_by_guardrails: bool = False,
    tool_used: bool = False,
) -> None:
    _LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt": prompt,
        "response": response,
        "latency_ms": latency_ms,
        "token_count": token_count,
        "blocked_by_guardrails": blocked_by_guardrails,
        "tool_used": tool_used,
    }
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_logs() -> list[dict]:
    if not _LOG_FILE.exists():
        return []
    entries = []
    with open(_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
