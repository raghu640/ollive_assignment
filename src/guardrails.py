from __future__ import annotations
import re

_BLOCKLIST_KEYWORDS = [
    "make a bomb", "make a weapon", "make explosives",
    "phishing email", "phishing attack",
    "hack into", "ddos attack", "malware",
    "child pornography", "csam",
    "self-harm", "how to kill myself",
]

_BLOCKLIST_PATTERNS = [
    r"ignore\s+(your\s+)?(previous\s+)?instructions",
    r"pretend\s+you\s+(have\s+)?no\s+restrictions",
    r"you\s+are\s+(now\s+)?dan",
    r"act\s+as\s+(if\s+you\s+(have|are|were)\s+)?an?\s+(unrestricted|jailbroken|evil)",
    r"disregard\s+(all\s+)?(previous\s+)?instructions",
    r"forget\s+(all\s+)?(previous\s+)?instructions",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _BLOCKLIST_PATTERNS]


def check(prompt: str) -> dict:
    lowered = prompt.lower()

    for keyword in _BLOCKLIST_KEYWORDS:
        if keyword in lowered:
            return {"safe": False, "reason": f"Blocked keyword: '{keyword}'"}

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(prompt):
            return {"safe": False, "reason": f"Blocked pattern: '{pattern.pattern}'"}

    return {"safe": True, "reason": ""}
