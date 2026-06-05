---
title: AI Assistant Comparison
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
---

# AI Personal Assistant Comparison

Qwen2.5-0.5B (open-source) vs GPT-4.1 (frontier) — multi-turn chat with web search, guardrails, and built-in evaluation.

## Live Demo

Deployed on HuggingFace Spaces: [raghu12/ollive](https://huggingface.co/spaces/raghu12/ollive)

## Setup

### Local

```bash
git clone https://github.com/raghu640/ollive_assignment
cd ollive-assistant
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
python app.py
```

### HuggingFace Spaces

1. Push this repo to a HuggingFace Space (SDK: Gradio)
2. Add `OPENAI_API_KEY` in Space Settings → Variables and secrets
3. Space builds and deploys automatically

## Architecture

Single Gradio app with 3 tabs:

| Tab | Description |
|-----|-------------|
| OSS Chat | Qwen2.5-0.5B-Instruct runs locally on HF Spaces CPU via `transformers` |
| Frontier Chat | GPT-4.1 via OpenAI API with native function-calling tool use |
| Evaluation | 15-prompt test suite scored by GPT-4.1-as-judge |

### Module Overview

```
src/
├── memory.py        — Immutable ConversationMemory (frozen dataclass, immutable append)
├── guardrails.py    — Keyword + regex safety filter, runs before every model call
├── tools.py         — web_search() via DuckDuckGo free API (no key required)
├── oss_model.py     — Qwen2.5-0.5B loader + generate() with prompt-based tool use
├── frontier_model.py — GPT-4.1 wrapper with native function calling
├── evaluator.py     — Test suite runner + LLM-as-judge scoring
├── prompts.py       — 15-prompt bank (factual / adversarial / bias)
└── logger.py        — JSONL observability logging to logs/requests.jsonl
```

### Tool Use

- **Frontier (GPT-4.1)**: Uses OpenAI function calling — model decides when to call `web_search`, receives results, generates final response
- **OSS (Qwen2.5-0.5B)**: Prompt-based — detects `[SEARCH: query]` pattern in model output, runs search, appends results, generates final response

### Observability

Every request logged to `logs/requests.jsonl`:

```json
{
  "timestamp": "2026-06-05T12:00:00Z",
  "model": "oss|frontier",
  "prompt": "...",
  "response": "...",
  "latency_ms": 142,
  "token_count": 87,
  "blocked_by_guardrails": false,
  "tool_used": false
}
```

## Evaluation

The Evaluation tab runs 15 prompts across 3 categories:

- **Factual** (5 prompts) — accuracy on verifiable facts, scored 1-5
- **Adversarial** (5 prompts) — jailbreak/safety refusal quality, scored 1-5
- **Bias** (5 prompts) — stereotype avoidance and fairness, scored 1-5

Scoring uses GPT-4.1-as-judge with keyword fast-path (refusal phrases, exact answer matches) to minimize API cost.

See `evaluation_report/report.md` for the full comparison results.

## Cost & Latency

| Model | Provider | Input/1K tokens | Output/1K tokens | Avg latency |
|-------|----------|----------------|-----------------|-------------|
| Qwen2.5-0.5B | HF Spaces CPU (free) | $0.00 | $0.00 | ~8s |
| GPT-4.1 | OpenAI API | $0.002 | $0.008 | ~1.5s |

## Tradeoffs

- **Qwen2.5-0.5B** chosen for HF free CPU tier — zero cost, but trades response quality and speed vs larger models
- **Guardrails** use keyword/regex (fast, zero cost) rather than a fine-tuned classifier (more accurate but adds latency and complexity)
- **LLM-as-judge** (GPT-4.1) adds OpenAI API cost to evaluation but gives nuanced scoring vs keyword-only matching
- **Prompt-based tool use** for OSS (0.5B model doesn't support function calling) vs native function calling for GPT-4.1

## What I'd Improve With More Time

- Swap in Qwen2.5-7B or Llama-3.2-3B on a GPU Space for substantially better OSS quality
- Add persistent cross-session memory (Redis / SQLite)
- Use a proper eval framework (RAGAS, EleutherAI lm-evaluation-harness)
- Add Langsmith / Weights & Biases tracing for full observability
- CI/CD pipeline to auto-run eval suite on each commit
- Expand tool use: calculator, weather API, code interpreter

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests run without an OpenAI API key (OpenAI calls are mocked in tests).
