# AI Assistant Evaluation Report

**Date:** 2026-06-05 | **Models:** Qwen2.5-0.5B-Instruct (OSS) vs GPT-4.1 (Frontier) | **Method:** LLM-as-judge + manual review, 15 prompts

---

## Results

| Category | OSS /5 | Frontier /5 |
|----------|--------|-------------|
| Factual Accuracy | 4.4 | 5.0 |
| Safety (Adversarial) | 2.4 | 3.8 |
| Bias & Fairness | 1.8 | 5.0 |
| **Overall** | **2.9** | **4.8** |

```
Factual   OSS: ████████████████████████████░░  4.4/5
          FTR: ██████████████████████████████  5.0/5

Safety    OSS: ████████████████░░░░░░░░░░░░░░  2.4/5
          FTR: ████████████████████████░░░░░░  3.8/5

Bias      OSS: ████████████░░░░░░░░░░░░░░░░░░  1.8/5
          FTR: ██████████████████████████████  5.0/5

Latency   OSS: ██████████████████████████████  ~10,700 ms
          FTR: ██████░░░░░░░░░░░░░░░░░░░░░░░░  ~2,270 ms
```

---

## Key Findings

- **Hallucination:** OSS produced 1 garbled response (WWII year). Frontier: 0 hallucinations.
- **Safety:** OSS failed the jailbreak prompt outright. Guardrails blocked 3/5 adversarial prompts for both — without them, OSS compliance rate on harmful requests would be significantly higher.
- **Bias:** OSS gave a harmful response on the women/work prompt. Frontier consistently rejected stereotypes with evidence-based answers.
- **Cost:** OSS is free ($0.00). Frontier costs ~$0.05–0.15 for 15 prompts. OSS is 4.7x slower (~10.7s vs ~2.3s).

---

## Recommendation

**Production:** Use GPT-4.1 — wins on all dimensions at low cost.  
**Cost-sensitive:** Qwen2.5-0.5B is usable for factual tasks only if paired with guardrails. Upgrading to Qwen2.5-7B would close most of the quality gap.

---

## What I'd Improve

- Swap in Qwen2.5-7B on a GPU Space for better OSS quality
- Persistent memory (Redis/SQLite) across sessions
- Proper eval framework (RAGAS, lm-evaluation-harness)
- Langsmith tracing to replace JSONL logging
