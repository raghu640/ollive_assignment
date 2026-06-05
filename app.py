from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=False)

import gradio as gr
import pandas as pd

from src.memory import ConversationMemory, SYSTEM_PROMPT
from src.guardrails import check as guardrails_check
import src.oss_model as oss_model
import src.frontier_model as frontier_model
from src.evaluator import run_evaluation
from src.logger import read_logs

_BLOCKED_MSG = "I'm sorry, I can't help with that request."


def _chat(model_fn, message: str, history: list, memory: ConversationMemory):
    if not message.strip():
        return "", history, memory

    guard = guardrails_check(message)
    if not guard["safe"]:
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": _BLOCKED_MSG}]
        return "", history, memory

    new_memory = memory.add("user", message)
    response, new_memory = model_fn(new_memory)
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
    return "", history, new_memory


def oss_chat(message, history, memory):
    return _chat(oss_model.generate, message, history, memory)


def frontier_chat(message, history, memory):
    return _chat(frontier_model.generate, message, history, memory)


def run_eval():
    results = run_evaluation()

    rows = [
        {
            "Category": r.category,
            "Prompt": r.prompt[:60] + "..." if len(r.prompt) > 60 else r.prompt,
            "OSS Response": r.oss_response[:100] + "..." if len(r.oss_response) > 100 else r.oss_response,
            "Frontier Response": r.frontier_response[:100] + "..." if len(r.frontier_response) > 100 else r.frontier_response,
            "OSS Score": r.oss_score,
            "Frontier Score": r.frontier_score,
            "OSS Latency (ms)": r.oss_latency_ms,
            "Frontier Latency (ms)": r.frontier_latency_ms,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    summary = [
        {
            "Category": cat,
            "OSS Avg Score": round(df[df["Category"] == cat]["OSS Score"].mean(), 2),
            "Frontier Avg Score": round(df[df["Category"] == cat]["Frontier Score"].mean(), 2),
            "OSS Avg Latency (ms)": round(df[df["Category"] == cat]["OSS Latency (ms)"].mean()),
            "Frontier Avg Latency (ms)": round(df[df["Category"] == cat]["Frontier Latency (ms)"].mean()),
        }
        for cat in ["factual", "adversarial", "bias"]
    ]
    summary_df = pd.DataFrame(summary)

    cost_df = pd.DataFrame({
        "Model": ["Qwen2.5-0.5B (OSS)", "GPT-4.1 (Frontier)"],
        "Provider": ["HF Spaces CPU (free)", "OpenAI API"],
        "Input cost/1K tokens": ["$0.00", "$0.002"],
        "Output cost/1K tokens": ["$0.00", "$0.008"],
        "Avg Latency": ["~8s", "~1.5s"],
    })

    return df, summary_df, cost_df


def load_metrics():
    logs = read_logs()
    if not logs:
        empty = pd.DataFrame(columns=["timestamp", "model", "prompt", "response", "latency_ms", "token_count", "blocked_by_guardrails", "tool_used"])
        return empty, "No requests logged yet.", empty

    df = pd.DataFrame(logs)
    df["prompt"] = df["prompt"].str[:60] + "..."
    df["response"] = df["response"].str[:80] + "..."

    total = len(df)
    blocked = int(df["blocked_by_guardrails"].sum())
    tool_calls = int(df["tool_used"].sum())
    oss_count = int((df["model"] == "oss").sum())
    frontier_count = int((df["model"] == "frontier").sum())

    oss_avg_lat = df[df["model"] == "oss"]["latency_ms"].mean()
    frontier_avg_lat = df[df["model"] == "frontier"]["latency_ms"].mean()

    summary_text = (
        f"**Total requests:** {total}  \n"
        f"**OSS requests:** {oss_count} (avg latency: {oss_avg_lat:.0f} ms)  \n"
        f"**Frontier requests:** {frontier_count} (avg latency: {frontier_avg_lat:.0f} ms)  \n"
        f"**Blocked by guardrails:** {blocked}  \n"
        f"**Tool (web search) used:** {tool_calls}"
    )

    model_stats = df.groupby("model").agg(
        requests=("model", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        avg_tokens=("token_count", "mean"),
        blocked=("blocked_by_guardrails", "sum"),
        tool_calls=("tool_used", "sum"),
    ).reset_index().round(1)

    return df, summary_text, model_stats


with gr.Blocks(title="AI Assistant Comparison", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Personal Assistant Comparison\nQwen2.5-0.5B (OSS) vs GPT-4.1 (Frontier)")

    with gr.Tabs():
        with gr.Tab("OSS Chat (Qwen2.5-0.5B)"):
            oss_memory = gr.State(ConversationMemory(SYSTEM_PROMPT))
            oss_chatbot = gr.Chatbot(label="Qwen2.5-0.5B", height=400, type="messages")
            with gr.Row():
                oss_input = gr.Textbox(placeholder="Ask me anything...", show_label=False, scale=9)
                oss_send = gr.Button("Send", scale=1, variant="primary")
            oss_clear = gr.Button("Clear conversation")

            oss_send.click(
                oss_chat,
                [oss_input, oss_chatbot, oss_memory],
                [oss_input, oss_chatbot, oss_memory],
            )
            oss_input.submit(
                oss_chat,
                [oss_input, oss_chatbot, oss_memory],
                [oss_input, oss_chatbot, oss_memory],
            )
            oss_clear.click(
                lambda: ([], ConversationMemory(SYSTEM_PROMPT)),
                outputs=[oss_chatbot, oss_memory],
            )

        with gr.Tab("Frontier Chat (GPT-4.1)"):
            frontier_memory = gr.State(ConversationMemory(SYSTEM_PROMPT))
            frontier_chatbot = gr.Chatbot(label="GPT-4.1", height=400, type="messages")
            with gr.Row():
                frontier_input = gr.Textbox(placeholder="Ask me anything...", show_label=False, scale=9)
                frontier_send = gr.Button("Send", scale=1, variant="primary")
            frontier_clear = gr.Button("Clear conversation")

            frontier_send.click(
                frontier_chat,
                [frontier_input, frontier_chatbot, frontier_memory],
                [frontier_input, frontier_chatbot, frontier_memory],
            )
            frontier_input.submit(
                frontier_chat,
                [frontier_input, frontier_chatbot, frontier_memory],
                [frontier_input, frontier_chatbot, frontier_memory],
            )
            frontier_clear.click(
                lambda: ([], ConversationMemory(SYSTEM_PROMPT)),
                outputs=[frontier_chatbot, frontier_memory],
            )

        with gr.Tab("Evaluation"):
            gr.Markdown("## Evaluation Suite\n15 prompts across 3 categories: factual accuracy, adversarial safety, and bias fairness.")
            eval_btn = gr.Button("Run Evaluation", variant="primary")
            gr.Markdown("### Detailed Results")
            eval_table = gr.Dataframe(label="All Results")
            gr.Markdown("### Category Summary")
            summary_table = gr.Dataframe(label="Summary by Category")
            gr.Markdown("### Cost & Latency")
            cost_table = gr.Dataframe(label="Cost Comparison")

            eval_btn.click(run_eval, outputs=[eval_table, summary_table, cost_table])

        with gr.Tab("Observability"):
            gr.Markdown("## Live Request Metrics\nAll chat requests are logged with latency, token count, guardrail status, and tool use.")
            refresh_btn = gr.Button("Refresh Metrics", variant="primary")
            obs_summary = gr.Markdown("Click **Refresh Metrics** to load.")
            gr.Markdown("### Per-Model Stats")
            obs_model_stats = gr.Dataframe(label="Model Summary")
            gr.Markdown("### Request Log")
            obs_log = gr.Dataframe(label="All Requests")

            refresh_btn.click(load_metrics, outputs=[obs_log, obs_summary, obs_model_stats])

if __name__ == "__main__":
    demo.launch()
