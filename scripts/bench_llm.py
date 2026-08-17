"""Benchmark candidate controller LLMs the way the agent actually calls them.

Critically this uses Ollama's **OpenAI-compatible** endpoint, because that is
what Pipecat uses. Benchmarking against Ollama's native /api/chat endpoint
measures a different code path with different defaults -- notably `think`,
which does not exist on /v1 and is replaced by `reasoning_effort`.

Measures time-to-first-*sentence*, not just first token: streaming TTS cannot
begin speaking mid-clause, so that is the number the listener feels.

Run: uv run python scripts/bench_llm.py
"""

from __future__ import annotations

import asyncio
import re
import statistics
import sys
import time

from openai import AsyncOpenAI

MODELS = ["glm-4.7-flash:latest", "qwen3.5:35b"]
BASE_URL = "http://localhost:11434/v1"
REASONING_EFFORT = "none"

SYSTEM = (
    "You are a local voice assistant. Reply in one or two short spoken "
    "sentences. Never use markdown, lists, or emoji."
)
PROMPTS = [
    "Hey, what's the weather like usually in Seattle in August?",
    "Remind me what we decided about the memory architecture.",
    "What's a good way to explain recursion to a kid?",
]
SENTENCE_END = re.compile(r"[.!?]")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": (
                "Search the user's long-term memory for prior conversations, "
                "decisions, preferences and personal facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current or external information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laptop_run",
            "description": "Run a shell command on the user's local laptop.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

TOOL_CASES = [
    ("What did we decide about the memory architecture?", "memory_recall"),
    ("Who won the F1 race this weekend?", "web_search"),
    ("How much disk space do I have left?", "laptop_run"),
    ("Thanks, that's helpful.", None),
    ("What's my sister's name again?", "memory_recall"),
    ("Kill the process listening on port 8080.", "laptop_run"),
    ("Say that again but shorter.", None),
]


def _extra() -> dict:
    return {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}


async def measure_latency(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        stream=True,
        max_tokens=120,
        temperature=0.3,
        extra_body=_extra(),
    )
    ttft = first_sentence = None
    text = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if not delta:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t0
        text += delta
        if first_sentence is None and SENTENCE_END.search(text):
            first_sentence = time.perf_counter() - t0
    return {
        "ttft": ttft,
        "first_sentence": first_sentence,
        "total": time.perf_counter() - t0,
        "text": text.strip(),
    }


async def measure_tools(client: AsyncOpenAI, model: str) -> tuple[int, list[float], list[str]]:
    correct = 0
    latencies: list[float] = []
    misses: list[str] = []
    for utterance, expected in TOOL_CASES:
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": utterance},
            ],
            tools=TOOLS,
            temperature=0.0,
            max_tokens=120,
            extra_body=_extra(),
        )
        latencies.append(time.perf_counter() - t0)
        calls = resp.choices[0].message.tool_calls or []
        got = calls[0].function.name if calls else None
        if got == expected:
            correct += 1
        else:
            misses.append(f"{utterance!r} expected {expected} got {got}")
    return correct, latencies, misses


async def main() -> int:
    client = AsyncOpenAI(base_url=BASE_URL, api_key="ollama")
    print(f"endpoint={BASE_URL}  reasoning_effort={REASONING_EFFORT!r}\n")

    for model in MODELS:
        print(f"=== {model} ===", flush=True)
        try:
            t0 = time.perf_counter()
            await measure_latency(client, model, "hi")  # warm the weights
            print(f"  warmup: {time.perf_counter() - t0:.2f}s", flush=True)

            ttfs: list[float] = []
            for prompt in PROMPTS:
                r = await measure_latency(client, model, prompt)
                if r["ttft"] is None:
                    print("  NO CONTENT -- model returned reasoning only")
                    continue
                ttfs.append(r["first_sentence"] or r["total"])
                print(
                    f"  TTFT {r['ttft'] * 1000:6.0f}ms | 1st sentence "
                    f"{(r['first_sentence'] or 0) * 1000:6.0f}ms | {r['text'][:70]!r}",
                    flush=True,
                )

            correct, tool_lat, misses = await measure_tools(client, model)
            print(
                f"  --> median first sentence "
                f"{statistics.median(ttfs) * 1000:.0f}ms | tools {correct}/{len(TOOL_CASES)}"
                f" | median tool decision {statistics.median(tool_lat) * 1000:.0f}ms"
            )
            for m in misses:
                print(f"      miss: {m}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
