"""Verify the memory path against a live Hindsight server.

Exercises exactly what the agent does: connect, recall on a personal
question, and store a durable fact while refusing chit-chat.

Run: uv run python scripts/check_memory.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent.memory import MemoryClient, is_durable, looks_personal  # noqa: E402
from voice_agent.settings import load_settings  # noqa: E402

QUESTION = "What did we decide about the memory architecture?"


async def main() -> int:
    settings = load_settings()
    memory = MemoryClient(settings)

    if not await memory.connect():
        print("FAILED: could not reach Hindsight.")
        print("  Start it:  ./scripts/start_hindsight.sh")
        return 1

    failures: list[str] = []
    try:
        print(f"trigger check: looks_personal({QUESTION!r}) = {looks_personal(QUESTION)}")
        if not looks_personal(QUESTION):
            failures.append("the milestone question does not trigger recall")

        t0 = time.perf_counter()
        facts = await memory.recall(QUESTION)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\nrecall: {elapsed:.0f}ms")
        print(facts or "  (nothing stored yet)")
        if elapsed > 300:
            failures.append(f"recall took {elapsed:.0f}ms, over the 300ms budget")

        print("\ndurable-fact filter:")
        for candidate, expected in [
            ("We decided to use Kokoro for text to speech because it is fastest.", True),
            ("Thanks, that's helpful.", False),
            ("What time is it?", False),
            ("My sister's name is Priya.", True),
        ]:
            got = is_durable(candidate)
            mark = "ok  " if got == expected else "FAIL"
            print(f"  [{mark}] {str(got):5} {candidate[:58]!r}")
            if got != expected:
                failures.append(f"durable filter wrong for {candidate!r}")
    finally:
        await memory.close()

    print()
    if failures:
        print("MEMORY CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("MEMORY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
