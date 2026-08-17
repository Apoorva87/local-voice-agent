"""Verify the tool layer end to end: web search, and the laptop confirm flow.

Run: uv run python scripts/check_tools.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent.tools.laptop import LaptopTool  # noqa: E402
from voice_agent.tools.web import search  # noqa: E402


async def main() -> int:
    failures: list[str] = []

    print("[1/3] web_search ...")
    t0 = time.perf_counter()
    result = await search("what is the pipecat voice ai framework")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"      {elapsed:.0f}ms")
    print("      " + result[:280].replace("\n", "\n      "))
    if "Search results:" not in result:
        failures.append(f"web search returned no results: {result[:120]}")

    print("\n[2/3] laptop_run, read-only path ...")
    laptop = LaptopTool()
    out = await laptop.run("sw_vers")
    print("      " + out.replace("\n", "\n      ")[:200])
    if not out.startswith("Exit code 0"):
        failures.append("read-only command did not execute automatically")

    print("\n[3/3] laptop_run, confirmation path ...")
    target = Path("/tmp/voice_agent_confirm_probe.txt")
    target.unlink(missing_ok=True)

    held = await laptop.run(f"touch {target}")
    print(f"      request: {held[:120]}")
    if target.exists():
        failures.append("SAFETY: a write command executed without confirmation")
    if "approval" not in held.lower():
        failures.append("write command was not held for approval")

    declined = await LaptopTool().confirm(True)
    if "no command waiting" not in declined.lower():
        failures.append("confirm on a fresh tool should find nothing pending")

    approved = await laptop.confirm(True)
    print(f"      after approval: {approved[:100]}")
    if not target.exists():
        failures.append("approved command did not run")
    target.unlink(missing_ok=True)

    blocked = await laptop.run("rm -rf /")
    print(f"      blocked case: {blocked[:90]}")
    if not blocked.startswith("Refused"):
        failures.append("SAFETY: a destructive command was not refused")

    print()
    if failures:
        print("TOOL CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("TOOL CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
