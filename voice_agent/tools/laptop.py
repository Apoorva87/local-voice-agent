"""Local shell access, gated by the confirmation policy.

Read-only commands run immediately. Anything else is held until the user
approves it out loud, and genuinely destructive commands are refused
regardless. See ``policy.py`` for why the bar is set where it is.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from loguru import logger

from voice_agent.tools.policy import Risk, classify

# Long enough for a real command, short enough that the conversation does not
# stall waiting on one.
COMMAND_TIMEOUT_SECS = 20.0
# Spoken back to the user, so it must stay short.
MAX_OUTPUT_CHARS = 800


@dataclass
class PendingCommand:
    command: str
    reason: str
    requested_at: float

    @property
    def stale(self) -> bool:
        """Approval must follow the request closely, or it is not consent."""
        return (time.time() - self.requested_at) > 120


class LaptopTool:
    """Runs shell commands with a one-slot confirmation queue."""

    def __init__(self):
        self._pending: PendingCommand | None = None

    async def run(self, command: str) -> str:
        """Entry point for the model's ``laptop_run`` tool call."""
        verdict = classify(command)

        if verdict.risk is Risk.BLOCKED:
            logger.warning(f"Blocked command: {command!r} ({verdict.reason})")
            self._pending = None
            return (
                f"Refused, because {verdict.reason}. "
                "Tell the user you will not run that."
            )

        if verdict.risk is Risk.NEEDS_CONFIRMATION:
            self._pending = PendingCommand(command, verdict.reason, time.time())
            logger.info(f"Awaiting confirmation for: {command!r} ({verdict.reason})")
            return (
                f"This needs the user's spoken approval, because {verdict.reason}. "
                f"Ask them to confirm running: {command}. "
                "Do not claim it has run. If they agree, call laptop_confirm."
            )

        return await self._execute(command)

    async def confirm(self, approved: bool = True) -> str:
        """Entry point for ``laptop_confirm`` after the user agrees."""
        pending = self._pending
        self._pending = None

        if pending is None:
            return "There is no command waiting for approval."
        if not approved:
            return f"Cancelled. Did not run: {pending.command}"
        if pending.stale:
            return (
                "That request is too old to run on the earlier approval. "
                "Ask the user to say what they want again."
            )

        logger.info(f"User approved: {pending.command!r}")
        return await self._execute(pending.command)

    async def _execute(self, command: str) -> str:
        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT_SECS)
            code = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            return f"The command was still running after {COMMAND_TIMEOUT_SECS:.0f} seconds, so it was stopped."
        except Exception as exc:
            return f"The command could not be run: {exc}"

        elapsed = (time.perf_counter() - t0) * 1000
        output = (stdout or b"").decode("utf-8", errors="replace").strip()
        logger.info(f"Ran {command!r} -> exit {code} in {elapsed:.0f}ms")

        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
        if not output:
            return f"Finished with exit code {code} and no output."
        return f"Exit code {code}. Output:\n{output}"
