"""Long-term memory via the Hindsight local MCP server.

Two decisions shape this module.

**Memory is not an LLM-visible tool.** Hindsight exposes 32 tools, including
``delete_bank`` and ``clear_memories``; a voice agent must never have those a
mis-transcription away. More than that, recall happens *before* the model
runs and retain happens *after* the turn ends, so neither is a decision the
model needs to make. Keeping them out of the tool list removes two schemas
from every turn's context and makes misuse structurally impossible.

**Recall is triggered deterministically.** Measured tool-calling showed the
fast controller reliably failing to call memory on personal questions
("What's my sister's name again?") while getting web and shell tools right
every time. Rather than pay 3x latency for a bigger model, the pipeline
detects those questions itself and recalls before the LLM runs.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack

from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from voice_agent.settings import Settings

# The only two Hindsight tools this agent ever calls.
RECALL_TOOL = "recall"
RETAIN_TOOL = "sync_retain"

# Phrasings that mean "this is about me, my life, or something we said before".
# Deliberately generous: a needless recall costs ~200ms, while a missed one
# makes the agent look like it has amnesia.
_PERSONAL_PATTERNS = [
    r"\b(?:what|when|where|who|why|how)\b[^?]*\b(?:my|our|we|i)\b",
    r"\bmy\s+\w+",
    r"\bwe\s+(?:decided|agreed|discussed|talked|said|chose|picked)\b",
    r"\b(?:do you |can you )?remember\b",
    r"\blast\s+(?:time|week|month|year)\b",
    r"\bearlier\b.*\b(?:said|mentioned|talked)\b",
    r"\bwhat did i\b",
    r"\bwhat did we\b",
    r"\bremind me\b",
    r"\bagain\?$",
]
_PERSONAL_RE = re.compile("|".join(_PERSONAL_PATTERNS), re.IGNORECASE)


def looks_personal(transcript: str) -> bool:
    """True when a turn should trigger a memory lookup before responding.

    The PRD's rule is "before personal questions, call recall". This makes
    that rule mechanical rather than a hope about model behaviour.
    """
    text = (transcript or "").strip()
    if len(text) < 3:
        return False
    return bool(_PERSONAL_RE.search(text))


def compact_recall(raw: object, budget_chars: int) -> str:
    """Trim Hindsight's response to something worth spending tokens on.

    Recall returns full scoring metadata per hit. The LLM needs the facts and
    nothing else, and the PRD caps injected context at roughly 100-300 tokens
    to keep generation fast.
    """
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        results = payload.get("results", []) if isinstance(payload, dict) else []
    except (json.JSONDecodeError, AttributeError, TypeError):
        return str(raw)[:budget_chars]

    lines: list[str] = []
    seen: set[str] = set()
    used = 0
    for item in results:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # Hindsight appends provenance with " | When: ... | Involving: ...".
        # Keep the fact; the agent is speaking, not citing.
        fact = text.split(" | ")[0].strip()
        if not fact:
            continue
        # Hindsight stores the same fact as both an observation and a world
        # fact, and the two copies often differ only by trailing punctuation
        # or case. Compare on a normalised key so the agent does not say the
        # same thing twice.
        key = re.sub(r"[^a-z0-9]+", " ", fact.lower()).strip()
        if key in seen:
            continue
        if used + len(fact) > budget_chars:
            break
        seen.add(key)
        lines.append(fact)
        used += len(fact)
    return "\n".join(f"- {line}" for line in lines)


def is_durable(fact: str) -> bool:
    """The PRD's durable-fact filter: no memory write happens without it.

    Storing chit-chat pollutes recall for every later turn, so this errs
    toward not writing. Durable means a decision, preference, or stable fact
    about the user or the project -- not pleasantries and not questions.
    """
    text = (fact or "").strip()
    if len(text) < 15 or text.endswith("?"):
        return False
    lowered = text.lower()

    chatter = (
        "thanks", "thank you", "hello", "hi ", "hey ", "okay", "ok ",
        "never mind", "nothing", "cool", "nice", "sorry", "good morning",
    )
    if lowered.startswith(chatter):
        return False

    durable_markers = (
        "decided", "prefer", "always", "never", "my name", "i am", "i'm",
        "we use", "we chose", "we agreed", "remember that", "i live",
        "i work", "my ", "our ", "going to", "plan to", "rejected",
        "instead of", "because",
    )
    return any(marker in lowered for marker in durable_markers)


class MemoryClient:
    """A persistent MCP session against one Hindsight bank.

    The bank is pinned in the URL path rather than passed as an argument, so
    there is no way to read or write another bank by accident.
    """

    def __init__(self, settings: Settings):
        self._settings = settings.memory
        self._url = self._settings.url.rstrip("/") + "/" + self._settings.bank + "/"
        self._budget_chars = self._settings.recall_budget_tokens * 4  # ~4 chars/token
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> bool:
        """Open the session. Returns False if Hindsight is unreachable.

        A memory outage must degrade the agent, never break it: the
        conversation continues without recall.
        """
        if not self._settings.enabled:
            logger.info("Memory disabled (MEMORY_ENABLED=false)")
            return False
        stack = AsyncExitStack()
        try:
            read, write, _ = await stack.enter_async_context(streamablehttp_client(self._url))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception as exc:
            await stack.aclose()
            logger.warning(
                f"Memory unavailable at {self._url} ({exc}); "
                "continuing without recall. Start it with: scripts/start_hindsight.sh"
            )
            return False
        self._stack, self._session = stack, session
        logger.info(f"Memory: Hindsight bank {self._settings.bank!r} at {self._url}")
        return True

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = self._session = None

    async def _call(self, tool: str, args: dict) -> str | None:
        if self._session is None:
            return None
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool, args),
                timeout=self._settings.recall_timeout_secs,
            )
        except asyncio.TimeoutError:
            # Never let memory hold up a spoken reply.
            logger.warning(f"Memory {tool} timed out after {self._settings.recall_timeout_secs}s")
            return None
        except Exception as exc:
            logger.warning(f"Memory {tool} failed: {exc}")
            return None
        return "".join(getattr(block, "text", "") for block in result.content)

    async def recall(self, query: str) -> str:
        """Look up facts for this query. Returns "" when there is nothing."""
        raw = await self._call(RECALL_TOOL, {"query": query})
        return compact_recall(raw, self._budget_chars) if raw else ""

    async def retain(self, fact: str) -> bool:
        """Store a durable fact. Silently declines anything that is not."""
        if not is_durable(fact):
            return False
        return await self._call(RETAIN_TOOL, {"content": fact}) is not None
