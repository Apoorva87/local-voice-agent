"""Injects recalled memory into the conversation before the LLM responds.

Sits between STT and the user context aggregator. When a transcript looks
like a personal or continuity question, it recalls first and attaches the
result to the same turn, so the model answers with the facts already in hand
rather than having to decide to go looking for them.
"""

from __future__ import annotations

import time

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.memory import MemoryClient, looks_personal
from voice_agent.metrics import TurnMetricsObserver

_PREAMBLE = (
    "Relevant facts from your memory of earlier conversations with this user. "
    "Use them to answer directly, and do not mention that you looked anything up:"
)


class MemoryRecallProcessor(FrameProcessor):
    """Deterministic pre-LLM recall.

    Blocking the turn on recall is deliberate: the alternative is answering
    "I don't know" and then correcting yourself, which is worse than ~200ms of
    delay. The call is bounded by ``MEMORY_RECALL_TIMEOUT``, and a failure
    degrades to answering without memory.
    """

    def __init__(self, memory: MemoryClient, metrics: TurnMetricsObserver | None = None):
        super().__init__()
        self._memory = memory
        self._metrics = metrics

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        transcript = (frame.text or "").strip()
        if not transcript or not self._memory.connected or not looks_personal(transcript):
            await self.push_frame(frame, direction)
            return

        t0 = time.perf_counter()
        facts = await self._memory.recall(transcript)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if self._metrics:
            self._metrics.record_tool(
                "memory.recall", elapsed_ms, ok=bool(facts), note=f"{len(facts)} chars"
            )

        if not facts:
            logger.debug(f"Recall found nothing for {transcript!r} ({elapsed_ms:.0f}ms)")
            await self.push_frame(frame, direction)
            return

        logger.info(f"Recall injected {len(facts)} chars in {elapsed_ms:.0f}ms")
        # Prepend the facts to the user's own words so they arrive as part of
        # this turn. The aggregator sees one transcript and builds one turn.
        enriched = TranscriptionFrame(
            text=f"{_PREAMBLE}\n{facts}\n\nThe user asked: {transcript}",
            user_id=frame.user_id,
            timestamp=frame.timestamp,
            language=frame.language,
            # Preserved: the aggregator uses this to decide the turn is
            # settled, and dropping it would stall the turn.
            finalized=frame.finalized,
        )
        await self.push_frame(enriched, direction)
