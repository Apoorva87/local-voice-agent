"""Pipeline assembly and the agent entrypoint."""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from voice_agent.memory import MemoryClient
from voice_agent.metrics import TurnMetricsObserver
from voice_agent.prompts import GREETING, SYSTEM_PROMPT
from voice_agent.recall_processor import MemoryRecallProcessor
from voice_agent.services import (
    build_llm,
    build_stt,
    build_tts,
    build_turn_strategies,
    build_vad_processor,
)
from voice_agent.settings import Settings, load_settings
from voice_agent.warmup import warm_all


def build_transport_params() -> TransportParams:
    """WebRTC transport, configured for a full-duplex conversation.

    ``audio_in_passthrough`` matters: Smart Turn v3 lives inside the user
    aggregator and needs the raw audio frames to reach it, not just the
    transcript. Without passthrough the turn analyzer would be deaf.
    """
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_passthrough=True,
    )


async def run_agent(transport: SmallWebRTCTransport, settings: Settings) -> None:
    """Wire the pipeline and run it until the client disconnects."""
    stt = build_stt(settings)
    llm = build_llm(settings)
    tts = build_tts(settings)
    vad = build_vad_processor(settings)
    turn_strategies = build_turn_strategies(settings)

    memory = MemoryClient(settings)
    await memory.connect()  # degrades gracefully if Hindsight is not running

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=turn_strategies),
    )

    metrics = TurnMetricsObserver(log_dir=settings.log_dir) if settings.metrics_enabled else None
    observers = [metrics] if metrics else []

    pipeline = Pipeline(
        [
            transport.input(),
            # VAD sits first so barge-in is detected the instant audio arrives,
            # and because segmented STT keys off its speech frames.
            vad,
            stt,
            # Recall runs on the settled transcript, before the model sees it.
            MemoryRecallProcessor(memory, metrics),
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=False,
        ),
        observers=observers,
        idle_timeout_secs=None,  # an always-available agent should not time out
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        # Speak a fixed greeting rather than asking the LLM to invent one:
        # deterministic, and it skips an LLM round trip at connect time.
        logger.info("Client connected; greeting and starting to listen")
        await task.queue_frames([TTSSpeakFrame(GREETING)])
        # Load model weights while the greeting plays, so the user's first
        # sentence is not the one that pays cold-start cost.
        asyncio.create_task(warm_all(settings))

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        await memory.close()


async def bot(runner_args: RunnerArguments) -> None:
    """Entrypoint discovered by ``pipecat.runner.run``."""
    settings = load_settings()
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=build_transport_params(),
    )
    await run_agent(transport, settings)


# Kept module-level so it is easy to find when tuning the first thing the
# agent says; see prompts.GREETING.
__all__ = ["bot", "run_agent", "build_transport_params", "GREETING"]
