"""Drive the real agent pipeline headlessly and report voice-to-voice latency.

Builds the exact processor chain the agent runs -- VAD, Smart Turn v3, MLX
Whisper, memory recall, the LLM with its tools, and Kokoro -- and pushes
synthesised speech through it as if it came from a microphone. Confirms the
agent answers, and measures the PRD's headline number: the time from the end
of the user's speech to the first sample of the agent's.

This is not a substitute for talking to it. Synthesised speech has none of
the pauses, filler words or room noise that make turn detection hard, so the
mid-sentence-pause milestone still has to be checked by voice. What this does
catch is a pipeline that is wired wrong.

Run: uv run python scripts/check_pipeline.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402

from voice_agent.bot import build_core  # noqa: E402
from voice_agent.services import build_tts  # noqa: E402
from voice_agent.prompts import GREETING  # noqa: E402
from voice_agent.settings import load_settings  # noqa: E402
from voice_agent.warmup import warm_all  # noqa: E402

UTTERANCE = "What did we decide about the memory architecture?"


def _last_turn_record(log_dir: Path) -> dict | None:
    """Most recently written turn from the metrics log."""
    import json

    turn_dir = Path(log_dir) / "turns"
    files = sorted(turn_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    lines = [ln for ln in files[-1].read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


class Tap(FrameProcessor):
    """Records transcripts as they leave STT, before the aggregator eats them."""

    def __init__(self, sink: "Sink"):
        super().__init__()
        self._sink = sink

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            self._sink.transcripts.append(frame.text.strip())
        await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    """Stands in for the audio output transport."""

    def __init__(self):
        super().__init__()
        self.reply_text: list[str] = []
        self.transcripts: list[str] = []
        self.first_bot_audio_at: float | None = None
        self.speech_ended_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame) and frame.text:
            self.reply_text.append(frame.text)
        elif isinstance(frame, (TTSAudioRawFrame, BotStartedSpeakingFrame)):
            if self.first_bot_audio_at is None:
                self.first_bot_audio_at = time.perf_counter()
        await self.push_frame(frame, direction)


async def synthesize(text: str) -> tuple[bytes, int]:
    """Make speech audio to feed the pipeline, using the same TTS engine."""
    settings = load_settings()
    captured: list[bytes] = []
    rate = 0

    class Grab(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            nonlocal rate
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSAudioRawFrame):
                captured.append(frame.audio)
                rate = frame.sample_rate or rate
            await self.push_frame(frame, direction)

    task = PipelineTask(Pipeline([build_tts(settings), Grab()]), idle_timeout_secs=None)
    runner = PipelineRunner(handle_sigint=False)
    run = asyncio.create_task(runner.run(task))
    await task.queue_frames([TTSSpeakFrame(text), EndFrame()])
    await asyncio.wait_for(run, timeout=120)
    return b"".join(captured), rate


async def main() -> int:
    settings = load_settings()

    print(f"synthesising input speech: {UTTERANCE!r}")
    pcm, rate = await synthesize(UTTERANCE)
    if not pcm:
        print("FAILED: could not synthesise input audio")
        return 1
    print(f"  {len(pcm) / 2 / rate:.2f}s @ {rate}Hz\n")

    core = await build_core(settings)
    print("warming models so the first turn is not the cold one...")
    await warm_all(settings)

    sink = Sink()
    tap = Tap(sink)
    processors = list(core.processors)
    # Right after STT: the user aggregator consumes TranscriptionFrames, so a
    # tap at the end of the pipeline would never see them.
    processors.insert(2, tap)
    task = PipelineTask(
        Pipeline([*processors, sink, core.aggregators.assistant()]),
        observers=[core.metrics] if core.metrics else [],
        idle_timeout_secs=None,
    )
    runner = PipelineRunner(handle_sigint=False)
    run = asyncio.create_task(runner.run(task))

    # Speak the greeting first, exactly as the agent does on connect. This
    # warms the TTS service's own ONNX session -- warming a separate Kokoro
    # instance does not help, since the service holds its own.
    await task.queue_frames([TTSSpeakFrame(GREETING)])
    await asyncio.sleep(3.0)
    sink.first_bot_audio_at = None  # discard the greeting's timings
    if core.metrics:
        core.metrics.reset()

    # Feed the audio in real time so VAD and Smart Turn v3 see a plausible
    # stream rather than one instantaneous burst.
    chunk = rate * 2 // 50  # 20ms
    frames = [
        InputAudioRawFrame(audio=pcm[i : i + chunk], sample_rate=rate, num_channels=1)
        for i in range(0, len(pcm), chunk)
    ]
    print("speaking into the pipeline...")
    for frame in frames:
        await task.queue_frames([frame])
        await asyncio.sleep(0.02)

    # Trailing silence, so the turn detector sees the utterance actually end.
    silence = b"\x00" * chunk
    for _ in range(50):  # 1 second
        await task.queue_frames(
            [InputAudioRawFrame(audio=silence, sample_rate=rate, num_channels=1)]
        )
        await asyncio.sleep(0.02)
    sink.speech_ended_at = time.perf_counter()

    # Give the agent time to transcribe, think, and start speaking.
    for _ in range(300):  # up to 30s
        if sink.first_bot_audio_at is not None:
            break
        await asyncio.sleep(0.1)

    await task.queue_frames([EndFrame()])
    try:
        await asyncio.wait_for(run, timeout=60)
    except asyncio.TimeoutError:
        await task.cancel()
    await core.memory.close()

    print()
    heard = " ".join(sink.transcripts)
    reply = "".join(sink.reply_text).strip()
    print(f"heard:  {heard!r}")
    print(f"replied: {reply!r}")

    failures = []
    if not heard:
        failures.append("the pipeline produced no transcript")
    if not reply:
        failures.append("the agent produced no spoken reply")

    if sink.first_bot_audio_at is None:
        failures.append("the agent never started speaking")

    # Read the agent's own instrumentation rather than timing it separately.
    # This checks the metrics pipeline is working as well as the audio one.
    turn = _last_turn_record(settings.log_dir)
    if turn:
        print("\nper-turn metrics (from the agent's own log):")
        for key in (
            "voice_to_voice_ms",
            "turn_hold_ms",
            "stt_ms",
            "llm_ttft_ms",
            "tts_ms",
            "tools",
        ):
            if turn.get(key) is not None:
                print(f"  {key}: {turn[key]}")
        v2v = turn.get("voice_to_voice_ms")
        if v2v and v2v > 500:
            print(f"\n  NOTE: {v2v:.0f}ms exceeds the 500ms target.")
    else:
        failures.append("no turn metrics were written")

    print()
    if failures:
        print("PIPELINE CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PIPELINE CHECK PASSED: audio in, spoken answer out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
