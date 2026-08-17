"""End-to-end smoke test with no microphone required.

Synthesises speech with Kokoro, feeds that audio back through MLX Whisper,
then asks the LLM to reply. This proves every model in the pipeline loads and
runs on this machine, warms the on-disk caches so the first real conversation
does not pay download cost, and prints per-stage latency.

The services are driven through real Pipecat pipelines rather than called
directly, because a service only has a task manager and a negotiated sample
rate once a pipeline has started it.

Run: uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402

from voice_agent.services import build_llm, build_stt, build_tts  # noqa: E402
from voice_agent.settings import load_settings  # noqa: E402

PHRASE = "What did we decide about the memory architecture?"
SAMPLE_RATE = 24000


class Capture(FrameProcessor):
    """Collects frames flowing past so the test can assert on them."""

    def __init__(self):
        super().__init__()
        self.audio: list[bytes] = []
        self.sample_rate = 0
        self.transcripts: list[str] = []
        self.errors: list[str] = []
        self.first_audio_at: float | None = None
        self.first_transcript_at: float | None = None
        self.started_at = time.perf_counter()

    def mark_start(self) -> None:
        """Reset the clock to the moment input actually starts flowing."""
        self.started_at = time.perf_counter()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            if self.first_audio_at is None:
                self.first_audio_at = time.perf_counter() - self.started_at
            self.audio.append(frame.audio)
            self.sample_rate = frame.sample_rate or self.sample_rate
        elif isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                if self.first_transcript_at is None:
                    self.first_transcript_at = time.perf_counter() - self.started_at
                self.transcripts.append(frame.text.strip())
        elif isinstance(frame, ErrorFrame):
            self.errors.append(str(frame.error))
        await self.push_frame(frame, direction)


async def _run(
    processors: list[FrameProcessor], frames: list[Frame], settle: float = 0.0
) -> None:
    """Push frames through a real pipeline, then close it.

    ``settle`` leaves time for work that continues after the last input frame
    (a segmented STT transcribes *after* it sees end-of-speech), so the
    EndFrame does not cut the pipeline short.
    """
    task = PipelineTask(Pipeline(processors), idle_timeout_secs=None)
    runner = PipelineRunner(handle_sigint=False)
    run = asyncio.create_task(runner.run(task))
    await task.queue_frames(frames)
    if settle:
        await asyncio.sleep(settle)
    await task.queue_frames([EndFrame()])
    await asyncio.wait_for(run, timeout=180)


async def main() -> int:
    settings = load_settings()
    print(
        f"LLM={settings.models.llm_model}  STT={settings.models.stt_model}  "
        f"TTS={settings.models.tts_voice}\n"
    )
    failures: list[str] = []

    # --- TTS -----------------------------------------------------------
    print("[1/3] Kokoro TTS ...", flush=True)
    tts_out = Capture()
    t0 = time.perf_counter()
    await _run([build_tts(settings), tts_out], [TTSSpeakFrame(PHRASE)])
    tts_total = time.perf_counter() - t0

    pcm = b"".join(tts_out.audio)
    if not pcm:
        detail = f": {tts_out.errors[0]}" if tts_out.errors else ""
        failures.append(f"TTS produced no audio{detail}")
        print(f"      FAIL: no audio produced{detail}")
    else:
        secs = len(pcm) / 2 / max(tts_out.sample_rate, 1)
        print(
            f"      ok: {secs:.2f}s of audio @ {tts_out.sample_rate}Hz, "
            f"first chunk {(tts_out.first_audio_at or 0) * 1000:.0f}ms, "
            f"total {tts_total:.2f}s"
        )

    # --- STT -----------------------------------------------------------
    print("[2/3] MLX Whisper STT ...", flush=True)
    text = ""
    if pcm:
        stt_out = Capture()
        # A segmented STT service buffers audio between the VAD speech frames
        # specifically -- not the turn-level ones -- which is why VADProcessor
        # has to sit ahead of STT in the real pipeline.
        chunk = SAMPLE_RATE * 2 // 10  # 100ms of 16-bit mono
        audio_frames: list[Frame] = [VADUserStartedSpeakingFrame()]
        audio_frames += [
            InputAudioRawFrame(
                audio=pcm[i : i + chunk],
                sample_rate=tts_out.sample_rate,
                num_channels=1,
            )
            for i in range(0, len(pcm), chunk)
        ]
        audio_frames.append(VADUserStoppedSpeakingFrame())

        stt_out.mark_start()
        await _run([build_stt(settings), stt_out], audio_frames, settle=10.0)

        text = " ".join(stt_out.transcripts)
        if text:
            # Measured to the transcript frame, not to pipeline shutdown, so
            # the settle time above is excluded.
            print(
                f"      ok: {(stt_out.first_transcript_at or 0) * 1000:.0f}ms "
                f"for {len(pcm) / 2 / tts_out.sample_rate:.1f}s of audio -> {text!r}"
            )
        else:
            detail = f": {stt_out.errors[0]}" if stt_out.errors else ""
            failures.append(f"STT returned an empty transcript{detail}")
            print(f"      FAIL: empty transcript{detail}")
    else:
        failures.append("STT skipped: no audio from TTS")
        print("      SKIP: no audio to transcribe")

    # --- LLM -----------------------------------------------------------
    print("[3/3] Ollama LLM ...", flush=True)
    build_llm(settings)  # proves the service constructs with our settings
    try:
        # Ollama's OpenAI-compatible endpoint. It ignores the key, but the
        # OpenAI client refuses to send a request without one.
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=settings.models.llm_base_url, api_key="ollama")
        t0 = time.perf_counter()
        stream = await client.chat.completions.create(
            model=settings.models.llm_model,
            messages=[{"role": "user", "content": text or PHRASE}],
            stream=True,
            max_tokens=60,
            # Same switch the agent uses; without it these models emit only
            # reasoning tokens and the content stream comes back empty.
            extra_body=(
                {"reasoning_effort": settings.models.llm_reasoning_effort}
                if settings.models.llm_reasoning_effort
                else {}
            ),
        )
        first = None
        reply = ""
        async for chunk_resp in stream:
            delta = chunk_resp.choices[0].delta.content if chunk_resp.choices else None
            if delta:
                if first is None:
                    first = time.perf_counter() - t0
                reply += delta
        if not reply.strip():
            failures.append("LLM returned no content (a reasoning-only model?)")
            print("      FAIL: no content tokens")
        else:
            print(f"      ok: TTFT {(first or 0) * 1000:.0f}ms -> {reply.strip()[:90]!r}")
    except Exception as exc:
        failures.append(f"LLM failed: {exc}")
        print(f"      FAIL: {exc}\n      Is Ollama running?  ollama serve")

    print()
    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED: every model loads and runs locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
