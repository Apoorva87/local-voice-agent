"""Per-turn latency instrumentation.

The PRD treats this as required, not optional: priorities #1 and #2 are
latency claims, and a latency claim without a number attached is a vibe.
Every turn is written to ``logs/turns/<session>.jsonl`` so regressions are
diffable across runs rather than spot-checked by ear.

Implemented as a Pipecat observer rather than inline processors so that
measurement never sits in the audio path and cannot itself add latency.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


def _ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000, 1)


@dataclass
class TurnRecord:
    """One user turn, from the moment they stop talking to the moment we do."""

    turn: int
    speech_started_at: float | None = None
    vad_silence_at: float | None = None
    speech_ended_at: float | None = None
    transcript_at: float | None = None
    llm_first_token_at: float | None = None
    bot_audio_at: float | None = None
    bot_done_at: float | None = None
    interrupted_at: float | None = None
    interrupt_to_silence_ms: float | None = None
    transcript: str = ""
    tools: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """Durations that matter, derived from raw timestamps.

        The stage boundaries are not the naive sequential ones. A segmented
        STT starts transcribing when *VAD* hears silence, which is earlier
        than when *Smart Turn* declares the turn over -- so transcription
        overlaps the turn-hold window instead of following it. Measuring
        ``stt_ms`` from turn end yields negative numbers and hides the fact
        that STT is usually free.
        """
        # The LLM cannot start until both the turn is over and the transcript
        # exists, whichever lands later.
        llm_start = max(
            (t for t in (self.speech_ended_at, self.transcript_at) if t is not None),
            default=None,
        )
        stt_hidden = (
            self.transcript_at is not None
            and self.speech_ended_at is not None
            and self.transcript_at <= self.speech_ended_at
        )
        return {
            "turn": self.turn,
            "transcript": self.transcript,
            # The headline number: user stops talking -> agent starts talking.
            "voice_to_voice_ms": _ms(self.speech_ended_at, self.bot_audio_at),
            # How long Smart Turn v3 held the turn open after Silero first
            # heard silence. This is the price of *not* cutting the user off,
            # and the number to tune when the agent feels sluggish or jumpy.
            "turn_hold_ms": _ms(self.vad_silence_at, self.speech_ended_at),
            # Transcription time, measured from when STT actually began.
            "stt_ms": _ms(self.vad_silence_at, self.transcript_at),
            # True when transcription finished inside the turn-hold window,
            # meaning it cost the user nothing.
            "stt_hidden_by_turn_hold": stt_hidden,
            "llm_ttft_ms": _ms(llm_start, self.llm_first_token_at),
            "tts_ms": _ms(self.llm_first_token_at, self.bot_audio_at),
            "user_speech_ms": _ms(self.speech_started_at, self.speech_ended_at),
            "bot_speech_ms": _ms(self.bot_audio_at, self.bot_done_at),
            "interrupted": self.interrupted_at is not None,
            "interrupt_to_silence_ms": self.interrupt_to_silence_ms,
            "tools": self.tools,
        }


class TurnMetricsObserver(BaseObserver):
    """Times each pipeline stage and writes one JSON line per completed turn.

    Targets from the PRD, checked on every turn rather than by sampling:
      * <500 ms of dead air after a real turn end
      * interruption stops obsolete speech within 250 ms
    """

    VOICE_TO_VOICE_TARGET_MS = 500.0
    INTERRUPT_TARGET_MS = 250.0

    def __init__(self, log_dir: Path, session_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id or uuid.uuid4().hex[:8]
        self._dir = Path(log_dir) / "turns"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{self._session_id}.jsonl"
        self._turn_index = 0
        self._current: TurnRecord | None = None
        # Frames fan out to several processors, so the same frame instance is
        # observed more than once. Record only the first sighting of each.
        self._seen: set[int] = set()
        logger.info(f"Turn metrics -> {self._path}")

    # -- lifecycle -------------------------------------------------------

    def _begin_turn(self, now: float) -> TurnRecord:
        self._turn_index += 1
        self._current = TurnRecord(turn=self._turn_index, speech_started_at=now)
        return self._current

    def _flush(self) -> None:
        if self._current is None:
            return
        record = self._current.summary()
        self._current = None
        try:
            with self._path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # never let logging break the conversation
            logger.warning(f"Could not write turn metrics: {exc}")

        v2v = record["voice_to_voice_ms"]
        if v2v is not None:
            over = " OVER TARGET" if v2v > self.VOICE_TO_VOICE_TARGET_MS else ""
            logger.info(
                f"turn {record['turn']}: voice-to-voice {v2v}ms{over} "
                f"(stt {record['stt_ms']}ms, llm {record['llm_ttft_ms']}ms, "
                f"tts {record['tts_ms']}ms)"
            )

    def reset(self) -> None:
        """Drop the turn in progress without logging it.

        Used by the headless check to discard the startup greeting, which is
        not a user turn and would otherwise be written as one.
        """
        self._current = None

    def record_tool(self, name: str, latency_ms: float, ok: bool, note: str = "") -> None:
        """Called by the tool layer; the PRD requires per-tool success tracking."""
        if self._current is not None:
            self._current.tools.append(
                {"name": name, "latency_ms": round(latency_ms, 1), "ok": ok, "note": note}
            )

    # -- observation -----------------------------------------------------

    async def on_push_frame(self, data: FramePushed) -> None:
        frame: Frame = data.frame
        key = id(frame)
        if key in self._seen:
            return
        self._seen.add(key)
        if len(self._seen) > 4096:
            self._seen.clear()

        now = time.time()

        if isinstance(frame, UserStartedSpeakingFrame):
            # New speech while a turn is open means the previous one never
            # completed cleanly (e.g. barge-in); close its books first.
            if self._current is not None:
                self._flush()
            self._begin_turn(now)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Silero merely heard silence. The turn is NOT over yet -- Smart
            # Turn v3 still has to rule on whether this was a pause or an end.
            if self._current is not None and self._current.vad_silence_at is None:
                self._current.vad_silence_at = now

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Smart Turn v3's verdict. This is t=0 for the headline metric.
            if self._current is None:
                self._begin_turn(now)
            self._current.speech_ended_at = now

        elif isinstance(frame, TranscriptionFrame):
            if self._current is not None and self._current.transcript_at is None:
                self._current.transcript_at = now
                self._current.transcript = (frame.text or "").strip()

        elif isinstance(frame, LLMTextFrame):
            # LLMTextFrame, not LLMFullResponseStartFrame: the latter fires
            # when generation is dispatched, which makes TTFT look like ~1ms
            # and pushes the real wait into the TTS bucket.
            if self._current is not None and self._current.llm_first_token_at is None:
                self._current.llm_first_token_at = now

        elif isinstance(frame, (BotStartedSpeakingFrame, TTSAudioRawFrame)):
            # TTSAudioRawFrame too: BotStartedSpeakingFrame is emitted by the
            # output transport, so headless runs would otherwise never record
            # the moment the agent starts speaking.
            if self._current is not None and self._current.bot_audio_at is None:
                self._current.bot_audio_at = now

        elif isinstance(frame, InterruptionFrame):
            if self._current is not None:
                self._current.interrupted_at = now

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._current is not None:
                self._current.bot_done_at = now
                if self._current.interrupted_at is not None:
                    elapsed = _ms(self._current.interrupted_at, now)
                    self._current.interrupt_to_silence_ms = elapsed
                    if elapsed and elapsed > self.INTERRUPT_TARGET_MS:
                        logger.warning(
                            f"interruption took {elapsed}ms to silence "
                            f"(target {self.INTERRUPT_TARGET_MS}ms)"
                        )
                self._flush()

    async def cleanup(self) -> None:
        self._flush()
        await super().cleanup()
