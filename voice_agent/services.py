"""Factories for the swappable pieces of the pipeline.

Each model the PRD says to "benchmark and pick" is built here behind a
function, so replacing one (Kyutai STT, a different Ollama model, another
TTS) never means editing pipeline wiring.
"""

from __future__ import annotations

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import (
    FilterIncompleteUserTurnStrategies,
    UserTurnStrategies,
    default_user_turn_start_strategies,
)

from voice_agent.settings import Settings


def build_vad_processor(settings: Settings) -> VADProcessor:
    """Silero VAD: detects *speech presence* only.

    It answers "is someone talking right now", which is what triggers
    interruption of the bot. It deliberately does not decide when a turn is
    over -- that is Smart Turn v3's job.
    """
    turn = settings.turn
    analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=turn.vad_confidence,
            start_secs=turn.vad_start_secs,
            stop_secs=turn.vad_stop_secs,
            min_volume=turn.vad_min_volume,
        )
    )
    return VADProcessor(vad_analyzer=analyzer)


def build_turn_strategies(settings: Settings) -> UserTurnStrategies:
    """Turn detection -- PRD priority #1.

    Smart Turn v3 classifies end-of-turn from the raw waveform (~8M params,
    ~12ms CPU), so a mid-sentence pause reads as "still going" while a
    finished clause reads as "your turn". A silence timer cannot make that
    distinction at any threshold, which is why the PRD rules one out.

    With ``TURN_LLM_GATE=true`` the acoustic verdict is additionally gated on
    the LLM's own judgement of whether the utterance is complete. Stronger,
    but it costs a round trip on every turn, so it is opt-in.
    """
    turn = settings.turn
    analyzer = LocalSmartTurnAnalyzerV3()
    # SmartTurnParams is a pydantic model; set only what we tune.
    analyzer.params.stop_secs = turn.smart_turn_stop_secs
    analyzer.params.pre_speech_ms = turn.smart_turn_pre_speech_ms
    analyzer.params.max_duration_secs = turn.smart_turn_max_duration_secs

    stop = [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)]
    start = default_user_turn_start_strategies()

    if turn.llm_gate:
        logger.info("Turn detection: Smart Turn v3 + LLM completion gate")
        return FilterIncompleteUserTurnStrategies(start=start, stop=stop)

    logger.info(f"Turn detection: Smart Turn v3 (stop_secs={turn.smart_turn_stop_secs})")
    return UserTurnStrategies(start=start, stop=stop)


def build_stt(settings: Settings) -> WhisperSTTServiceMLX:
    """MLX Whisper, running on the Apple Silicon GPU.

    This is a *segmented* service: it transcribes the whole utterance once the
    turn ends rather than continuously decoding partials. That is the right
    shape here, because Smart Turn v3 ends the turn from audio -- the
    transcript is never on the endpointing critical path, so one fast batch
    decode beats repeated re-decodes.
    """
    name = settings.models.stt_model.upper()
    try:
        model = MLXModel[name]
    except KeyError as exc:
        valid = ", ".join(m.name for m in MLXModel)
        raise ValueError(f"Unknown STT_MODEL {name!r}. Valid options: {valid}") from exc

    logger.info(f"STT: MLX Whisper {model.value}")
    return WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model=model.value,
            language=Language(settings.models.stt_language),
        )
    )


def build_llm(settings: Settings) -> OLLamaLLMService:
    """Ollama over its OpenAI-compatible endpoint.

    Default is glm-4.7-flash, chosen by measurement: ~150-250ms to first
    token and ~400ms to first sentence, versus 690-890ms for qwen3.5:35b.
    First *sentence* is what matters, since TTS cannot start mid-clause.
    """
    models = settings.models
    # Thinking must be switched off explicitly. These models reason by default
    # over the OpenAI-compatible endpoint and would otherwise spend the whole
    # token budget on reasoning, returning empty content -- silence, to a
    # listener. Passed via `extra`, which merges into the request body.
    extra: dict[str, object] = {}
    if models.llm_reasoning_effort:
        extra["reasoning_effort"] = models.llm_reasoning_effort

    logger.info(
        f"LLM: {models.llm_model} at {models.llm_base_url} "
        f"(reasoning_effort={models.llm_reasoning_effort or 'unset'})"
    )
    return OLLamaLLMService(
        base_url=models.llm_base_url,
        settings=OLLamaLLMService.Settings(
            model=models.llm_model,
            temperature=models.llm_temperature,
            max_tokens=models.llm_max_tokens,
            extra=extra,
        ),
    )


def build_tts(settings: Settings) -> KokoroTTSService:
    """Kokoro-82M via ONNX -- the fastest local option.

    The PRD is explicit that expressiveness is not a design constraint, so
    this is chosen purely on latency. Weights land in ~/.cache/pipecat/ on
    first run.
    """
    logger.info(f"TTS: Kokoro-82M (voice={settings.models.tts_voice})")
    return KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=settings.models.tts_voice)
    )
