"""Runtime configuration, all overridable by environment variable.

Every component the PRD says to "benchmark and pick" is a setting here rather
than a literal in the pipeline, so swapping one never means editing wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


@dataclass(frozen=True)
class TurnSettings:
    """Turn detection — PRD priority #1.

    Smart Turn v3 decides end-of-turn from the waveform, so these VAD numbers
    only govern *speech presence*, not endpointing. That separation is the
    whole point: a silence timer cannot tell a thinking pause from a finished
    sentence, and this is the component the PRD says makes or breaks the build.
    """

    # Silero only answers "is there speech right now".
    vad_confidence: float = field(default_factory=lambda: _env_float("VAD_CONFIDENCE", 0.7))
    vad_start_secs: float = field(default_factory=lambda: _env_float("VAD_START_SECS", 0.2))
    # Deliberately generous: Smart Turn v3, not this timer, ends the turn.
    # Set low enough to stay responsive if the turn model is ever bypassed.
    vad_stop_secs: float = field(default_factory=lambda: _env_float("VAD_STOP_SECS", 0.8))
    vad_min_volume: float = field(default_factory=lambda: _env_float("VAD_MIN_VOLUME", 0.6))

    # How long Smart Turn v3 waits after speech before ruling on the turn.
    smart_turn_stop_secs: float = field(
        default_factory=lambda: _env_float("SMART_TURN_STOP_SECS", 0.2)
    )
    smart_turn_pre_speech_ms: float = field(
        default_factory=lambda: _env_float("SMART_TURN_PRE_SPEECH_MS", 0.0)
    )
    smart_turn_max_duration_secs: float = field(
        default_factory=lambda: _env_float("SMART_TURN_MAX_DURATION_SECS", 8.0)
    )

    # Second-layer defence: let the LLM veto an end-of-turn call. Costs a round
    # trip, so it stays off until the baseline is measured (PRD build step 2).
    llm_gate: bool = field(default_factory=lambda: _env_bool("TURN_LLM_GATE", False))


@dataclass(frozen=True)
class ModelSettings:
    """Model choices. Defaults are the measured winners on M4 Max."""

    # Chosen on measured TTFT (149-253ms) and first-sentence (~400ms).
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "glm-4.7-flash:latest"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    # Voice replies are short; capping stops the model rambling past its welcome.
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "300")))
    # Load-bearing. glm-4.7-flash and qwen3.5 are hybrid reasoning models and
    # think by default on Ollama's OpenAI-compatible endpoint -- which is the
    # endpoint Pipecat uses. Left on, the model emits only reasoning tokens and
    # the user hears silence. "none" is the one value that disables it there;
    # "low" does not. Set to "" for models that reject the parameter.
    llm_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", "none")
    )

    # Q4 turbo: the quality/latency knee on Apple Silicon for conversational audio.
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "LARGE_V3_TURBO_Q4"))
    stt_language: str = field(default_factory=lambda: os.getenv("STT_LANGUAGE", "en"))

    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "af_heart"))


@dataclass(frozen=True)
class MemorySettings:
    """Hindsight MCP. Local embedded Postgres, no data leaves the machine."""

    enabled: bool = field(default_factory=lambda: _env_bool("MEMORY_ENABLED", True))
    # Not Hindsight's default 8888: another service may hold, and it
    # binds 127.0.0.1 while Hindsight binds the wildcard, so "localhost"
    # silently resolves to Jupyter and every MCP call 403s.
    url: str = field(
        default_factory=lambda: os.getenv("HINDSIGHT_URL", "http://127.0.0.1:8899/mcp")
    )
    bank: str = field(default_factory=lambda: os.getenv("HINDSIGHT_BANK", "default"))
    # PRD: keep injected context compact so response generation stays fast.
    recall_budget_tokens: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_RECALL_BUDGET", "300"))
    )
    recall_timeout_secs: float = field(
        default_factory=lambda: _env_float("MEMORY_RECALL_TIMEOUT", 3.0)
    )


@dataclass(frozen=True)
class Settings:
    turn: TurnSettings = field(default_factory=TurnSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)

    log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
    )
    metrics_enabled: bool = field(default_factory=lambda: _env_bool("METRICS_ENABLED", True))


def load_settings() -> Settings:
    return Settings()
