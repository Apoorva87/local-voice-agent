"""Pre-load models so the first real turn is not the slow one.

Measured cold costs on this machine: MLX Whisper needs several seconds to
load its weights, and Ollama takes 6-9s to bring a model into memory. Paying
that on the user's first sentence would blow the latency budget on the one
turn that sets their expectations. Warmup runs concurrently with the greeting.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from loguru import logger

from voice_agent.settings import Settings


async def warm_llm(settings: Settings) -> None:
    """Force Ollama to load the model into memory."""
    from openai import AsyncOpenAI

    models = settings.models
    client = AsyncOpenAI(base_url=models.llm_base_url, api_key="ollama")
    extra = (
        {"reasoning_effort": models.llm_reasoning_effort}
        if models.llm_reasoning_effort
        else {}
    )
    await client.chat.completions.create(
        model=models.llm_model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        extra_body=extra,
    )


async def warm_stt(settings: Settings) -> None:
    """Force MLX Whisper to compile and load its weights.

    Transcribing one second of silence is enough to pay the load cost.
    """
    import mlx_whisper
    from pipecat.services.whisper.stt import MLXModel

    model = MLXModel[settings.models.stt_model.upper()]
    silence = np.zeros(16000, dtype=np.float32)
    await asyncio.to_thread(
        mlx_whisper.transcribe, silence, path_or_hf_repo=model.value
    )


async def warm_tts(settings: Settings) -> None:
    """Load Kokoro's ONNX session and its voice pack.

    Measured: the first synthesis in a process costs well over a second,
    which lands directly in voice-to-voice latency because TTS is the last
    stage before the user hears anything. Every later call is far cheaper.
    """
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

    model = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    if not model.exists() or not voices.exists():
        return  # first run downloads them; nothing to warm yet

    def _run() -> None:
        kokoro = Kokoro(str(model), str(voices))
        kokoro.create("Ready.", voice=settings.models.tts_voice, lang="en-us", speed=1.0)

    await asyncio.to_thread(_run)


async def warm_all(settings: Settings) -> None:
    """Warm every model. A failure here is logged, never fatal."""
    tasks = {
        "llm": warm_llm(settings),
        "stt": warm_stt(settings),
        "tts": warm_tts(settings),
    }
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.warning(f"Warmup for {name} failed (first turn will be slow): {result}")
    logger.info(f"Model warmup finished in {time.perf_counter() - t0:.1f}s")
