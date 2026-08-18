"""CLI entrypoint.

Run with ``uv run voice-agent`` (or ``uv run python -m voice_agent``), then
open the printed URL. The browser tab is the microphone and speaker.
"""

from __future__ import annotations

import sys

from loguru import logger

from voice_agent.bot import bot


def cli() -> None:
    from pipecat.runner.run import main

    # Pipecat's runner locates the per-connection entrypoint by looking for a
    # `bot` attribute on sys.modules["__main__"]. Under a console script that
    # module is the generated wrapper in .venv/bin, not this file, so `bot`
    # would be invisible and every WebRTC connection would fail with
    # "Could not find 'bot' function" *after* successfully negotiating ICE.
    # Attaching it explicitly makes the entrypoint discoverable however the
    # process was started.
    sys.modules["__main__"].bot = bot

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()


if __name__ == "__main__":
    cli()
