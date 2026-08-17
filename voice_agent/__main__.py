"""CLI entrypoint.

Run with ``uv run voice-agent`` (or ``uv run python -m voice_agent``), then
open the printed URL. The browser tab is the microphone and speaker.
"""

from __future__ import annotations

import sys

from loguru import logger

from voice_agent.bot import bot  # noqa: F401  (discovered by the pipecat runner)


def cli() -> None:
    from pipecat.runner.run import main

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()


if __name__ == "__main__":
    cli()
