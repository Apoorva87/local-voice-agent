"""Declares the tools the model can call and wires up their handlers.

Only web search and laptop control are exposed. Memory is handled outside the
model entirely (see ``voice_agent.memory``), which keeps this list short --
every schema here costs context and decision latency on every single turn.
"""

from __future__ import annotations

import time

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService

from voice_agent.metrics import TurnMetricsObserver
from voice_agent.tools.laptop import LaptopTool
from voice_agent.tools.web import search as web_search

WEB_SEARCH = FunctionSchema(
    name="web_search",
    description=(
        "Search the public web for current events, news, prices, or any fact "
        "you do not already know. Do not use this for anything about the user "
        "personally."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "A short search query, as you would type into a search box.",
        }
    },
    required=["query"],
)

LAPTOP_RUN = FunctionSchema(
    name="laptop_run",
    description=(
        "Run a shell command on the user's laptop to inspect or change the "
        "system. Read-only commands run immediately. Anything that modifies, "
        "deletes, installs or sends something will come back asking for the "
        "user's spoken approval -- relay that request and never claim the "
        "command has run until it actually has."
    ),
    properties={
        "command": {
            "type": "string",
            "description": "The exact shell command to run.",
        }
    },
    required=["command"],
)

LAPTOP_CONFIRM = FunctionSchema(
    name="laptop_confirm",
    description=(
        "Call this only after the user has verbally agreed to run a command "
        "that was waiting for approval. Set approved to false if they declined."
    ),
    properties={
        "approved": {
            "type": "boolean",
            "description": "True if the user agreed, false if they refused.",
        }
    },
    required=["approved"],
)


def build_tools_schema() -> ToolsSchema:
    return ToolsSchema(standard_tools=[WEB_SEARCH, LAPTOP_RUN, LAPTOP_CONFIRM])


def register_tools(llm: LLMService, metrics: TurnMetricsObserver | None = None) -> None:
    """Attach handlers for every declared tool.

    ``cancel_on_interruption`` is what makes the PRD's "cancel stale tool
    calls immediately" real: if the user starts talking again, an in-flight
    search or command result is abandoned rather than spoken late.
    """
    laptop = LaptopTool()

    async def _record(name: str, started: float, ok: bool, note: str = "") -> None:
        if metrics:
            metrics.record_tool(name, (time.perf_counter() - started) * 1000, ok, note)

    async def handle_web_search(params: FunctionCallParams):
        started = time.perf_counter()
        query = (params.arguments or {}).get("query", "")
        result = await web_search(query)
        await _record("web_search", started, ok="Search results:" in result, note=query[:40])
        await params.result_callback(result)

    async def handle_laptop_run(params: FunctionCallParams):
        started = time.perf_counter()
        command = (params.arguments or {}).get("command", "")
        result = await laptop.run(command)
        await _record("laptop_run", started, ok=result.startswith("Exit code"), note=command[:40])
        await params.result_callback(result)

    async def handle_laptop_confirm(params: FunctionCallParams):
        started = time.perf_counter()
        approved = bool((params.arguments or {}).get("approved", False))
        result = await laptop.confirm(approved)
        await _record("laptop_confirm", started, ok=result.startswith("Exit code"))
        await params.result_callback(result)

    # Searches and commands are abandoned the moment the user speaks again.
    llm.register_function("web_search", handle_web_search, cancel_on_interruption=True)
    llm.register_function("laptop_run", handle_laptop_run, cancel_on_interruption=True)
    # Confirmation must NOT be cancelled mid-flight: the user has already
    # approved it, and half-running a command is worse than finishing it.
    llm.register_function("laptop_confirm", handle_laptop_confirm, cancel_on_interruption=False)

    logger.info("Tools registered: web_search, laptop_run, laptop_confirm")
