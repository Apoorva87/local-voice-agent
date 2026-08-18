#!/usr/bin/env bash
# Start the Hindsight memory server (local, embedded Postgres, no cloud).
#
# Two settings here are load-bearing:
#
#   Port 8899, not Hindsight's default 8888, which is a common conflict
#   (Jupyter uses it too). If another service already holds 127.0.0.1:8888
#   while Hindsight binds the wildcard, "localhost" resolves to that service
#   and every MCP call returns 403.
#
#   llama3.2:3b, not the conversation model. Hindsight parses JSON out of its
#   extraction LLM, and reasoning models (glm-4.7-flash, qwen3.5) emit
#   thinking tokens that break that parse with a JSONDecodeError. It also
#   keeps background extraction off the GPU the voice model is using.
set -euo pipefail

PORT="${HINDSIGHT_PORT:-8899}"
MODEL="${HINDSIGHT_EXTRACTION_MODEL:-llama3.2:3b}"

if ! curl -sf "http://localhost:11434/api/version" >/dev/null; then
  echo "Ollama is not running. Start it with: ollama serve" >&2
  exit 1
fi

if ! ollama list | grep -q "^${MODEL%%:*}"; then
  echo "Pulling extraction model ${MODEL}..."
  ollama pull "${MODEL}"
fi

echo "Starting Hindsight on 127.0.0.1:${PORT} (extraction model: ${MODEL})"
exec env \
  HINDSIGHT_API_LLM_PROVIDER=ollama \
  HINDSIGHT_API_LLM_MODEL="${MODEL}" \
  uvx --from hindsight-api hindsight-local-mcp --host 127.0.0.1 --port "${PORT}"
