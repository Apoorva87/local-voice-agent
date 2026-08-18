# Local Voice Conversation Agent

An always-on conversational voice agent that runs entirely on this machine.
No API keys, no cloud calls, no account — the only traffic that leaves the
laptop is a web search, and only when you ask a question that needs one.

Built to the priorities in [`project.prd`](project.prd): turn detection
first, response speed second, functionality third, expressiveness explicitly
last.

## Quick start

```bash
# 1. Ollama, with the conversation model
ollama serve
ollama pull glm-4.7-flash

# 2. Memory server (embedded Postgres, fully local)
./scripts/start_hindsight.sh

# 3. The agent
uv run voice-agent
```

Then open <http://localhost:7860> and start talking. The browser tab is the
microphone and speaker.

First launch downloads about 2 GB of model weights (Whisper, Kokoro) into
`~/.cache/`. Later launches start in seconds.

## Verifying it works

Each script checks one layer and prints real numbers:

```bash
uv run pytest                            # 46 unit tests, no models needed
uv run python scripts/smoke_test.py      # every model loads and runs
uv run python scripts/check_memory.py    # Hindsight recall + durable-fact filter
uv run python scripts/check_tools.py     # web search, shell policy, confirmation
uv run python scripts/check_pipeline.py  # full pipeline, headless, with latency
uv run python scripts/bench_llm.py       # compare candidate controller models
```

`check_pipeline.py` runs the real agent without a microphone: it synthesises
speech, feeds it in as if from a mic, and reports the per-turn metrics.

## Architecture

```text
Browser mic (echo cancellation)
  → SmallWebRTC in
  → Silero VAD ............... is anyone speaking?
  → MLX Whisper .............. transcribe the utterance
  → Smart Turn v3 ............ is the turn actually over?
  → Memory recall ............ deterministic, before the model runs
  → glm-4.7-flash + tools .... respond / search / run a command
  → Kokoro-82M ............... speak, streamed by sentence
  → SmallWebRTC out → Browser speaker
```

Every stage is timed and written to `logs/turns/<session>.jsonl`.

### Why these components

The PRD specified Kyutai STT and TTS **via Unmute**. Unmute cannot run here:
its README states *"Architecture must be x86_64, no aarch64 support is
planned"* and *"Neither is running on Mac."* It needs a CUDA GPU with 16 GB
of VRAM. Since Pipecat can host models in-process, Unmute's serving layer is
not needed at all — that removed an entire Docker tier from the design.

| Component | Choice | Why |
| --- | --- | --- |
| Orchestrator | Pipecat 1.7.0 | Frames, interruption, tool lifecycle |
| Transport | WebRTC + browser | The browser supplies echo cancellation |
| Turn detection | Smart Turn v3 | Semantic endpointing from audio, ~12 ms |
| STT | MLX Whisper large-v3-turbo-q4 | Runs on the Apple Silicon GPU |
| LLM | glm-4.7-flash via Ollama | Fastest to first *sentence* (see below) |
| TTS | Kokoro-82M (ONNX) | Fastest local option |
| Memory | Hindsight (local MCP) | Local embedded Postgres |

**Hibiki-Zero was evaluated and rejected.** It is a speech-to-speech
*translation* model (French/Spanish/Portuguese/German → English) and needs an
NVIDIA GPU. It does not converse and does not call tools.

## Measured behaviour

Controller models, benchmarked through Ollama's **OpenAI-compatible**
endpoint — the one Pipecat actually uses:

| Model | First sentence | Tool accuracy | Verdict |
| --- | --- | --- | --- |
| **glm-4.7-flash** | **310 ms** | 5/7 | Chosen |
| qwen3.5:35b | 764 ms | 7/7 | Fallback (`LLM_MODEL`) |
| gpt-oss:120b | no content | — | Reasoning-only; rejected |

glm's two misses were both *memory* lookups; it got web and shell tools right
every time. Rather than pay qwen's 2.5× latency, memory recall is triggered
deterministically in the pipeline (see below), which removes the dependency
on the model choosing to look.

### Current latency baseline

Measured by `check_pipeline.py`, warm, on an Apple M4 Max:

| Stage | Time |
| --- | --- |
| Turn hold (Smart Turn v3) | ~430 ms |
| STT | ~245 ms — *free, hidden inside the turn hold* |
| Memory recall | ~185 ms |
| LLM to first token | ~810 ms (includes recall) |
| TTS to first audio | ~620–830 ms |
| **Voice-to-voice** | **~1.4–1.6 s** |

**This misses the PRD's 500 ms target and is the next thing to work on.**
It is recorded here as the honest baseline for step 7 of the build sequence
(latency hardening) rather than presented as done. The leverage is in the LLM
and TTS stages; STT is already free, since transcription finishes inside the
window Smart Turn holds the turn open.

## Design decisions worth knowing

**Reasoning must be disabled explicitly.** `glm-4.7-flash` and `qwen3.5` are
hybrid reasoning models that think by default on Ollama's OpenAI-compatible
endpoint, returning *empty content* — silence, to a listener. Only
`reasoning_effort="none"` disables it; `"low"` does not. This is wired
through `LLM_REASONING_EFFORT`.

**Memory is not a tool the model can see.** Hindsight exposes 32 tools,
including `delete_bank` and `clear_memories`. Recall runs before the model
and retain runs after the turn, so neither needs to be a model decision.
Keeping them out of the tool list makes misuse structurally impossible and
saves two schemas of context on every turn.

**Recall triggers on a phrase heuristic**, not model judgement — see the
measured tool accuracy above.

**Shell access fails closed.** Only commands on an explicit read-only
allowlist run automatically. Anything else waits for spoken approval, and
genuinely destructive commands are refused outright. Any pipe, redirect or
command chaining forces confirmation, because a benign-looking prefix must
not launder what follows it. Both candidate models produced
`lsof -ti:8080 | xargs kill -9` confidently from a casual request, and speech
recognition can mishear.

## Configuration

Copy `.env.example` to `.env`. Every setting has a working default.

Notable ones:

- `LLM_MODEL` — swap the controller
- `LLM_REASONING_EFFORT` — must stay `none` for hybrid reasoning models
- `SMART_TURN_STOP_SECS` — how long to wait before ruling on end-of-turn
- `TURN_LLM_GATE` — adds an LLM veto on end-of-turn. Stronger protection
  against being cut off mid-sentence, at the cost of a round trip. Off by
  default; turn it on if turn detection cuts you off in practice.
- `HINDSIGHT_BANK` — which memory bank to use (created on first use)

## Known issues

**Hindsight runs on port 8899, not its default 8888.** Port 8888 is a
common conflict (it is a common default). If another service already holds
`127.0.0.1:8888` while Hindsight binds the wildcard, `localhost` resolves to
that service and every MCP call returns 403 — a confusing failure, since
Hindsight's own log still says it started. `scripts/start_hindsight.sh`
avoids it.

**Hindsight's extraction model must not be a reasoning model.** It parses
JSON out of its LLM, and thinking tokens break that parse with a
`JSONDecodeError`. The start script uses `llama3.2:3b`, which also keeps
background extraction off the GPU the voice model is using.

**`pipecat-ai[mlx-whisper]` alone cannot import.** The whisper module imports
`faster_whisper` at top level despite a comment claiming MLX loads lazily, so
the `whisper` extra is installed alongside it.

## Not built yet

- Post-turn memory writeback (`is_durable` filter exists and is tested; it is
  not yet called after each turn)
- Spoken filler while a slow tool runs (`TOOL_FILLERS` is defined, not wired)
- Latency hardening to reach the 500 ms target
