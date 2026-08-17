# Local Voice Conversation Agent — Design

Date: 2026-08-17
Status: Approved (decisions delegated to implementer)
Supersedes component choices in `project.prd` where noted.

## 1. Why this document exists

`project.prd` (v2) names a component stack that cannot run on the target
machine. This document records what was verified, what changed, and why.
The PRD's *priorities* are unchanged and still govern every decision:

1. Turn detection
2. Response speed
3. Functionality (tools, memory)
4. Expressiveness (explicitly not a constraint)

## 2. Target hardware

| Property | Value |
| --- | --- |
| Machine | Apple M4 Max |
| Memory | unified memory |
| OS | macOS on Apple Silicon |
| Arch | arm64 (aarch64) |

## 3. Verified findings that change the PRD

### 3.1 Unmute cannot run here — removed from the stack

`kyutai-labs/unmute` states: *"Architecture must be x86_64, no aarch64
support is planned"* and *"Neither is running on Mac."* It requires a
CUDA GPU with >=16 GB VRAM and targets Linux or Windows+WSL.

The PRD routes **both** STT and TTS "via Unmute", so this is a
load-bearing failure, not a detail. Unmute is removed entirely.

**What replaces it:** the models run natively on Apple Silicon in-process
via MLX / ONNX. We do not need Unmute's serving layer at all, because
Pipecat hosts the models directly. This removes a whole Docker
deployment tier from the design.

### 3.2 Kyutai STT survives as a *model*, not as a deployment

Kyutai STT/TTS do run on Apple Silicon through `moshi-mlx`
(`kyutai/stt-2.6b-en-mlx`). But they ship as standalone scripts, not
Pipecat services, so adoption means hand-writing a Pipecat service around
a 2.6B model.

**Decision:** not now. Pipecat ships a working MLX Whisper service today.
STT sits behind a factory function so this can be revisited without
touching the pipeline.

### 3.3 Hibiki-Zero is out of scope

Evaluated at request. It is a **speech-to-speech translation** model
(FR/ES/PT/DE -> EN, 3B params, released 2026-02) and requires an NVIDIA
GPU with >=8 GB VRAM. It does not converse and does not call tools. Wrong
category of model and wrong hardware. Rejected.

### 3.4 Turn detection is now a solved, bundled problem

Pipecat 1.7.0 makes **Smart Turn v3** the *default* stop strategy:
`default_user_turn_stop_strategies()` returns
`[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]`.

Smart Turn v3 is a semantic endpointing model (~8M params, Whisper-tiny
backbone) that classifies end-of-turn from the **waveform**, not the
transcript, at ~12 ms CPU inference. This is exactly the component the
PRD calls for and it costs almost nothing.

Pipecat also ships `FilterIncompleteUserTurnStrategies`, which gates turn
completion on an **LLM verdict** (the model prefixes replies with
complete / incomplete-short / incomplete-long markers). This is a second,
stronger defence against cutting the user off mid-sentence, at the cost
of an extra LLM round trip.

**Decision:** ship Smart Turn v3 as the default. Expose the LLM gate
behind `TURN_LLM_GATE=true`, default off. Rationale: the PRD mandates
measuring a speed baseline before adding cost. If the mid-sentence-pause
milestone fails on Smart Turn v3 alone, the gate is the documented
escalation.

## 4. Model selection — measured, not assumed

Benchmarked on this machine against locally available Ollama models.
Voice latency is dominated by **time-to-first-sentence**, because
streaming TTS cannot begin speaking mid-clause.

| Model | Cold load | TTFT | First sentence | Tool accuracy | Median tool decision |
| --- | --- | --- | --- | --- | --- |
| **glm-4.7-flash** | 6.2 s | 149-253 ms | ~400 ms | 6/7 | 464 ms |
| qwen3.5:35b | 9.4 s | ~260 ms | 688-890 ms | 7/7 | 1315 ms |
| gpt-oss:120b | 18.9 s | no content emitted | - | not tested | - |

`gpt-oss:120b` streamed zero content tokens before completion — output
went entirely to its reasoning channel. Reasoning models are hostile to
voice latency: the user hears silence for the whole thinking phase.
Rejected despite ample RAM.

**Decision: `glm-4.7-flash` is the controller.** Its single tool-calling
miss was "What's my sister's name again?" (failed to call memory recall).
That exact weakness is already addressed by the PRD's own rule — *"Before
personal questions, call Hindsight `recall`"* — implemented as a
deterministic pre-LLM trigger (section 6.2). Paying qwen's 3x latency to
recover one case the architecture handles anyway is a bad trade against
priorities #1 and #2. `qwen3.5:35b` stays available via `LLM_MODEL`.

Both models emitted confident destructive shell commands
(`lsof -ti:8080 | xargs kill -9`). This is direct evidence that
`laptop.run` must never auto-execute writes (section 7).

## 5. Component stack (final)

| Component | Choice | Notes |
| --- | --- | --- |
| Orchestrator | Pipecat 1.7.0 | Frames, interruption, tool lifecycle |
| Transport | `SmallWebRTCTransport` + browser client | Chosen for **echo cancellation** |
| VAD | `SileroVADAnalyzer` via `VADProcessor` | Speech presence only |
| Turn detection | `LocalSmartTurnAnalyzerV3` | Semantic endpointing, ~12 ms |
| STT | `WhisperSTTServiceMLX` (`LARGE_V3_TURBO_Q4`) | Apple Silicon MLX |
| LLM | `OLLamaLLMService`, `glm-4.7-flash` | OpenAI-compatible at :11434/v1 |
| TTS | `KokoroTTSService` (kokoro-onnx) | Fastest local; expressiveness not a goal |
| Memory | Hindsight local MCP, bank `default` | Embedded Postgres, fully local |
| Tools | `MCPClient` + native function schemas | Policy-gated (section 7) |

### 5.1 Transport rationale

WebRTC via a browser tab is chosen specifically because the browser
supplies **acoustic echo cancellation, noise suppression, and auto gain
control**. The PRD requires "echo cancellation working" in step 1.
Without AEC, an always-on agent hears its own TTS output and interrupts
itself — the dominant failure mode for this class of system. Native
`sounddevice` I/O would be ~20-40 ms faster but would require headphones
or a hand-built AEC subproject. This is the same reason Pipecat's own
macOS reference agent uses WebRTC.

### 5.2 Why segmented STT is correct here

`WhisperSTTServiceMLX` is a `SegmentedSTTService`: it transcribes a
complete utterance once the turn ends, rather than emitting partial
hypotheses continuously. This is not a compromise. Because Smart Turn v3
decides turn end from audio, the transcript is not on the critical path
for endpointing — so a single fast batch transcription after turn end is
both simpler and lower total latency than continuous decoding with
repeated re-decodes. Streaming STT would only matter if endpointing
depended on transcript content.

## 6. Architecture

```text
Browser mic (AEC/NS/AGC)
  -> SmallWebRTC in
  -> VADProcessor (Silero)          speech present?
  -> User aggregator
       stop strategy: SmartTurnV3   turn actually over?
  -> WhisperSTTServiceMLX           transcribe the settled turn
  -> [pre-LLM memory trigger]       deterministic recall
  -> OLLamaLLMService (glm-4.7-flash) + tool schemas
  -> KokoroTTSService               stream by sentence
  -> SmallWebRTC out -> Browser speaker
                 |
                 +-> TurnMetrics (every stage, every turn)
```

### 6.1 State machine

`LISTENING -> THINKING -> TOOL_RUNNING -> SPEAKING`, with interruption
returning to `LISTENING` from any state. Audio ingestion is never blocked
on a tool call. Pipecat's interruption handling cancels in-flight LLM
generation and TTS on `UserStartedSpeakingFrame`; in-flight tool calls are
cancelled by the same signal.

### 6.2 Deterministic memory recall

A pre-LLM trigger inspects the settled transcript for personal/continuity
references (possessives about people, "we decided", "remember when",
"what did I say about", etc.). On a hit, `memory.recall` is dispatched
before the LLM turn and the compact result injected into context. This
removes reliance on the model *choosing* to recall, which is glm's one
measured weakness. Injected results are capped at ~300 tokens per the PRD.

## 7. Tool policy

Read-only tools execute automatically. Writes, sends, purchases, and
destructive or system-level actions require spoken confirmation before
execution.

| Tool | Class | Behaviour |
| --- | --- | --- |
| `memory.recall` | read | auto |
| `memory.retain` | write | auto, but gated by durable-fact filter |
| `web.search` | read | auto, with spoken filler if slow |
| `laptop.run` | read or write | classified per command; writes confirm |

Classification of `laptop.run` is by explicit allowlist of read-only
commands, defaulting to "requires confirmation" when unmatched. Fail
closed.

Every tool call logs: call ID, transcript span, latency, cancellation,
result size, errors. The PRD marks this as required instrumentation, not
optional — it is what makes priorities #1-#2 measurable.

## 8. Instrumentation

Per-turn record, emitted as JSONL to `logs/turns/`:

- `turn_end_detected_at` — Smart Turn v3 verdict
- `stt_complete_at`, `first_llm_token_at`, `first_tts_audio_at`
- `voice_to_voice_ms` — turn end -> first agent audio (headline metric)
- `interrupt_to_silence_ms` — when interrupted
- per tool: name, latency, success, cancelled

Targets from the PRD: <500 ms dead air after real turn end; interruption
stops speech within 250 ms; tool dispatch begins within 300 ms of stable
transcript.

## 9. Build order

Follows the PRD sequence.

1. Turn detection + audio baseline (VAD, Smart Turn v3, STT, metrics) —
   no LLM, no TTS. Tune against recorded samples with real pauses.
2. Add LLM + TTS. Record the voice-to-voice baseline.
3. Hindsight `recall` + deterministic trigger. Verify latency budget.
4. `laptop.run` with the confirmation policy.
5. `web.search` with cancellation and spoken filler.
6. Memory writeback with durable-fact filter.
7. Latency hardening; optionally revisit expressiveness.

## 10. Explicitly deferred

- Kyutai STT/TTS via `moshi-mlx` (custom Pipecat service)
- Moshi / speech-native voice-to-voice models
- Emotion-tuned TTS (Chatterbox and similar)
- Native audio I/O with a purpose-built AEC
- Hibiki-Zero (wrong category, wrong hardware)

## 11. Known issues

`pipecat-ai[mlx-whisper]` alone cannot import
`pipecat.services.whisper.stt`: the module's top-level `faster_whisper`
import raises, despite an in-file comment stating MLX is imported lazily.
Workaround: also install the `whisper` extra. Both extras are pinned in
`pyproject.toml`.
