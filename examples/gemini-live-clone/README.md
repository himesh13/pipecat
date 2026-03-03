# Gemini Live Clone

A self-contained voice-AI web application that mimics Google's Gemini Live
experience. Connect from your browser, click **Start conversation**, and have
a real-time voice chat with an AI agent — powered by
[Pipecat](https://github.com/pipecat-ai/pipecat).

Two server variants are provided so you can choose the right cost/quality
trade-off for your use case:

| Variant | File | STT | LLM | TTS | Cost |
|---------|------|-----|-----|-----|------|
| **Gemini Live** | `server.py` | Bundled | Gemini Live | Bundled | ~$0.0035 / min audio |
| **Open-Source** | `server_open_source.py` | Whisper (local) | Ollama (local) | Kokoro (local) | **$0** per call (after hardware) |

Both variants include an **orchestration layer** so the LLM can route
conversations to specialised agents or trigger background automations — see
[Orchestration](#orchestration) below.

---

## Option A — Gemini Live (`server.py`)

Best quality, lowest latency. Requires a Google AI Studio API key.
`GeminiLiveLLMService` bundles speech recognition, reasoning, and speech
synthesis into a single real-time service.

```
Browser mic ──WebRTC──► FastAPI server
                         SmallWebRTCTransport
                         VADProcessor (Silero)
                         GeminiLiveLLMService ◄── Google Gemini Live API
                              │ function calls
                         Orchestration handlers
                         SmallWebRTCTransport
                              │
                       Browser speakers
```

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini Live API)

### Setup

```bash
pip install "pipecat-ai[google,runner,silero]"
cp .env.example .env   # add your GOOGLE_API_KEY
python server.py
```

Open **http://localhost:7860**.

### Voice options

Edit `server.py` and change `voice_id` on `GeminiLiveLLMService`:

| Voice  | Character |
|--------|-----------|
| Puck   | Upbeat, playful (default) |
| Aoede  | Warm, melodic |
| Charon | Calm, measured |
| Fenrir | Deep, confident |
| Kore   | Bright, energetic |

---

## Option B — Open-Source / Cost-Free (`server_open_source.py`)

Replaces every paid API with a locally-running open-source component.
After the one-time model download there are **no per-call charges**.

```
Browser mic ──WebRTC──► FastAPI server
                         SmallWebRTCTransport
                         SileroVAD
                         WhisperSTTService  (local, faster-whisper)
                         LLMContextAggregatorPair
                         OLLamaLLMService   (local Ollama instance)
                              │ function calls
                         Orchestration handlers
                         KokoroTTSService   (local, kokoro-onnx)
                         SmallWebRTCTransport
                              │
                       Browser speakers
```

### Prerequisites

1. **[Ollama](https://ollama.com/download)** — local LLM runtime

   ```bash
   # macOS / Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull a model (choose one)
   ollama pull llama3.2        # 2 GB, fast, great quality
   ollama pull gemma3          # Google open-source, excellent quality
   ollama pull mistral         # 4 GB, good multilingual support
   ```

2. **Python dependencies**

   ```bash
   pip install "pipecat-ai[runner,silero,whisper,kokoro]"
   ```

### Run

```bash
# No API keys required
python server_open_source.py

# To use a different Ollama model:
OLLAMA_MODEL=gemma3 python server_open_source.py
```

Open **http://localhost:7860**.

### Whisper model sizes

Edit `server_open_source.py` and change the `model=` argument on
`WhisperSTTService`:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `Model.DISTIL_MEDIUM_EN` | ~500 MB | Fastest | Good (English only) |
| `Model.SMALL` | ~500 MB | Fast | Good (multilingual) |
| `Model.MEDIUM` | ~1.5 GB | Moderate | Better |
| `Model.LARGE` | ~3 GB | Slowest | Best |

### Kokoro voice options

Edit `server_open_source.py` and change `voice_id=` on `KokoroTTSService`:
`af_heart`, `af_bella`, `am_adam`, `am_michael`, `bf_emma`, `bm_george`

---

## Orchestration

Both server variants support LLM-driven orchestration through Pipecat's
standard [function-calling API](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/llm_service.py).

### How it works

```
User speech
    │
    ▼
STT / Gemini Live
    │  text transcript
    ▼
LLM (Gemini Live or Ollama)
    │  spoken reply + optional tool calls
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
TTS / Gemini Live                    Orchestration handler
(audio to browser)                   route_to_agent()
                                     trigger_automation()
                                          │
                                          ▼
                                  Your backend system
                                  (agent fleet, CRM, n8n, …)
```

### Available tools

| Tool | When to use | Agents / automations |
|------|-------------|----------------------|
| `route_to_agent` | Hand off to a specialist | `billing`, `support`, `sales`, `general` |
| `trigger_automation` | Fire a background workflow | `create_ticket`, `update_crm`, `send_notification`, `log_feedback` |

### Extending the handlers

Both handlers are defined at the top of each server file.  Replace the
`TODO` bodies with real integration code:

```python
# Example: POST to an n8n webhook when a ticket needs to be created
async def handle_trigger_automation(params: FunctionCallParams):
    automation = params.arguments.get("automation")
    payload    = params.arguments.get("params", {})

    if automation == "create_ticket":
        async with aiohttp.ClientSession() as session:
            await session.post(
                os.getenv("N8N_WEBHOOK_URL"),
                json={"automation": automation, **payload},
            )

    await params.result_callback({"status": "triggered", "automation": automation})
```

Common integration targets:

| Target | What to do in the handler |
|--------|--------------------------|
| **n8n / Make / Zapier** | `aiohttp.post(WEBHOOK_URL, json=payload)` |
| **Second Pipecat bot** | `await task.queue_frames([LLMRunFrame()])` on a second pipeline |
| **Slack / Teams** | Use the respective SDK / webhook |
| **CRM (Salesforce, HubSpot)** | Use the REST API or Python SDK |
| **Database** | `await db.execute(...)` |

---

## Configuration

| Variable | Default | Applies to |
|----------|---------|------------|
| `GOOGLE_API_KEY` | *(required)* | `server.py` only |
| `OLLAMA_MODEL` | `llama3.2` | `server_open_source.py` only |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | `server_open_source.py` only |
| `HOST` | `0.0.0.0` | Both |
| `PORT` | `7860` | Both |

---

## Troubleshooting

- **"Microphone access denied"** — allow microphone access in your browser.
- **"Could not connect to the bot server"** — make sure `server.py` (or
  `server_open_source.py`) is running and check its console for errors.
- **No audio from the agent** — ensure your system volume is not muted.
- **Ollama connection refused** — run `ollama serve` and make sure your model
  is pulled (`ollama list`).
- **LLM never calls tools** — some smaller Ollama models have limited
  function-calling support; try `llama3.2`, `mistral-nemo`, or `gemma3`.
- **First response is slow** — Whisper and Kokoro download model files on
  first run (~500 MB each); subsequent runs start immediately.

