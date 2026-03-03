# Gemini Live Clone

A self-contained voice-AI web application that mimics Google's Gemini Live
experience. Connect from your browser, click **Start conversation**, and have
a real-time voice chat with a Gemini-powered agent — powered by
[Pipecat](https://github.com/pipecat-ai/pipecat).

## How it works

```
Browser (mic) ──WebRTC──► server.py (FastAPI)
                               │
                    SmallWebRTCTransport
                               │
                     VADProcessor (Silero)
                               │
                  GeminiLiveLLMService  ◄── Google Gemini Live API
                               │
                    SmallWebRTCTransport
                               │
Browser (speakers) ◄─WebRTC──
```

`GeminiLiveLLMService` handles **speech recognition, language understanding,
and speech synthesis** natively — no separate STT or TTS services are needed.

## Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key with access to
  the Gemini Live API (model `gemini-2.0-flash-live-001` or later)
- A browser with WebRTC support (Chrome / Edge / Firefox / Safari)

## Setup

1. **Install dependencies**

   ```bash
   pip install "pipecat-ai[google,runner,silero]"
   ```

   Or if you prefer `uv` and there is a `pyproject.toml` in this directory:

   ```bash
   uv sync
   ```

2. **Configure your API key**

   ```bash
   cp .env.example .env
   # Edit .env and paste your GOOGLE_API_KEY
   ```

3. **Run the server**

   ```bash
   python server.py
   ```

4. **Open the UI**

   Navigate to **http://localhost:7860** in your browser, allow microphone
   access, and click **Start conversation**.

## Voice options

Edit `server.py` and change `voice_id` on `GeminiLiveLLMService`:

| Voice  | Personality   |
|--------|---------------|
| Puck   | Upbeat, playful (default) |
| Aoede  | Warm, melodic |
| Charon | Calm, measured |
| Fenrir | Deep, confident |
| Kore   | Bright, energetic |

## Configuration

| Environment variable | Default     | Description             |
|----------------------|-------------|-------------------------|
| `GOOGLE_API_KEY`     | *(required)*| Google AI Studio key    |
| `HOST`               | `0.0.0.0`   | Server bind address     |
| `PORT`               | `7860`      | Server port             |

## Troubleshooting

- **"Microphone access denied"** — click the lock icon in your browser's
  address bar and allow microphone access.
- **"Could not connect to the bot server"** — make sure `server.py` is running
  and check its console for errors.
- **No audio from the agent** — ensure your system volume is not muted and
  that the browser tab is not muted.
- **First response is slow** — the Silero VAD model is downloaded on first
  run; subsequent runs are faster.
