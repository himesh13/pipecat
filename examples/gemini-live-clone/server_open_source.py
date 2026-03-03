#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live Clone — Cost-Reduced Open-Source Server.

Replaces the paid Gemini Live service with three fully local, open-source
components that run on your own hardware at zero per-call cost:

    STT  → Whisper (faster-whisper, runs locally)
    LLM  → Ollama  (any open-source model — Llama 3.2, Gemma 3, Mistral, …)
    TTS  → Kokoro  (kokoro-onnx, runs locally)

Compared to the full Gemini Live pipeline (`server.py`), this variant:
  • has no per-minute or per-token API charges (hardware costs apply)
  • requires a local Ollama instance (https://ollama.com)
  • uses your CPU/GPU for inference (higher latency on low-end hardware)

Prerequisites:
    # Install Ollama  →  https://ollama.com/download
    ollama pull llama3.2   # or any other model

    pip install "pipecat-ai[runner,silero,whisper,kokoro]"

Setup:
    # No API keys needed — just run the server
    python server_open_source.py

Then open http://localhost:7860 in your browser.

Cost comparison vs Gemini Live API
───────────────────────────────────
Component     Gemini Live (server.py)     Open-Source (this file)
─────────────────────────────────────────────────────────────────
STT           Bundled in Gemini Live      Whisper small (local, free)
LLM           Bundled in Gemini Live      Llama 3.2 / any Ollama model
TTS           Bundled in Gemini Live      Kokoro ONNX (local, free)
Monthly cost  ~$0.0035 / min of audio     $0 after hardware cost
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

STATIC_DIR = Path(__file__).parent / "static"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))

# Ollama model to use — change to any model you have pulled locally.
# Recommended: llama3.2 (small, fast), gemma3 (Google open-source), mistral
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

_webrtc_handler: Optional[SmallWebRTCRequestHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _webrtc_handler
    _webrtc_handler = SmallWebRTCRequestHandler()
    yield
    if _webrtc_handler:
        await _webrtc_handler.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_bot(connection: SmallWebRTCConnection):
    """Create and run a cost-free open-source pipeline for a single WebRTC connection."""

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    # ── STT: Whisper runs locally on your machine, no API key needed ──────
    # Model.DISTIL_MEDIUM_EN is a distilled medium model — fast and English-only (~500 MB).
    # Swap for Model.SMALL for multilingual, or Model.LARGE for best quality.
    stt = WhisperSTTService(model=Model.DISTIL_MEDIUM_EN)

    # ── LLM: Ollama runs open-source models locally, free after download ──
    llm = OLLamaLLMService(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # ── TTS: Kokoro ONNX runs locally, no API key needed ─────────────────
    # Voice options: af_heart, af_bella, am_adam, am_michael, bf_emma, ...
    tts = KokoroTTSService(voice_id="af_heart")

    system_prompt = (
        "You are a helpful AI assistant in a real-time voice conversation. "
        "Be conversational and concise. Avoid bullet points, markdown, or "
        "special symbols that don't sound natural when spoken aloud."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Silero VAD is used for local turn detection (start/stop of user speech).
    # In the Gemini Live pipeline this is handled by Gemini's built-in VAD;
    # here we use a separate SileroVADAnalyzer since each component is independent.
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),       # Audio from browser microphone
            stt,                     # Whisper: audio → text
            user_aggregator,         # Accumulate user speech into context
            llm,                     # Ollama: text → text response
            tts,                     # Kokoro: text → audio
            transport.output(),      # Audio to browser speakers
            assistant_aggregator,    # Accumulate assistant response into context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting conversation")
        messages.append({"role": "system", "content": "Greet the user warmly and introduce yourself briefly."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Gemini Live clone web UI."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC offer — negotiate the peer connection and launch a bot."""

    async def on_connection(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_bot, connection)

    answer = await _webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Handle trickle ICE candidate exchange."""
    await _webrtc_handler.handle_patch_request(request)
    return {"status": "ok"}


if __name__ == "__main__":
    logger.info(f"Starting Open-Source Voice Agent at http://localhost:{PORT}")
    logger.info(f"Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
    uvicorn.run(app, host=HOST, port=PORT)
