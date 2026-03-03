#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live Clone - FastAPI server.

A self-contained web application that clones Google's Gemini Live experience.
Users connect via browser using WebRTC and can have a real-time voice
conversation with a Gemini-powered AI agent.

Prerequisites:
    pip install "pipecat-ai[google,runner,silero]"

    Or with uv in this directory:
        uv sync

Setup:
    cp .env.example .env
    # Add your GOOGLE_API_KEY to .env

Run:
    python server.py

Then open http://localhost:7860 in your browser.
"""

import asyncio
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
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
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
    """Create and run a Pipecat pipeline for a single WebRTC connection."""

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    system_instruction = """
    You are a helpful AI assistant powered by Google Gemini, in a real-time voice conversation.
    Be conversational, concise, and engaging. Speak naturally — avoid bullet points, markdown,
    or special symbols that don't sound natural when spoken aloud.
    Listen carefully and respond thoughtfully to what the user says.
    """

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        voice_id="Puck",  # Options: Aoede, Charon, Fenrir, Kore, Puck
    )

    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5))
    )

    pipeline = Pipeline(
        [
            transport.input(),   # Audio from browser microphone
            vad_processor,       # Detect speech start/stop
            llm,                 # Gemini Live: speech recognition + reasoning + synthesis
            transport.output(),  # Audio to browser speakers
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
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Greet the user warmly and briefly introduce yourself.",
                        }
                    ]
                )
            ]
        )

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
    logger.info(f"Starting Gemini Live Clone at http://localhost:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
