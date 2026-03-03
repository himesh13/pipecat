#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live Clone - FastAPI server.

A self-contained web application that clones Google's Gemini Live experience.
Users connect via browser using WebRTC and can have a real-time voice
conversation with a Gemini-powered AI agent.

Orchestration
─────────────
After the LLM generates a response it can call tool functions that act as an
orchestration layer — routing the conversation to a specialised agent or
triggering an external automation.  Two tools are wired up by default:

  • route_to_agent(agent, reason)
      The LLM hands off to a specialist (e.g. "billing", "support", "sales").
      Extend `handle_route_to_agent` to forward the call to your real agent
      fleet (Daily co-browsing, a second Pipecat pipeline, an HTTP relay, …).

  • trigger_automation(automation, params)
      The LLM fires a background workflow (e.g. create a CRM ticket, send a
      Slack alert, start an n8n workflow).  Extend `handle_trigger_automation`
      to call your automation backend (Zapier, Make, n8n, custom webhook, …).

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

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
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


# ── Orchestration handlers ────────────────────────────────────────────────────
# The LLM calls these tool functions to dispatch to agents or automations.
# Replace the stub bodies with real integrations (HTTP webhooks, message queues,
# a second Pipecat pipeline, etc.).

async def handle_route_to_agent(params: FunctionCallParams):
    """Hand the conversation off to a specialist agent.

    Extend this function to:
      - Forward the call to a different Pipecat pipeline / bot
      - Trigger a Daily room transfer
      - Send an HTTP request to your agent fleet
    """
    agent = params.arguments.get("agent", "unknown")
    reason = params.arguments.get("reason", "")
    logger.info(f"[Orchestration] Routing to agent='{agent}' reason='{reason}'")
    # TODO: implement real agent routing (HTTP call, queue message, etc.)
    await params.result_callback({"status": "routed", "agent": agent})


async def handle_trigger_automation(params: FunctionCallParams):
    """Trigger a background automation workflow.

    Extend this function to:
      - POST to an n8n / Make / Zapier webhook
      - Create a CRM record or support ticket
      - Send a Slack / Teams notification
    """
    automation = params.arguments.get("automation", "unknown")
    payload = params.arguments.get("params", {})
    logger.info(f"[Orchestration] Triggering automation='{automation}' payload={payload}")
    # TODO: implement real automation dispatch (aiohttp.post to webhook, etc.)
    await params.result_callback({"status": "triggered", "automation": automation})


# ── Tool definitions ──────────────────────────────────────────────────────────

route_to_agent_tool = FunctionSchema(
    name="route_to_agent",
    description=(
        "Hand the conversation off to a specialised agent. "
        "Call this when the user needs help that a specialist agent should handle, "
        "such as billing inquiries, technical support, or sales questions."
    ),
    properties={
        "agent": {
            "type": "string",
            "enum": ["billing", "support", "sales", "general"],
            "description": "The specialist agent to route to.",
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of why you are routing to this agent.",
        },
    },
    required=["agent", "reason"],
)

trigger_automation_tool = FunctionSchema(
    name="trigger_automation",
    description=(
        "Trigger a background automation or workflow. "
        "Call this to create tickets, update CRM records, send notifications, "
        "or start any automated process without interrupting the conversation."
    ),
    properties={
        "automation": {
            "type": "string",
            "enum": ["create_ticket", "update_crm", "send_notification", "log_feedback"],
            "description": "The automation workflow to trigger.",
        },
        "params": {
            "type": "object",
            "description": "Key-value pairs passed to the automation workflow.",
        },
    },
    required=["automation"],
)

orchestration_tools = ToolsSchema(
    standard_tools=[route_to_agent_tool, trigger_automation_tool]
)

# Build the orchestration section of the system instruction from the tool schemas
# so the text stays in sync with the enum values defined above.
_agent_values = route_to_agent_tool.properties["agent"]["enum"]
_automation_values = trigger_automation_tool.properties["automation"]["enum"]
_ORCHESTRATION_INSTRUCTIONS = (
    "You have access to two orchestration tools:\n\n"
    "1. route_to_agent — Hand the conversation to a specialist. "
    "Available agents: " + ", ".join(f'"{v}"' for v in _agent_values) + ".\n\n"
    "2. trigger_automation — Silently kick off a background workflow. "
    "Available automations: " + ", ".join(f'"{v}"' for v in _automation_values) + ".\n\n"
    "Always complete your spoken response to the user before calling a tool."
)

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

    system_instruction = (
        "You are a helpful AI assistant powered by Google Gemini, in a real-time voice conversation.\n"
        "Be conversational, concise, and engaging. Speak naturally — avoid bullet points, markdown,\n"
        "or special symbols that don't sound natural when spoken aloud.\n"
        "Listen carefully and respond thoughtfully to what the user says.\n\n"
        + _ORCHESTRATION_INSTRUCTIONS
    )

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        voice_id="Puck",  # Options: Aoede, Charon, Fenrir, Kore, Puck
        tools=orchestration_tools,
    )

    # Register orchestration handlers
    llm.register_function("route_to_agent", handle_route_to_agent)
    llm.register_function("trigger_automation", handle_trigger_automation)

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
