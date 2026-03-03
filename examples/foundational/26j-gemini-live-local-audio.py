#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local audio Gemini Live agent example.

This example sets up a Gemini Live AI agent that you can talk to directly
using your computer's microphone and speakers — no cloud transport required.

Prerequisites:
    pip install "pipecat-ai[google,local,silero]"
    # On macOS you also need: brew install portaudio

Usage:
    GOOGLE_API_KEY=<your-key> python 26j-gemini-live-local-audio.py

Press Ctrl+C to stop the agent.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    # Local audio transport: capture from microphone, play back through speakers
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    system_instruction = """
    You are a helpful AI assistant powered by Google Gemini.
    Your goal is to have a natural, engaging voice conversation with the user.
    Your output will be spoken aloud, so avoid special characters that can't
    easily be spoken, such as emojis or bullet points.
    Respond to what the user says in a helpful and friendly way.
    """

    # Gemini Live handles STT, LLM reasoning, and TTS all in one service
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        voice_id="Puck",  # Options: Aoede, Charon, Fenrir, Kore, Puck
    )

    # VAD (Voice Activity Detection) detects when the user starts/stops speaking
    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5))
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Microphone input
            vad_processor,      # Detect speech boundaries
            llm,                # Gemini Live (STT + reasoning + TTS built-in)
            transport.output(), # Speaker output
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Kick off the conversation with a greeting once the pipeline starts
    await task.queue_frames(
        [
            LLMMessagesAppendFrame(
                messages=[
                    {
                        "role": "user",
                        "content": "Greet the user and introduce yourself briefly.",
                    }
                ]
            )
        ]
    )

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
