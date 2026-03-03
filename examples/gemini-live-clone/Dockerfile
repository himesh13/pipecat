#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# Dockerfile for the Gemini Live variant (server.py).
# Uses Google Gemini Live API for STT, LLM, and TTS.
#
# Build:
#   docker build -t gemini-live-clone .
#
# Run:
#   docker run --env-file .env -p 7860:7860 gemini-live-clone
#
# Or use docker compose:
#   docker compose --profile gemini up

FROM python:3.12-slim

WORKDIR /app

# Install pipecat with Google Gemini Live, runner and Silero VAD extras
RUN pip install --no-cache-dir "pipecat-ai[google,runner,silero]"

# Copy application code
COPY server.py .
COPY static/ static/

EXPOSE 7860

CMD ["python", "server.py"]
