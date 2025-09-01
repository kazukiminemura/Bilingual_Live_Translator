"""Realtime Speech Translation Web App using FastAPI and WebSocket.

This module rewrites the original Flask based implementation to use
`FastAPI` and its native WebSocket support.  Audio is recorded on the
server, transcribed with Whisper and translated with HuggingFace
translation models.  Results and debug information are pushed to all
connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import tempfile
import threading
import time
import wave
from datetime import datetime
from typing import Dict, Set

import numpy as np
import sounddevice as sd
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Maintain active websocket connections.  We keep a simple set of WebSocket
# objects and broadcast to all connected clients whenever new information is
# available.
connections: Set[WebSocket] = set()

# Load Whisper model and translation pipelines.
model = whisper.load_model("medium")
# Translation models execute on GPU 0 by default for faster performance.
translator_ja_en = pipeline(
    "translation", model="Helsinki-NLP/opus-mt-ja-en", device=0
)
translator_en_ja = pipeline(
    "translation", model="Helsinki-NLP/opus-mt-en-jap", device=0
)


# ---------------------------------------------------------------------------
# Application state and configuration
# ---------------------------------------------------------------------------

# Default translation direction (Japanese -> English)
translation_direction = "ja-en"

# Audio configuration
FS = 16000
CHANNELS = 3
RECORD_SECONDS = 2

# Silence detection threshold (peak amplitude)
silence_threshold = 100

# Debug settings
DEBUG_MODE = True
debug_history: list[Dict] = []
audio_queue: "queue.Queue[tuple[str, float]]" = queue.Queue(maxsize=5)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def broadcast(event: str, data: Dict) -> None:
    """Broadcast an event with data to all connected clients.

    ``broadcast`` can be invoked from both asynchronous and synchronous
    contexts.  When an event loop is already running (e.g. inside the
    WebSocket handler) we schedule the coroutine on that loop.  Otherwise we
    create a temporary loop via ``asyncio.run``.  This avoids the
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``
    that occurred previously.
    """

    async def _broadcast() -> None:
        for ws in list(connections):
            try:
                await ws.send_json({"event": event, "data": data})
            except Exception:
                # Ignore failures for disconnected clients
                pass

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_broadcast())
    else:
        loop.create_task(_broadcast())


def log_debug(message: str, data: Dict | None = None) -> None:
    """Log debug information and emit it to clients."""

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    debug_entry = {"timestamp": timestamp, "message": message, "data": data}
    debug_history.append(debug_entry)

    # Keep only the last 50 entries
    if len(debug_history) > 50:
        debug_history.pop(0)

    if DEBUG_MODE:
        print(f"[DEBUG {timestamp}] {message}")
        if data:
            print(f"[DEBUG {timestamp}] Data: {json.dumps(data, indent=2, ensure_ascii=False)}")

    broadcast("debug_info", debug_entry)


# ---------------------------------------------------------------------------
# Audio recording and processing
# ---------------------------------------------------------------------------

def record_audio_to_file() -> tuple[str, float]:
    """Record audio from the microphone and store it in a temporary file."""

    log_debug("Starting audio recording", {"duration": RECORD_SECONDS, "sample_rate": FS})
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    start_time = time.time()
    audio = sd.rec(int(RECORD_SECONDS * FS), samplerate=FS, channels=CHANNELS, dtype="int16")
    sd.wait()
    recording_time = time.time() - start_time

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    # Broadcast current audio level to all clients scaled to 0-500 range
    level = (peak / 32768.0)
    
    # Display level value using log_debug
    log_debug("Audio level calculated", {"level": level, "peak": peak, "rms": rms})
    
    broadcast("audio_level", {"level": level})

    log_debug(
        "Audio recording completed",
        {
            "file_path": tmp_file.name,
            "peak_amplitude": peak,
            "rms_amplitude": rms,
            "recording_time": f"{recording_time:.2f}s",
            "audio_shape": audio.shape,
            "audio_level": level,  # levelもログに追加
        },
    )

    with wave.open(tmp_file.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(audio.tobytes())

    return tmp_file.name, peak


def record_audio_loop() -> None:
    """Background thread that records audio and queues it for processing."""

    log_debug("Background recording thread started")
    while True:
        wav_path, peak = record_audio_to_file()
        if peak < silence_threshold:
            log_debug("Skipping silent audio", {"peak": peak, "threshold": silence_threshold})
            os.remove(wav_path)
            continue
        audio_queue.put((wav_path, peak))


def recognize_and_translate() -> None:
    """Background thread that performs speech recognition and translation."""

    log_debug("Background recognition thread started")
    while True:
        wav_path, peak = audio_queue.get()
        try:
            log_debug("Starting transcription", {"whisper_model": "medium"})
            transcription_start = time.time()
            result = model.transcribe(wav_path)
            transcription_time = time.time() - transcription_start

            original_text = result.get("text", "").strip()
            segments = result.get("segments", [])

            log_debug(
                "Transcription completed",
                {
                    "original_text": original_text,
                    "transcription_time": f"{transcription_time:.2f}s",
                    "language": result.get("language"),
                    "segments_count": len(segments),
                },
            )

            if not original_text:
                log_debug("Skipping empty transcription")
                continue

            log_debug("Starting translation", {"direction": translation_direction, "input_text": original_text})
            translation_start = time.time()
            if translation_direction == "ja-en":
                translation_result = translator_ja_en(original_text)
                translation = translation_result[0]["translation_text"]
            else:
                translation_result = translator_en_ja(original_text)
                translation = translation_result[0]["translation_text"]
            translation_time = time.time() - translation_start

            log_debug(
                "Translation completed",
                {
                    "translated_text": translation,
                    "translation_time": f"{translation_time:.2f}s",
                },
            )

            broadcast(
                "subtitle",
                {
                    "original": original_text,
                    "translated": translation,
                    "debug": {
                        "peak_amplitude": peak,
                        "transcription_time": f"{transcription_time:.2f}s",
                        "translation_time": f"{translation_time:.2f}s",
                        "language_detected": result.get("language"),
                        "segments_count": len(segments),
                        "direction": translation_direction,
                    },
                },
            )

        except Exception as e:  # pragma: no cover - best effort logging
            error_msg = f"Error in recognition/translation: {e}"
            log_debug("ERROR occurred", {"error": error_msg, "error_type": type(e).__name__})
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            audio_queue.task_done()


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize background threads when the application starts."""

    log_debug(
        "Application starting",
        {
            "whisper_model": "medium",
            "translation_models": [
                "Helsinki-NLP/opus-mt-ja-en",
                "Helsinki-NLP/opus-mt-en-jap",
            ],
            "default_direction": translation_direction,
            "audio_config": {
                "sample_rate": FS,
                "channels": CHANNELS,
                "record_duration": RECORD_SECONDS,
            },
        },
    )

    threading.Thread(target=record_audio_loop, daemon=True).start()
    threading.Thread(target=recognize_and_translate, daemon=True).start()


# ---------------------------------------------------------------------------
# Web routes and websocket handling
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the main HTML page."""

    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle websocket connections from the client."""

    await ws.accept()
    connections.add(ws)
    log_debug("Client connected")

    # Send recent debug history to the new client
    for entry in debug_history[-10:]:
        await ws.send_json({"event": "debug_info", "data": entry})

    # Notify client that the application is ready
    await ws.send_json({"event": "app_ready"})

    try:
        while True:
            message = await ws.receive_json()
            event = message.get("event")
            data = message.get("data", {})

            if event == "set_direction":
                global translation_direction
                direction = data.get("direction")
                if direction in ("ja-en", "en-ja"):
                    old = translation_direction
                    translation_direction = direction
                    log_debug("Translation direction changed", {"old_direction": old, "new_direction": direction})

            elif event == "set_silence_threshold":
                global silence_threshold
                threshold = data.get("threshold")
                if isinstance(threshold, (int, float)) and threshold >= 0:
                    old = silence_threshold
                    silence_threshold = threshold
                    log_debug("Silence threshold changed", {"old_threshold": old, "new_threshold": threshold})

            elif event == "toggle_debug":
                global DEBUG_MODE
                DEBUG_MODE = data.get("enabled", True)
                log_debug("Debug mode toggled", {"debug_enabled": DEBUG_MODE})

            elif event == "clear_debug":
                debug_history.clear()
                log_debug("Debug history cleared")
                await ws.send_json({"event": "debug_cleared"})

            elif event == "get_debug_history":
                await ws.send_json({"event": "debug_history", "data": {"history": debug_history}})

    except WebSocketDisconnect:
        connections.discard(ws)
        log_debug("Client disconnected")


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

