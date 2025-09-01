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
from typing import Dict, Set, Optional
import logging

import numpy as np
import sounddevice as sd
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Maintain active websocket connections.  We keep a simple set of WebSocket
# objects and broadcast to all connected clients whenever new information is
# available.
connections: Set[WebSocket] = set()

# Model loading with error handling
try:
    model = whisper.load_model("medium")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    model = None

# Translation models with better error handling and device detection
translator_ja_en: Optional[pipeline] = None
translator_en_ja: Optional[pipeline] = None

try:
    # Auto-detect device (use CPU if GPU not available)
    device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else -1
    
    translator_ja_en = pipeline(
        task="translation",
        model="facebook/m2m100_418M",
        tokenizer="facebook/m2m100_418M",
        device=device,             # cuda:0
        src_lang="ja",
        tgt_lang="en"
    )
    translator_en_ja = pipeline(
        task="translation",
        model="facebook/m2m100_418M",
        tokenizer="facebook/m2m100_418M",
        device=device,
        src_lang="en",
        tgt_lang="ja"
    )
    logger.info(f"Translation models loaded on device: {device}")
except Exception as e:
    logger.error(f"Failed to load translation models: {e}")


# ---------------------------------------------------------------------------
# Application state and configuration
# ---------------------------------------------------------------------------

# Default translation direction (Japanese -> English)
translation_direction = "ja-en"

# Audio configuration
FS = 16000
CHANNELS = 1  # 修正: モノラル録音に変更（計算効率とファイルサイズの最適化）
RECORD_SECONDS = 2

# Silence detection threshold (peak amplitude)
silence_threshold = 100

# Debug settings
DEBUG_MODE = True
debug_history: list[Dict] = []
audio_queue: "queue.Queue[tuple[str, float]]" = queue.Queue(maxsize=5)

# Thread control
recording_active = True
processing_active = True


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

async def broadcast_async(event: str, data: Dict) -> None:
    """Async version of broadcast for use within async contexts."""
    if not connections:
        return
        
    disconnected = set()
    for ws in connections:
        try:
            await ws.send_json({"event": event, "data": data})
        except Exception:
            disconnected.add(ws)
    
    # Clean up disconnected clients
    connections.difference_update(disconnected)


def broadcast(event: str, data: Dict) -> None:
    """Broadcast an event with data to all connected clients.

    ``broadcast`` can be invoked from both asynchronous and synchronous
    contexts.  When an event loop is already running (e.g. inside the
    WebSocket handler) we schedule the coroutine on that loop.  Otherwise we
    create a temporary loop via ``asyncio.run``.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_async(event, data))
    except RuntimeError:
        # No event loop running, create one
        asyncio.run(broadcast_async(event, data))


def log_debug(message: str, data: Dict | None = None) -> None:
    """Log debug information and emit it to clients."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    debug_entry = {"timestamp": timestamp, "message": message, "data": data}
    debug_history.append(debug_entry)

    # Keep only the last 50 entries
    if len(debug_history) > 50:
        debug_history.pop(0)

    if DEBUG_MODE:
        logger.info(f"[DEBUG {timestamp}] {message}")
        if data:
            logger.info(f"[DEBUG {timestamp}] Data: {json.dumps(data, indent=2, ensure_ascii=False)}")

    broadcast("debug_info", debug_entry)


# ---------------------------------------------------------------------------
# Audio recording and processing
# ---------------------------------------------------------------------------

def record_audio_to_file() -> tuple[str, float]:
    """Record audio from the microphone and store it in a temporary file."""
    if model is None:
        raise RuntimeError("Whisper model not loaded")
        
    log_debug("Starting audio recording", {"duration": RECORD_SECONDS, "sample_rate": FS})
    
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

        start_time = time.time()
        audio = sd.rec(int(RECORD_SECONDS * FS), samplerate=FS, channels=CHANNELS, dtype="int16")
        sd.wait()
        recording_time = time.time() - start_time

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))

        # Broadcast current audio level to all clients scaled to 0-1 range
        level = min(peak / 32768.0, 1.0)  # 修正: レベルを0-1に正規化
        
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
                "audio_level": level,
            },
        )

        # WAVファイルの書き込み
        with wave.open(tmp_file.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(FS)
            wf.writeframes(audio.tobytes())

        return tmp_file.name, peak
        
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        raise


def record_audio_loop() -> None:
    """Background thread that records audio and queues it for processing."""
    log_debug("Background recording thread started")
    
    try:
        while recording_active:
            try:
                wav_path, peak = record_audio_to_file()
                if peak < silence_threshold:
                    log_debug("Skipping silent audio", {"peak": peak, "threshold": silence_threshold})
                    os.remove(wav_path)
                    continue
                
                # キューが満杯の場合は古いアイテムを削除
                try:
                    audio_queue.put((wav_path, peak), timeout=0.1)
                except queue.Full:
                    log_debug("Audio queue full, dropping oldest item")
                    try:
                        old_path, _ = audio_queue.get_nowait()
                        if os.path.exists(old_path):
                            os.remove(old_path)
                        audio_queue.put((wav_path, peak), timeout=0.1)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in recording loop: {e}")
                time.sleep(1)  # 短い休止後に再試行
                
    except Exception as e:
        logger.error(f"Recording thread crashed: {e}")


def recognize_and_translate() -> None:
    """Background thread that performs speech recognition and translation."""
    log_debug("Background recognition thread started")
    
    if model is None or translator_ja_en is None or translator_en_ja is None:
        log_debug("ERROR: Models not properly loaded")
        return
    
    while processing_active:
        try:
            wav_path, peak = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue
            
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
            
            try:
                if translation_direction == "ja-en":
                    # M2M100モデルで日本語→英語翻訳
                    translation_result = translator_ja_en(original_text)  # English token ID
                else:
                    # M2M100モデルで英語→日本語翻訳
                    translation_result = translator_en_ja(original_text)  # Japanese token ID
                    
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
            except Exception as e:
                log_debug("Translation error", {"error": str(e), "input_text": original_text})

        except Exception as e:
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
                "facebook/m2m100_418M",
                "facebook/m2m100_418M",
            ],
            "default_direction": translation_direction,
            "audio_config": {
                "sample_rate": FS,
                "channels": CHANNELS,
                "record_duration": RECORD_SECONDS,
            },
        },
    )

    # バックグラウンドスレッドの開始
    recording_thread = threading.Thread(target=record_audio_loop, daemon=True, name="AudioRecording")
    processing_thread = threading.Thread(target=recognize_and_translate, daemon=True, name="AudioProcessing")
    
    recording_thread.start()
    processing_thread.start()
    
    logger.info("Background threads started successfully")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources when the application shuts down."""
    global recording_active, processing_active
    
    recording_active = False
    processing_active = False
    
    # Clear remaining audio files in queue
    while not audio_queue.empty():
        try:
            wav_path, _ = audio_queue.get_nowait()
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except queue.Empty:
            break
    
    logger.info("Application shutdown completed")


# ---------------------------------------------------------------------------
# Web routes and websocket handling
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "whisper": model is not None,
            "translator_ja_en": translator_ja_en is not None,
            "translator_en_ja": translator_en_ja is not None,
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle websocket connections from the client."""
    # global宣言を関数の最初に移動
    global translation_direction, silence_threshold, DEBUG_MODE
    
    await ws.accept()
    connections.add(ws)
    log_debug("Client connected", {"total_connections": len(connections)})

    # Send recent debug history to the new client
    for entry in debug_history[-10:]:
        await ws.send_json({"event": "debug_info", "data": entry})

    # Notify client about current configuration
    await ws.send_json({
        "event": "app_ready",
        "data": {
            "translation_direction": translation_direction,
            "silence_threshold": silence_threshold,
            "debug_mode": DEBUG_MODE,
        }
    })

    try:
        while True:
            message = await ws.receive_json()
            event = message.get("event")
            data = message.get("data", {})

            if event == "set_direction":
                direction = data.get("direction")
                if direction in ("ja-en", "en-ja"):
                    old = translation_direction
                    translation_direction = direction
                    log_debug("Translation direction changed", {
                        "old_direction": old, 
                        "new_direction": direction
                    })
                    # 設定変更を全クライアントに通知
                    await broadcast_async("direction_changed", {"direction": direction})

            elif event == "set_silence_threshold":
                threshold = data.get("threshold")
                if isinstance(threshold, (int, float)) and 0 <= threshold <= 32768:  # 修正: 有効範囲を制限
                    old = silence_threshold
                    silence_threshold = threshold
                    log_debug("Silence threshold changed", {
                        "old_threshold": old, 
                        "new_threshold": threshold
                    })
                    await broadcast_async("threshold_changed", {"threshold": threshold})

            elif event == "toggle_debug":
                enabled = data.get("enabled", True)
                DEBUG_MODE = enabled
                log_debug("Debug mode toggled", {"debug_enabled": DEBUG_MODE})
                await broadcast_async("debug_toggled", {"enabled": DEBUG_MODE})

            elif event == "clear_debug":
                debug_history.clear()
                log_debug("Debug history cleared")
                await ws.send_json({"event": "debug_cleared"})

            elif event == "get_debug_history":
                await ws.send_json({
                    "event": "debug_history", 
                    "data": {"history": debug_history}
                })

            elif event == "ping":
                await ws.send_json({"event": "pong", "data": {"timestamp": time.time()}})

    except WebSocketDisconnect:
        connections.discard(ws)
        log_debug("Client disconnected", {"total_connections": len(connections)})
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connections.discard(ws)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # 設定の検証
    if model is None:
        logger.error("Cannot start: Whisper model failed to load")
        exit(1)
    
    if translator_ja_en is None or translator_en_ja is None:
        logger.error("Cannot start: Translation models failed to load")
        exit(1)

    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")