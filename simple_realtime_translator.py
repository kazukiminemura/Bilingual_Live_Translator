# Realtime Speech Translation Web App with Debug Features
# (Flask + WebSocket + Whisper + HuggingFace)

import os
import time
import whisper
import threading
import sounddevice as sd
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from transformers import pipeline
import tempfile
import wave
import json
import queue
from datetime import datetime

# Initialize components
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
model = whisper.load_model("medium")

# Prepare translation pipelines for both directions
# Explicitly run translations on the CPU to avoid failures when no GPU is
# available.  The pipeline defaults to ``device=0`` if CUDA is detected, which
# prevents translations from executing on CPU-only systems and results in no
# subtitles being emitted.
translator_ja_en = pipeline(
    "translation", model="Helsinki-NLP/opus-mt-ja-en", device=0
)
translator_en_ja = pipeline(
    "translation", model="Helsinki-NLP/opus-mt-en-jap", device=0
)

# Default translation direction (Japanese -> English)
translation_direction = 'ja-en'

# Audio config
FS = 16000
CHANNELS = 1
# Record shorter clips so speech recognition and translation run every second
RECORD_SECONDS = 1

# Silence detection threshold (peak amplitude)
silence_threshold = 10

# Debug settings
DEBUG_MODE = True
debug_history = []  # Store debug information
audio_queue = queue.Queue(maxsize=5)

def log_debug(message, data=None):
    """Log debug information with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    debug_entry = {
        "timestamp": timestamp,
        "message": message,
        "data": data
    }
    debug_history.append(debug_entry)
    
    # Keep only last 50 entries to prevent memory issues
    if len(debug_history) > 50:
        debug_history.pop(0)
    
    if DEBUG_MODE:
        print(f"[DEBUG {timestamp}] {message}")
        if data:
            print(f"[DEBUG {timestamp}] Data: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    # Emit debug info to web interface
    socketio.emit("debug_info", debug_entry)

def record_audio_to_file():
    """Record from microphone and return temporary WAV file path and peak amplitude.

    The peak amplitude is used to quickly detect whether the user actually
    spoke.  Whisper can hallucinate text on completely silent audio, so by
    measuring the recorded signal we can skip unnecessary transcription and
    avoid emitting subtitles when no input was provided.
    """
    log_debug("Starting audio recording", {"duration": RECORD_SECONDS, "sample_rate": FS})

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    print("[INFO] Recording...")
    
    start_time = time.time()
    audio = sd.rec(
        int(RECORD_SECONDS * FS),
        samplerate=FS,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    recording_time = time.time() - start_time
    
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    
    log_debug("Audio recording completed", {
        "file_path": tmp_file.name,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "recording_time": f"{recording_time:.2f}s",
        "audio_shape": audio.shape
    })

    with wave.open(tmp_file.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(audio.tobytes())
    
    log_debug("Audio file saved", {"file_size": os.path.getsize(tmp_file.name)})
    return tmp_file.name, peak

def record_audio_loop():
    """Background thread: records audio and queues it for processing."""
    log_debug("Background recording thread started")
    while True:
        wav_path, peak = record_audio_to_file()
        if peak < silence_threshold:
            log_debug("Skipping silent audio", {"peak": peak, "threshold": silence_threshold})
            os.remove(wav_path)
            continue
        audio_queue.put((wav_path, peak))


def recognize_and_translate():
    """Background thread: transcribes, translates, and emits results from queued audio."""
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

            # Log detailed transcription results
            log_debug("Transcription completed", {
                "original_text": original_text,
                "transcription_time": f"{transcription_time:.2f}s",
                "language": result.get("language"),
                "segments_count": len(segments),
                "segments": [
                    {
                        "text": seg.get("text", "").strip(),
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "no_speech_prob": seg.get("no_speech_prob")
                    } for seg in segments
                ]
            })

            if not original_text:
                log_debug("Skipping empty transcription")
                continue

            high_no_speech_segments = [
                seg for seg in segments if seg.get("no_speech_prob", 0) > 0.6
            ]

            # Whisper can assign high ``no_speech_prob`` values to very short
            # recordings even when speech is present.  When ``RECORD_SECONDS`` is
            # set to 1 this would cause valid speech to be skipped entirely.  To
            # ensure subtitles are still produced for these short clips we only
            # apply the "high no-speech" filter when the recording duration is
            # long enough to make the probability reliable.
            if (
                RECORD_SECONDS > 1
                and len(high_no_speech_segments) == len(segments)
                and segments
            ):
                log_debug(
                    "Skipping high no-speech probability segments",
                    {
                        "total_segments": len(segments),
                        "high_no_speech_segments": len(high_no_speech_segments),
                    },
                )
                continue

            # Choose translation pipeline based on current direction
            log_debug("Starting translation", {
                "direction": translation_direction,
                "input_text": original_text
            })

            translation_start = time.time()
            if translation_direction == "ja-en":
                translation_result = translator_ja_en(original_text)
                translation = translation_result[0]["translation_text"]
            else:
                translation_result = translator_en_ja(original_text)
                translation = translation_result[0]["translation_text"]
            translation_time = time.time() - translation_start

            log_debug("Translation completed", {
                "translated_text": translation,
                "translation_time": f"{translation_time:.2f}s",
                "translation_model": "Helsinki-NLP/opus-mt-ja-en" if translation_direction == "ja-en" else "Helsinki-NLP/opus-mt-en-ja"
            })

            print(f"[INFO] Original: {original_text} | Translated: {translation}")

            # Emit results with debug information
            socketio.emit("subtitle", {
                "original": original_text,
                "translated": translation,
                "debug": {
                    "peak_amplitude": peak,
                    "transcription_time": f"{transcription_time:.2f}s",
                    "translation_time": f"{translation_time:.2f}s",
                    "language_detected": result.get("language"),
                    "segments_count": len(segments),
                    "direction": translation_direction
                }
            })

            log_debug("Processing cycle completed successfully")

        except Exception as e:
            error_msg = f"Error in recognition/translation: {str(e)}"
            log_debug("ERROR occurred", {"error": error_msg, "error_type": type(e).__name__})
            print(f"[ERROR] {error_msg}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            audio_queue.task_done()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    log_debug("Client connected")
    print("[INFO] Client connected")
    # Send current debug history to newly connected client
    for entry in debug_history[-10:]:  # Send last 10 entries
        emit("debug_info", entry)

@socketio.on('set_direction')
def on_set_direction(data):
    """Update translation direction based on client selection."""
    global translation_direction
    direction = data.get('direction')
    if direction in ('ja-en', 'en-ja'):
        old_direction = translation_direction
        translation_direction = direction
        log_debug("Translation direction changed", {
            "old_direction": old_direction,
            "new_direction": translation_direction
        })
        print(f"[INFO] Translation direction set to {translation_direction}")

@socketio.on('set_silence_threshold')
def on_set_silence_threshold(data):
    """Update silence detection threshold."""
    global silence_threshold
    threshold = data.get('threshold')
    if isinstance(threshold, (int, float)) and threshold >= 0:
        old_threshold = silence_threshold
        silence_threshold = threshold
        log_debug("Silence threshold changed", {
            "old_threshold": old_threshold,
            "new_threshold": silence_threshold
        })
        print(f"[INFO] Silence threshold set to {silence_threshold}")

@socketio.on('toggle_debug')
def on_toggle_debug(data):
    """Toggle debug mode on/off."""
    global DEBUG_MODE
    DEBUG_MODE = data.get('enabled', True)
    log_debug("Debug mode toggled", {"debug_enabled": DEBUG_MODE})

@socketio.on('clear_debug')
def on_clear_debug():
    """Clear debug history."""
    global debug_history
    debug_history.clear()
    log_debug("Debug history cleared")
    emit("debug_cleared")

@socketio.on('get_debug_history')
def on_get_debug_history():
    """Send current debug history to client."""
    emit("debug_history", {"history": debug_history})

if __name__ == '__main__':
    log_debug("Application starting", {
        "whisper_model": "medium",
        "translation_models": ["Helsinki-NLP/opus-mt-ja-en", "Helsinki-NLP/opus-mt-en-ja"],
        "default_direction": translation_direction,
        "audio_config": {
            "sample_rate": FS,
            "channels": CHANNELS,
            "record_duration": RECORD_SECONDS
        }
    })
    
    # Start background threads for recording and processing
    threading.Thread(target=record_audio_loop, daemon=True).start()
    threading.Thread(target=recognize_and_translate, daemon=True).start()
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=5000)
