# Realtime Speech Translation Web App (Flask + WebSocket + Whisper + HuggingFace)

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
    "translation", model="Helsinki-NLP/opus-mt-ja-en", device=-1
)
translator_en_ja = pipeline(
    "translation", model="Helsinki-NLP/opus-mt-en-jap", device=-1
)

# Default translation direction (Japanese -> English)
translation_direction = 'ja-en'

# Audio config
FS = 16000
CHANNELS = 1
RECORD_SECONDS = 3

def record_audio_to_file():
    """Record from microphone and return temporary WAV file path and peak amplitude.

    The peak amplitude is used to quickly detect whether the user actually
    spoke.  Whisper can hallucinate text on completely silent audio, so by
    measuring the recorded signal we can skip unnecessary transcription and
    avoid emitting subtitles when no input was provided.
    """

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    print("[INFO] Recording...")
    audio = sd.rec(
        int(RECORD_SECONDS * FS),
        samplerate=FS,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    peak = float(np.max(np.abs(audio)))

    with wave.open(tmp_file.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(audio.tobytes())
    print("[INFO] Recording saved to:", tmp_file.name)
    return tmp_file.name, peak

def recognize_and_translate():
    """Background thread: records, transcribes, translates, and emits results."""
    while True:
        try:
            wav_path, peak = record_audio_to_file()

            # Skip processing when the recorded audio is effectively silent
            if peak < 500:  # empirical threshold for background noise
                os.remove(wav_path)
                continue

            result = model.transcribe(wav_path)
            original_text = result.get("text", "").strip()

            # Whisper may still output hallucinated text for silent segments.
            # Check the no_speech_prob of each segment and ignore if all
            # indicate silence or if the text itself is empty.
            segments = result.get("segments", [])
            if not original_text or all(
                seg.get("no_speech_prob", 0) > 0.6 for seg in segments
            ):
                os.remove(wav_path)
                continue

            # Choose translation pipeline based on current direction
            if translation_direction == "ja-en":
                translation = translator_ja_en(original_text)[0]["translation_text"]
            else:
                translation = translator_en_ja(original_text)[0]["translation_text"]

            print(
                f"[INFO] Original: {original_text} | Translated: {translation}"
            )
            socketio.emit(
                "subtitle", {"original": original_text, "translated": translation}
            )
            os.remove(wav_path)
        except Exception as e:
            print("[ERROR]", e)
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    print("[INFO] Client connected")


@socketio.on('set_direction')
def on_set_direction(data):
    """Update translation direction based on client selection."""
    global translation_direction
    direction = data.get('direction')
    if direction in ('ja-en', 'en-ja'):
        translation_direction = direction
        print(f"[INFO] Translation direction set to {translation_direction}")

if __name__ == '__main__':
    # Start background transcription thread
    threading.Thread(target=recognize_and_translate, daemon=True).start()
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=5000)
