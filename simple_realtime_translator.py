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
translator_ja_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
translator_en_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ja")

# Default translation direction (Japanese -> English)
translation_direction = 'ja-en'

# Audio config
FS = 16000
CHANNELS = 1
RECORD_SECONDS = 3

def record_audio_to_file():
    """Record from microphone and return temporary WAV file path."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    print("[INFO] Recording...")
    audio = sd.rec(int(RECORD_SECONDS * FS), samplerate=FS, channels=CHANNELS, dtype='int16')
    sd.wait()
    with wave.open(tmp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(audio.tobytes())
    print("[INFO] Recording saved to:", tmp_file.name)
    return tmp_file.name

def recognize_and_translate():
    """Background thread: records, transcribes, translates, and emits results."""
    while True:
        try:
            wav_path = record_audio_to_file()
            result = model.transcribe(wav_path)
            original_text = result['text']

            # Choose translation pipeline based on current direction
            if translation_direction == 'ja-en':
                translation = translator_ja_en(original_text)[0]['translation_text']
            else:
                translation = translator_en_ja(original_text)[0]['translation_text']

            print(f"[INFO] Original: {original_text} | Translated: {translation}")
            socketio.emit('subtitle', {'original': original_text, 'translated': translation})
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
