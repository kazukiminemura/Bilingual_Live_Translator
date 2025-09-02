# Bilingual Live Translator

Bilingual Live Translator captures spoken English or Japanese, converts it to text with OpenAI's Whisper model, and immediately translates the result using the `facebook/m2m100_418M` multilingual model from Hugging Face. The project exposes a simple command line interface and a minimal FastAPI web application for experimenting with real‑time translation.

## Requirements

* Python 3.10+
* `ffmpeg` for audio handling
* A microphone for live input
* A GPU is recommended for translation (the default device is GPU 0)

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

Translate a short piece of text:

```bash
python app.py --text "Hello" --source en --target ja
```

Transcribe and translate an audio file:

```bash
python app.py --audio path/to/audio.wav --source ja --target en
```

Record from the microphone for five seconds and translate to Japanese:

```bash
python app.py --mic --duration 5 --source en --target ja
```

## Web application

The `app.py` module also contains a FastAPI application with a WebSocket endpoint. Start the server with:

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in a browser and speak into the microphone to see the detected text and translation appear in real time.

## Backend selection

The speech recognizer and translation models can run on different hardware backends.
Set the environment variables below to choose the device:

```bash
export ASR_BACKEND=cpu            # or gpu, npu
export TRANSLATION_BACKEND=gpu    # or cpu, npu
```

You can also change these backends at runtime from the web interface.

## License

MIT

