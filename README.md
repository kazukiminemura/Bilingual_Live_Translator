# Bilingual Live Translator

This prototype demonstrates a simple bilingual speech translator that uses
OpenAI's Whisper model for speech recognition and lightweight translation models
from the Helsinki-NLP project via `transformers`. It follows the features
described in the [requirement definition](Requirment_Definition.md).

## Features
- Speech-to-text using OpenAI's Whisper model.
- English ⇄ Japanese translation with local Helsinki-NLP models (via
  `transformers`).
- Color coded console output (blue/green for original text, magenta/cyan for translation).

## Prerequisites
These steps ensure your microphone and audio libraries are properly configured.

- **Install PulseAudio Volume Control** to confirm that your microphone is detected:

  ```bash
  sudo apt-get install pavucontrol
  ```

- **Install PortAudio libraries** required by the `sounddevice` Python package:

  ```bash
  sudo apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev
  ```

- **Install the Python `sounddevice` module** and verify available audio devices:

  ```bash
  pip install sounddevice

  python - <<'PY'
  import sounddevice as sd
  print(sd.query_devices())
  print("Default input device:", sd.default.device)
  PY
  ```

If any of these commands fail, check your network configuration or proxy settings.

## Usage
Install dependencies (requires internet access):
```bash
pip install -r requirements.txt
```

The `requirements.txt` file pins a CPU-only build of PyTorch. This avoids
errors such as `Unable to load any of libcudnn...` when CUDA libraries are not
available on the system.

Translate text:
```bash
python app.py --text "Hello" --source en --target ja
```

Transcribe and translate an English audio file:
```bash
python app.py --audio path/to/audio.wav --source en --target ja
# Windows paths using backslashes also work
# python app.py --audio .\\sample_english_audio.wav --source en --target ja
```

Record from the microphone and translate to English (speak Japanese):
```bash
python app.py --mic --duration 5
```

Both commands print the original and translated text with colors for easy distinction.

## FastAPI server
A minimal web API built with FastAPI is provided in `fastapi_app.py`.

Start the server:

```bash
uvicorn fastapi_app:app --reload
```

Translate an audio file by sending a multipart request:

```bash
curl -F "file=@sample_english_audio.wav" -F "source=en" -F "target=ja" \
  http://localhost:8000/translate/audio
```

Translate plain text with JSON:

```bash
curl -X POST http://localhost:8000/translate/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","source":"en","target":"ja"}'
```

## Real-time word-by-word translation (experimental)
An experimental helper class is provided for streaming translation from the
microphone.  It splits incoming audio into short chunks, obtains
word-level timestamps from Whisper and translates each word immediately.

```bash
python - <<'PY'
from realtime_translator import RealTimeWordTranslator
rt = RealTimeWordTranslator()
rt.stream_from_mic()
PY
```

Speak English or Japanese into the microphone and the console will emit the
recognized word alongside its translation.
