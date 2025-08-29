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

## Usage
Install dependencies (requires internet access):
```bash
pip install -r requirements.txt
```

Translate text:
```bash
python app.py --text "Hello" --source en --target ja
```

Transcribe and translate an English audio file:
```bash
python app.py --audio path/to/audio.wav --source en --target ja
```

Both commands print the original and translated text with colors for easy distinction.
