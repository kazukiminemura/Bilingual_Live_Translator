# Bilingual Live Translator

This prototype demonstrates a simple bilingual speech translator that uses **OpenVINO**
for both speech recognition and machine translation.  It follows the features
described in the [requirement definition](Requirment_Definition.md).

## Features
- Speech-to-text using Whisper converted for OpenVINO.
- English ⇄ Japanese translation with MarianMT models accelerated by OpenVINO.
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

Transcribe and translate an audio file:
```bash
python app.py --audio path/to/audio.wav --source ja --target en
```

Both commands print the original and translated text with colors for easy distinction.
