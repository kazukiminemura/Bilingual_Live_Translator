import argparse
from dataclasses import dataclass

from colorama import Fore, Style
import speech_recognition as sr
from transformers import pipeline


@dataclass
class TranslationPair:
    text: str
    translation: str


class BilingualLiveTranslator:
    """Simple bilingual translator using Google Web Speech for transcription and
    lightweight translation models for real-time conversion."""

    def __init__(self) -> None:
        """Initialize recognizer and translation pipelines."""
        self.recognizer = sr.Recognizer()
        self.translators = {
            ("en", "ja"): pipeline("translation", model="Helsinki-NLP/opus-mt-en-ja"),
            ("ja", "en"): pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en"),
        }

    def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribe audio using the Google Web Speech API.

        Parameters
        ----------
        audio_path: str
            Path to audio file.
        language: str
            Source language code (e.g., ``"en"`` or ``"ja"``).
        """
        lang = "en-US" if language.startswith("en") else "ja-JP"
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)
        return self.recognizer.recognize_google(audio, language=lang)

    def translate_text(self, text: str, source: str, target: str) -> str:
        """Translate text between languages using local translation models.

        Currently supports English⇄Japanese via Helsinki-NLP's opus-mt models.
        """
        key = (source[:2], target[:2])
        translator = self.translators.get(key)
        if translator is None:
            raise ValueError("Unsupported language pair")
        return translator(text)[0]["translation_text"]

    def process_audio(self, audio_path: str, source: str, target: str) -> TranslationPair:
        """Transcribe and translate an audio file."""
        original = self.transcribe(audio_path, source)
        translated = self.translate_text(original, source, target)
        return TranslationPair(original, translated)


def color_print(pair: TranslationPair, source: str, target: str) -> None:
    """Print original and translation with color coding."""
    src_color = Fore.BLUE if source.startswith("en") else Fore.GREEN
    tgt_color = Fore.MAGENTA if target.startswith("ja") else Fore.CYAN
    print(f"{src_color}{pair.text}{Style.RESET_ALL}")
    print(f"{tgt_color}{pair.translation}{Style.RESET_ALL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bilingual Live Translator")
    parser.add_argument("--audio", help="Path to input audio file")
    parser.add_argument("--text", help="Translate given text instead of audio")
    parser.add_argument("--source", default="en", help="Source language code")
    parser.add_argument("--target", default="ja", help="Target language code")
    args = parser.parse_args()

    if args.audio is None and args.text is None:
        parser.error("Either --audio or --text must be provided")

    translator = BilingualLiveTranslator()

    if args.audio:
        pair = translator.process_audio(args.audio, args.source, args.target)
    else:
        translation = translator.translate_text(args.text, args.source, args.target)
        pair = TranslationPair(args.text, translation)

    color_print(pair, args.source, args.target)


if __name__ == "__main__":
    main()
