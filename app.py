import argparse
from dataclasses import dataclass

from colorama import Fore, Style

try:
    from nemo.collections.asr.models import ASRModel
    from optimum.intel import OVModelForSeq2SeqLM
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - handled during runtime
    ASRModel = OVModelForSeq2SeqLM = AutoTokenizer = None  # type: ignore


@dataclass
class TranslationPair:
    text: str
    translation: str


class BilingualLiveTranslator:
    """Simple bilingual translator using NeMo and OpenVINO for inference."""

    def __init__(self, device: str = "cpu"):
        if ASRModel is None or OVModelForSeq2SeqLM is None:
            raise RuntimeError("Required packages are not installed. See requirements.txt")
        # Load QuartzNet for speech recognition (English only)
        self.asr_model = ASRModel.from_pretrained(model_name="QuartzNet15x5Base-En")
        if device.lower() != "cpu":
            self.asr_model.to(device)
        # Load translation models
        self.en_ja_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ja")
        self.en_ja_model = OVModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ja")
        self.ja_en_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        self.ja_en_model = OVModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

    def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribe audio using QuartzNet.

        Parameters
        ----------
        audio_path: str
            Path to audio file.
        language: str
            Source language code. Only English ("en") is supported.
        """
        if not language.startswith("en"):
            raise ValueError("QuartzNet15x5Base-En only supports English audio")
        transcript = self.asr_model.transcribe(paths2audio_files=[audio_path])[0]
        return transcript.strip()

    def translate_text(self, text: str, source: str, target: str) -> str:
        """Translate text between English and Japanese."""
        if source.startswith("en") and target.startswith("ja"):
            tok, model = self.en_ja_tok, self.en_ja_model
        elif source.startswith("ja") and target.startswith("en"):
            tok, model = self.ja_en_tok, self.ja_en_model
        else:
            raise ValueError("Unsupported language pair")
        inputs = tok(text, return_tensors="pt")
        outputs = model.generate(**inputs)
        return tok.decode(outputs[0], skip_special_tokens=True)

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
