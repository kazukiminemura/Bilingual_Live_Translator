import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from colorama import Fore, Style, init
import whisper
from transformers import pipeline
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@dataclass
class TranslationPair:
    """データクラス for storing original text and translation"""
    text: str
    translation: str
    language_detected: str = ""

class BilingualLiveTranslator:
    """Simple bilingual translator using Whisper for transcription and
    lightweight translation models for real-time conversion.
    
    Supports:
    - English speech → Japanese subtitles
    - Japanese speech → English text
    """
    
    def __init__(self, whisper_model_size: str = "base") -> None:
        """Initialize Whisper model and translation pipelines.
        
        Parameters
        ----------
        whisper_model_size : str
            Whisper model size: tiny, base, small, medium, large
        """
        logger.debug("Initializing translator with Whisper model size: %s", whisper_model_size)

        print(f"🔄 Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)

        print("🔄 Loading translation models...")
        self.translators = {
            ("en", "ja"): pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-en-jap",
                device=-1  # Use CPU
            ),
            ("ja", "en"): pipeline(
                "translation", 
                model="Helsinki-NLP/opus-mt-jap-en",
                device=-1  # Use CPU
            ),
        }
        print("✅ Models loaded successfully!")
        logger.debug("Available translators: %s", list(self.translators.keys()))

    def transcribe(self, audio_path: str, language: str = None) -> tuple[str, str]:
        """Transcribe audio using OpenAI's Whisper model.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        language : str, optional
            Source language code. If None, auto-detect
            
        Returns
        -------
        tuple[str, str]
            Transcribed text and detected language
        """
        # Normalize path for cross-platform compatibility
        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.debug("Starting transcription for %s", audio_path)
        print(f"🎵 Transcribing audio: {audio_path}")
        
        if language:
            lang = language[:2]
            result = self.whisper_model.transcribe(str(audio_path), language=lang)
            detected_lang = lang
            logger.debug("Used specified language: %s", lang)
        else:
            # Auto-detect language
            result = self.whisper_model.transcribe(str(audio_path))
            detected_lang = result["language"]
            logger.debug("Auto-detected language: %s", detected_lang)

        text = result["text"].strip()
        print(f"🗣️  Detected language: {detected_lang}")
        logger.debug("Transcription result: %s", text)
        
        return text, detected_lang

    def translate_text(self, text: str, source: str, target: str) -> str:
        """Translate text between languages using local translation models.
        
        Currently supports:
        - English ⇄ Japanese via Helsinki-NLP's opus-mt models
        
        Parameters
        ----------
        text : str
            Text to translate
        source : str
            Source language code
        target : str
            Target language code
            
        Returns
        -------
        str
            Translated text
        """
        if not text.strip():
            return ""
            
        key = (source[:2], target[:2])
        translator = self.translators.get(key)
        
        if translator is None:
            supported_pairs = list(self.translators.keys())
            raise ValueError(f"Unsupported language pair: {source}→{target}. "
                           f"Supported pairs: {supported_pairs}")

        logger.debug("Translating text from %s to %s", source, target)
        print(f"🔄 Translating {source}→{target}...")

        tokenizer = translator.tokenizer
        max_len = getattr(tokenizer, "model_max_length", 512)
        tokens = tokenizer.encode(text, add_special_tokens=False)

        translations = []
        if len(tokens) > max_len:
            logger.debug("Input exceeds max length (%d > %d). Splitting...", len(tokens), max_len)
            start = 0
            while start < len(tokens):
                end = min(start + max_len, len(tokens))
                chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
                result = translator(chunk)
                translations.append(result[0]["translation_text"])
                start = end
        else:
            result = translator(text)
            translations.append(result[0]["translation_text"])

        translated = " ".join(translations)
        logger.debug("Translation result: %s", translated)
        return translated

    def process_audio(self, audio_path: str, source: str = None, target: str = None) -> TranslationPair:
        """Transcribe and translate an audio file.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        source : str, optional
            Source language. If None, auto-detect
        target : str, optional
            Target language. If None, auto-determine based on detected source
            
        Returns
        -------
        TranslationPair
            Original text and translation
        """
        logger.debug("Processing audio file: %s", audio_path)

        # Transcribe audio
        original_text, detected_lang = self.transcribe(audio_path, source)
        
        if not original_text:
            return TranslationPair("", "", detected_lang)
        
        # Auto-determine target language if not specified
        if target is None:
            if detected_lang == "en":
                target = "ja"
            elif detected_lang == "ja":
                target = "en"
            else:
                raise ValueError(f"Cannot auto-determine target for language: {detected_lang}")

        logger.debug("Detected language: %s, target language: %s", detected_lang, target)
        
        # Translate
        translated_text = self.translate_text(original_text, detected_lang, target)
        logger.debug("Original text: %s", original_text)
        logger.debug("Translated text: %s", translated_text)

        return TranslationPair(original_text, translated_text, detected_lang)

def color_print(pair: TranslationPair, source: str, target: str) -> None:
    """Print original and translation with color coding and formatting."""
    print("\n" + "="*60)
    
    # Source text
    src_color = Fore.CYAN if source.startswith("en") else Fore.GREEN
    src_label = "🇺🇸 English" if source.startswith("en") else "🇯🇵 Japanese"
    print(f"{src_color}【{src_label}】{Style.RESET_ALL}")
    print(f"{src_color}{pair.text}{Style.RESET_ALL}")
    
    print()
    
    # Translation
    tgt_color = Fore.MAGENTA if target.startswith("ja") else Fore.YELLOW
    tgt_label = "🇯🇵 Japanese" if target.startswith("ja") else "🇺🇸 English"
    print(f"{tgt_color}【{tgt_label}】{Style.RESET_ALL}")
    print(f"{tgt_color}{pair.translation}{Style.RESET_ALL}")
    
    print("="*60)

def validate_audio_file(audio_path: str) -> None:
    """Validate audio file exists and has supported extension."""
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    supported_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.avi', '.mov'}
    file_ext = path.suffix.lower()

    if file_ext not in supported_extensions:
        print(f"⚠️  Warning: {file_ext} might not be supported. "
              f"Supported formats: {', '.join(supported_extensions)}")

def main() -> None:
    """Main function with enhanced argument parsing and error handling."""
    # Initialize colorama for Windows compatibility
    init()
    
    parser = argparse.ArgumentParser(
        description="🎯 Bilingual Live Translator - English ⇄ Japanese",
        epilog="""
Examples:
  # English speech → Japanese subtitles
  python translator.py --audio english_speech.wav
  
  # Japanese speech → English text  
  python translator.py --audio japanese_speech.wav
  
  # Text translation
  python translator.py --text "Hello world" --source en --target ja
  
  # Auto-detect language from audio
  python translator.py --audio mixed_language.wav --auto
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", help="Path to input audio file")
    input_group.add_argument("--text", help="Translate given text instead of audio")
    
    # Language options
    parser.add_argument("--source", default=None, 
                       help="Source language code (en/ja). Auto-detect if not specified")
    parser.add_argument("--target", default=None,
                       help="Target language code (en/ja). Auto-determine if not specified")
    parser.add_argument("--auto", action="store_true",
                       help="Auto-detect source language and determine target")
    
    # Model options
    parser.add_argument("--whisper-model", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    
    # Output options
    parser.add_argument("--output", help="Save translation to file")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode - minimal output")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    logger.debug("Debug logging is enabled")
    
    try:
        # Initialize translator
        if not args.quiet:
            print(f"{Fore.CYAN}🚀 Starting Bilingual Live Translator{Style.RESET_ALL}")
        
        translator = BilingualLiveTranslator(args.whisper_model)
        
        # Process input
        if args.audio:
            validate_audio_file(args.audio)
            
            if args.auto or (args.source is None):
                # Auto-detect mode
                pair = translator.process_audio(args.audio, None, args.target)
                detected_source = pair.language_detected
                final_target = args.target or ("ja" if detected_source == "en" else "en")
            else:
                # Manual language specification
                pair = translator.process_audio(args.audio, args.source, args.target)
                detected_source = args.source or pair.language_detected
                final_target = args.target or ("ja" if detected_source == "en" else "en")
                
        else:  # Text translation
            if args.source is None or args.target is None:
                parser.error("Both --source and --target must be specified for text translation")
            
            translation = translator.translate_text(args.text, args.source, args.target)
            pair = TranslationPair(args.text, translation)
            detected_source = args.source
            final_target = args.target
        
        # Display results
        if not args.quiet:
            color_print(pair, detected_source, final_target)
        else:
            print(pair.translation)
        
        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(f"Original ({detected_source}): {pair.text}\n")
                f.write(f"Translation ({final_target}): {pair.translation}\n")
            print(f"💾 Saved to: {args.output}")
            
    except FileNotFoundError as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
    except ValueError as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
