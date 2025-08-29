"""Real-time word-by-word translator using microphone input.

This module provides a small helper class :class:`RealTimeWordTranslator`
that extends :class:`~app.BilingualLiveTranslator` with the ability to
listen to the microphone and emit translations for each recognized word in
near real time.  Audio is processed in short blocks and Whisper's
``word_timestamps`` option is used to obtain per-word transcription.  Each
word is then immediately translated using the lightweight Helsinki-NLP
models already utilised by the project.

The implementation is intentionally simple and intended for prototyping
purposes.  It runs entirely on the CPU and prints the original/translated
words to the console as they are produced.
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from app import BilingualLiveTranslator


class RealTimeWordTranslator(BilingualLiveTranslator):
    """Translate microphone input word-by-word in real time."""

    def stream_from_mic(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        block_duration: float = 1.0,
    ) -> None:
        """Listen to the microphone and translate audio blocks.

        Parameters
        ----------
        source:
            Source language code (``"en"`` or ``"ja"``).  If ``None`` the
            language is detected automatically by Whisper.
        target:
            Target language code.  If ``None`` the opposite language is used.
        block_duration:
            Size of the audio blocks in seconds.  Smaller values yield lower
            latency at the cost of higher processing overhead.
        """

        sample_rate = 16000
        block_frames = int(sample_rate * block_duration)
        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_, status):  # pragma: no cover -
            # Queue audio data for the main thread
            audio_queue.put(indata.copy())

        print("🎧 Listening... Press Ctrl+C to stop.")
        with sd.InputStream(
            callback=_callback, samplerate=sample_rate, channels=1
        ):
            buffer = np.zeros((0,), dtype=np.float32)
            try:
                while True:
                    data = audio_queue.get()
                    buffer = np.concatenate([buffer, data[:, 0]])
                    if len(buffer) >= block_frames:
                        segment = buffer[:block_frames]
                        buffer = buffer[block_frames:]
                        result = self.whisper_model.transcribe(
                            segment,
                            language=source,
                            fp16=False,
                            word_timestamps=True,
                        )
                        detected = source or result["language"]
                        tgt = target or ("ja" if detected == "en" else "en")
                        for seg in result.get("segments", []):
                            for word in seg.get("words", []):
                                trans = self.translate_text(
                                    word["word"], detected, tgt
                                )
                                print(f"{word['word']} -> {trans}", flush=True)
            except KeyboardInterrupt:  # pragma: no cover - manual stop
                print("\n🛑 Stopped.")

