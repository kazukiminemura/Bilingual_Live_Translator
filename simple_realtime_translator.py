import asyncio
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from transformers import pipeline

# Whisperモデル (小さめ: "tiny", "small", "medium" など選択可)
asr_model = WhisperModel("medium", device="cpu")  # GPUがあれば "cuda"

# 翻訳モデル (英語→日本語)
#translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-jap")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-jap-en")

# 音声設定
SAMPLE_RATE = 16000
BLOCK_DURATION = 3  # 秒ごとに処理（短くすれば細かく翻訳される）
BLOCK_SIZE = SAMPLE_RATE * BLOCK_DURATION

def process_audio(audio_chunk: np.ndarray):
    """1ブロック分の音声をテキスト化＋翻訳して表示"""
    segments, _ = asr_model.transcribe(audio_chunk, beam_size=5)
    text = " ".join([seg.text for seg in segments]).strip()

    if text:
        # 翻訳
        translated = translator(text)[0]["translation_text"]
        print(f"[音声] {text}")
        print(f"[翻訳] {translated}")
        print("-" * 40)

def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    audio_chunk = np.copy(indata[:, 0])  # モノラルで取得
    process_audio(audio_chunk)

async def main():
    print("🎤 リアルタイム翻訳開始 (Ctrl+Cで終了)")
    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE,
                        dtype=np.float32):
        while True:
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 終了しました")
