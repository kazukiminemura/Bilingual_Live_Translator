from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil
import os

from app import BilingualLiveTranslator

app = FastAPI(title="Bilingual Live Translator")

translator = BilingualLiveTranslator()


class TextTranslationRequest(BaseModel):
    text: str
    source: str
    target: str


@app.post("/translate/audio")
async def translate_audio(
    file: UploadFile = File(...),
    source: Optional[str] = Form(None),
    target: Optional[str] = Form(None),
):
    """Transcribe and translate an uploaded audio file.

    Parameters
    ----------
    file: UploadFile
        Audio file to process.
    source: str, optional
        Source language code ("en" or "ja"). If omitted, auto detect.
    target: str, optional
        Target language code. If omitted, automatically pick the opposite
        language of the detected source.
    """
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pair = translator.process_audio(tmp_path, source, target)
    finally:
        os.remove(tmp_path)

    return {
        "original": pair.text,
        "translation": pair.translation,
        "detected_language": pair.language_detected,
    }


@app.post("/translate/text")
async def translate_text(req: TextTranslationRequest):
    """Translate plain text between English and Japanese."""
    translation = translator.translate_text(req.text, req.source, req.target)
    return {"original": req.text, "translation": translation}


@app.get("/")
async def root():
    return {"message": "Bilingual Live Translator API"}
