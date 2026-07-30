"""FastAPI backend exposing the same DeepFace analyzer as the Streamlit app."""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from analyzer import analyze_pil

app = FastAPI(title="Ethnicity & Emotion Detector API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://whatismyethnicity.com,https://tool.whatismyethnicity.com,http://localhost,http://127.0.0.1",
    ).split(",")
    if origin.strip()
]

cors_origins = ["*"] if os.getenv("ALLOW_ALL_ORIGINS") == "1" else allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "").strip()


def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
) -> dict:
    _check_api_key(x_api_key)

    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    try:
        return analyze_pil(image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
