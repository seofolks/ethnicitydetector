"""Shared DeepFace analysis used by Streamlit app and FastAPI."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to BGR format using NumPy."""
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1]


def analyze_image(image_bgr: np.ndarray) -> dict[str, Any]:
    """Run DeepFace analysis for emotion and ethnicity."""
    from deepface import DeepFace

    result = DeepFace.analyze(
        img_path=image_bgr,
        actions=["emotion", "race"],
        enforce_detection=False,
    )
    if isinstance(result, list):
        result = result[0]
    return result


def analyze_pil(image: Image.Image) -> dict[str, Any]:
    """Analyze a PIL image and return a JSON-serializable summary."""
    raw = analyze_image(pil_to_bgr(image))
    emotion_scores = raw.get("emotion") or {}
    race_scores = raw.get("race") or {}
    return {
        "dominant_emotion": raw.get("dominant_emotion"),
        "dominant_ethnicity": raw.get("dominant_race"),
        "emotion_scores": {k: round(float(v), 4) for k, v in emotion_scores.items()},
        "ethnicity_scores": {k: round(float(v), 4) for k, v in race_scores.items()},
    }
