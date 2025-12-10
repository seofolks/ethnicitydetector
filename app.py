import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image


st.set_page_config(page_title="Ethnicity & Emotion Detector", page_icon="📷")


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to OpenCV BGR format using NumPy."""
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1]  # Reverse channels for BGR


def _analyze_image(image_bgr: np.ndarray) -> dict:
    """Run DeepFace analysis for emotion and ethnicity."""
    result = DeepFace.analyze(
        img_path=image_bgr,
        actions=["emotion", "race"],
        enforce_detection=False,
    )
    # DeepFace may return a list when multiple faces are found.
    if isinstance(result, list):
        result = result[0]
    return result


def _render_scores(title: str, scores: dict | None) -> None:
    if not scores:
        return
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    labels, values = zip(*sorted_items)
    st.markdown(f"**{title}**")
    st.table({"label": labels, "score": [round(v, 4) for v in values]})


def main() -> None:
    st.title("Ethnicity & Emotion Detector")
    st.write(
        "Upload a photo or take one with your webcam to estimate the visible "
        "emotion and ethnicity. Results are heuristic and should not be used "
        "for sensitive decisions."
    )

    input_mode = st.radio("Choose input source", ["Upload image", "Use webcam"], horizontal=True)

    image: Image.Image | None = None
    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload an image file", type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded:
            image = Image.open(uploaded)
    else:
        camera_photo = st.camera_input("Take a photo")
        if camera_photo:
            image = Image.open(camera_photo)

    if image:
        st.image(image, caption="Selected image", use_container_width=True)

        if st.button("Analyze"):
            with st.spinner("Running DeepFace... this can take a moment on first run."):
                try:
                    result = _analyze_image(_pil_to_bgr(image))
                except Exception as exc:  # pragma: no cover - surface errors to UI
                    st.error(f"Analysis failed: {exc}")
                    return

            st.success("Analysis complete.")
            st.write(f"Dominant emotion: **{result.get('dominant_emotion', 'n/a')}**")
            st.write(f"Dominant ethnicity: **{result.get('dominant_race', 'n/a')}**")

            _render_scores("Emotion scores", result.get("emotion"))
            _render_scores("Ethnicity scores", result.get("race"))

            st.caption("Powered by DeepFace. Accuracy depends on lighting, framing, and model limits.")
    else:
        st.info("Upload or capture an image to get started.")


if __name__ == "__main__":
    main()

