# WordPress Plugin: Ethnicity Detector

## What this is
A branded WordPress UI (`[ethnicity_detector]`) that calls the **same DeepFace code** in this GitHub repo through `api.py`.

Streamlit can stay as a backup. The plugin is the user-facing tool.

## Install plugin on WordPress
1. Zip the folder `wordpress-plugin/ethnicity-detector/`
2. WP Admin → Plugins → Add New → Upload Plugin → activate
3. Or upload that folder to `wp-content/plugins/ethnicity-detector/`
4. Add shortcode to a page: `[ethnicity_detector]`
5. Settings → Ethnicity Detector → set **DeepFace API URL**

## Host the DeepFace API (required for analysis)
Streamlit Community Cloud cannot serve this API. Deploy `api.py` somewhere Python-capable:

### Option A: Hugging Face Spaces (simple)
1. Create a new Docker Space
2. Point it at this repo (or upload `Dockerfile`, `api.py`, `analyzer.py`, `requirements.txt`, `packages.txt`)
3. After deploy, copy the Space URL into plugin settings (no trailing slash), e.g. `https://username-ethnicitydetector.hf.space`

### Option B: Railway / Render / VPS
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Optional env vars:
- `API_KEY` – if set, plugin must use the same key
- `ALLOWED_ORIGINS` – comma-separated origins
- `ALLOW_ALL_ORIGINS=1` – allow all CORS origins

## Flow
Browser → WordPress REST (`/wp-json/ethnicity-detector/v1/analyze`) → FastAPI `/analyze` → DeepFace
