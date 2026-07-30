# Ethnicity Detector API
# Deploy this Space to host the same DeepFace analyzer used by the WordPress plugin.
# Runtime: Docker

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt analyzer.py api.py ./
RUN pip install --no-cache-dir -r requirements.txt

ENV ALLOW_ALL_ORIGINS=1
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
