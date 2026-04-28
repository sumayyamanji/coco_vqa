# syntax=docker/dockerfile:1
FROM python:3.10-slim

# ---- system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies (cached layer) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# ---- copy project ----
COPY . .

# ---- Gradio port ----
EXPOSE 7860

# ---- default command ----
CMD ["python", "demo/app.py"]
