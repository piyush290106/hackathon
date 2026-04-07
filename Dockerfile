# ── Supply Chain OpenEnv — Dockerfile ─────────────────────────────────────────
# Compatible with Hugging Face Spaces (runs as non-root on port 7860)
# and standard OpenEnv evaluation (port 8000).

FROM python:3.11-slim

LABEL maintainer="piyush290106"
LABEL description="Supply Chain OpenEnv — real-world AI environment"

# HF Spaces uses port 7860; override with ENV PORT=8000 for local dev
ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HF Spaces runs as non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE ${PORT}

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
