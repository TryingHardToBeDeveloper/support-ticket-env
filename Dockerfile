# ── Dockerfile: Customer Support Ticket Resolution Environment ──
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

COPY --from=builder /usr/local /usr/local

RUN groupadd --system app && useradd --system --gid app --create-home app

# Copy source
COPY . /app/support_ticket_env
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true
ENV SUPPORT_ENV_MODE=production
WORKDIR /app/support_ticket_env
RUN chown -R app:app /app
USER app

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=5)"]

CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
