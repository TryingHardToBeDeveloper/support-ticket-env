# ── Dockerfile: Customer Support Ticket Resolution Environment ──
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app/support_ticket_env
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "support_ticket_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
