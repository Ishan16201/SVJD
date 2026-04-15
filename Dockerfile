# --- Stage 1: Builder ---
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/apt/lists/*

WORKDIR /app

# Copy and install requirements
# Note: We install dependencies into a local directory for easy copying
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    numba \
    pandas \
    aiohttp \
    MetaTrader5 \
    silicon-metatrader5

# --- Stage 2: Runtime ---
FROM python:3.11-slim

# Install runtime dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MT5_LOGIN="" \
    MT5_PASSWORD="" \
    MT5_SERVER="" \
    KELLY_FRACTION=0.20 \
    DRAWDOWN_HARD_LIMIT=0.06 \
    LOG_LEVEL="INFO"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Expose healthcheck port
EXPOSE 8080

# Healthcheck configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entrypoint
ENTRYPOINT ["python", "main.py"]
