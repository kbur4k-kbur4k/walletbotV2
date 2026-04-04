# ---- builder stage ----
FROM python:3.11-slim AS builder
WORKDIR /build

COPY requirements.txt pyproject.toml ./
COPY wallet_manager/ ./wallet_manager/
RUN grep -v '^torch' requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir --user -r requirements-docker.txt \
    && pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
    && pip install --no-cache-dir --user .

# ---- runtime stage ----
FROM python:3.11-slim AS runtime
WORKDIR /app

# Create non-root user
RUN groupadd --gid 10001 wallet \
    && useradd --uid 10001 --gid wallet --no-create-home --shell /bin/false wallet

# Copy installed packages and source
COPY --from=builder /root/.local /home/wallet/.local
COPY --chown=wallet:wallet wallet_manager/ ./wallet_manager/
COPY --chown=wallet:wallet pyproject.toml ./

ENV PATH=/home/wallet/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Writable runtime directories
RUN mkdir -p /app/logs /app/data \
    && chown -R wallet:wallet /app/logs /app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/health')" || exit 1

USER wallet

CMD ["python", "-m", "wallet_manager.orchestrator"]
