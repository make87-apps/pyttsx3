# Multi-stage build for Text-to-Speech application
FROM ghcr.io/make87/debian:bookworm AS base-image

ARG VIRTUAL_ENV=/make87/venv

# Install build dependencies and eSpeak for TTS
RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
        build-essential \
        python3-pip \
        python3 \
        libpython3-dev \
        python3-venv \
        espeak espeak-data libespeak1 libespeak-dev \
        # Optional: Add ffmpeg for advanced audio processing if using gtts fallback
        # ffmpeg \
    && python3 -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml first for better caching of dependencies
COPY pyproject.toml ./

# Install dependencies only (without the actual package)
RUN ${VIRTUAL_ENV}/bin/pip install \
        # eSpeak-NG for offline TTS
        "espeak-ng>=0.1.0" \
        "libespeak-ng1" \
        "pyttsx3>=2.90"

# Copy application code
COPY . .

# Install the application (production install, not editable)
RUN set -eux; \
    if [ -f ./pip.conf ]; then \
      echo "Found pip.conf, pointing PIP_CONFIG_FILE at it"; \
      export PIP_CONFIG_FILE="$(pwd)/pip.conf"; \
    else \
      echo "No pip.conf found, using default indexes"; \
    fi; \
    ${VIRTUAL_ENV}/bin/pip install .

# Final runtime image
FROM ghcr.io/make87/python3-debian12:latest

ARG VIRTUAL_ENV=/make87/venv

# Install only runtime dependencies for offline TTS
RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
        # eSpeak-NG runtime for offline TTS
        espeak-ng libespeak-ng1 libespeak1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from build stage
COPY --from=base-image ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Optional: Copy any additional runtime files if needed
# COPY --from=base-image /app/config /app/config

WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Health check for TTS functionality
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import pyttsx3; engine = pyttsx3.init(); print('TTS engine initialized successfully')" || exit 1

ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]