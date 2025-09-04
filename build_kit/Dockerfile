# ---------- Build stage ----------
    FROM ghcr.io/make87/debian:bookworm AS builder

    ARG VIRTUAL_ENV=/make87/venv
    
    RUN apt-get update \
     && apt-get install --no-install-suggests --no-install-recommends -y \
          build-essential python3 python3-venv python3-pip libpython3-dev \
     && python3 -m venv ${VIRTUAL_ENV} \
     && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel \
     && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    COPY pyproject.toml ./
    # Install only Python deps here for layer caching
    RUN ${VIRTUAL_ENV}/bin/pip install pyttsx3>=2.90
    
    COPY . .
    RUN ${VIRTUAL_ENV}/bin/pip install .
    
    # ---------- Runtime stage ----------
    # Use a Debian image that has /bin/sh so apt works
    FROM debian:bookworm-slim AS runtime
    
    ARG VIRTUAL_ENV=/make87/venv
    
    # eSpeak-NG runtime (no legacy libespeak1), plus minimal Python so venv is compatible
    RUN apt-get update \
     && apt-get install --no-install-suggests --no-install-recommends -y \
          python3 python3-venv \
          espeak-ng libespeak-ng1 \
     && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Bring over the virtualenv from builder
    COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
    WORKDIR /app
    
    ENV PYTHONPATH=/app
    ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
    
    # Lightweight healthcheck that exercises pyttsx3 init
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
      CMD python3 -c "import pyttsx3; pyttsx3.init(); print('ok')" || exit 1
    
    ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]
    