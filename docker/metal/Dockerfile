# syntax=docker/dockerfile:1
# Sovereign Runtime image: metal-arm64 host-agent bridge (design.md §2.6).
# Inference runs on the macOS host through llama.cpp; this small arm64 image
# exposes the same runtime contract without carrying a local vLLM CPU build.
#
# Build context: repository root — docker build -f docker/metal/Dockerfile .

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

COPY pyproject.toml requirements-agent.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/sovereign/
COPY lazarus /opt/sovereign/lazarus
RUN python3 -m pip install --no-cache-dir --require-hashes \
        --requirement /opt/sovereign/requirements-agent.lock && \
    python3 -m pip install --no-cache-dir --no-deps /opt/sovereign

RUN printf '#!/bin/sh\nexec python3 -m lazarus.appliance.launcher "$@"\n' > /usr/local/bin/run-sovereign-runtime && \
    printf '#!/bin/sh\nexec python3 -m lazarus.appliance.healthcheck "$@"\n' > /usr/local/bin/sovereign-runtime-healthcheck && \
    chmod +x /usr/local/bin/run-sovereign-runtime /usr/local/bin/sovereign-runtime-healthcheck

ENV SOVEREIGN_PROFILE=metal-arm64 \
    SOVEREIGN_ENGINE_BACKEND=agent \
    VLLM_BACKEND=agent

EXPOSE 8000
ENTRYPOINT ["run-sovereign-runtime"]
