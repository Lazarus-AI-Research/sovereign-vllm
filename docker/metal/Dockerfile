# syntax=docker/dockerfile:1
# Sovereign Runtime image: metal-arm64 profile — Phase 1 (design.md §2.6).
#
# Phase 1 ships the identical appliance on the vLLM CPU backend for aarch64.
# It passes the full contract; the manifest reports CPU execution honestly
# (backend=cpu, vendor=apple is NOT claimed for compute). Phase 2 swaps the
# engine backend for the host inference agent (Metal via llama.cpp) behind
# the unchanged contract; Phase 3 is the vLLM-Metal fork backend.
#
# Build context: repository root — docker build -f docker/metal/Dockerfile .

ARG VLLM_CPU_BASE=vllm-cpu:v0.25.0
FROM ${VLLM_CPU_BASE}

COPY pyproject.toml /opt/sovereign/
COPY lazarus /opt/sovereign/lazarus
RUN python3 -m pip install --no-cache-dir /opt/sovereign

RUN printf '#!/bin/sh\nexec python3 -m lazarus.appliance.launcher "$@"\n' > /usr/local/bin/run-sovereign-runtime && \
    printf '#!/bin/sh\nexec python3 -m lazarus.appliance.healthcheck "$@"\n' > /usr/local/bin/sovereign-runtime-healthcheck && \
    chmod +x /usr/local/bin/run-sovereign-runtime /usr/local/bin/sovereign-runtime-healthcheck

ENV SOVEREIGN_PROFILE=metal-arm64 \
    VLLM_BACKEND=cpu

EXPOSE 8000
ENTRYPOINT ["run-sovereign-runtime"]
