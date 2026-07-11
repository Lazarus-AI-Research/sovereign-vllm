# syntax=docker/dockerfile:1
# Sovereign Runtime image: cpu profile (bring-up, dev, and the base for
# Metal Phase 1). linux/amd64 + linux/arm64.
# Build context: repository root — docker build -f docker/cpu/Dockerfile .
#
# The base is upstream vLLM's CPU image built from the pinned source tree
# (docker/Dockerfile.cpu in vllm-project/vllm at the constraints.txt version):
#   docker build -f docker/Dockerfile.cpu -t vllm-cpu:v0.25.0 <vllm checkout>

ARG VLLM_CPU_BASE=vllm-cpu:v0.25.0
FROM ${VLLM_CPU_BASE}

COPY pyproject.toml /opt/sovereign/
COPY lazarus /opt/sovereign/lazarus
RUN python3 -m pip install --no-cache-dir /opt/sovereign

# §24 image contract: canonical binaries, independent of console-script paths.
RUN printf '#!/bin/sh\nexec python3 -m lazarus.appliance.launcher "$@"\n' > /usr/local/bin/run-sovereign-runtime && \
    printf '#!/bin/sh\nexec python3 -m lazarus.appliance.healthcheck "$@"\n' > /usr/local/bin/sovereign-runtime-healthcheck && \
    chmod +x /usr/local/bin/run-sovereign-runtime /usr/local/bin/sovereign-runtime-healthcheck

ENV SOVEREIGN_PROFILE=cpu-arm64 \
    VLLM_BACKEND=cpu

EXPOSE 8000
ENTRYPOINT ["run-sovereign-runtime"]
