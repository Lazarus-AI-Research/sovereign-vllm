# syntax=docker/dockerfile:1
# Sovereign Runtime image: cuda-x86_64 profile.
# Build context: repository root — docker build -f docker/cuda/Dockerfile .
#
# Bases on the upstream vLLM OpenAI image (pinned to match constraints.txt)
# and layers the Sovereign appliance on top. The appliance launcher is the
# entrypoint; it owns port 8000 and the §24 image contract.

FROM vllm/vllm-openai@sha256:fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb

COPY pyproject.toml constraints.txt README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/sovereign/
COPY lazarus /opt/sovereign/lazarus
RUN pip install --no-cache-dir --constraint /opt/sovereign/constraints.txt "/opt/sovereign[audio]"

# §24: required binaries (provided as console scripts by the package)
RUN test -x /usr/local/bin/run-sovereign-runtime && \
    test -x /usr/local/bin/sovereign-runtime-healthcheck

ENV SOVEREIGN_PROFILE=cuda-x86_64 \
    VLLM_BACKEND=cuda

EXPOSE 8000
ENTRYPOINT ["run-sovereign-runtime"]
