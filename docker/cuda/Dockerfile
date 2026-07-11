# syntax=docker/dockerfile:1
# Sovereign Runtime image: cuda-x86_64 profile.
# Build context: repository root — docker build -f docker/cuda/Dockerfile .
#
# Bases on the upstream vLLM OpenAI image (pinned to match constraints.txt)
# and layers the Sovereign appliance on top. The appliance launcher is the
# entrypoint; it owns port 8000 and the §24 image contract.

FROM vllm/vllm-openai:v0.25.0

COPY pyproject.toml constraints.txt /opt/sovereign/
COPY lazarus /opt/sovereign/lazarus
RUN pip install --no-cache-dir /opt/sovereign

# §24: required binaries (provided as console scripts by the package)
RUN ln -sf $(which run-sovereign-runtime) /usr/local/bin/run-sovereign-runtime && \
    ln -sf $(which sovereign-runtime-healthcheck) /usr/local/bin/sovereign-runtime-healthcheck

ENV SOVEREIGN_PROFILE=cuda-x86_64 \
    VLLM_BACKEND=cuda

EXPOSE 8000
ENTRYPOINT ["run-sovereign-runtime"]
