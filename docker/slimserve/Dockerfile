# syntax=docker/dockerfile:1
# Build context: Runtime repository root. No published SlimServe image is assumed.
# CUDA_BUILD_IMAGE MUST be an immutable, toolchain-equipped derivative of the
# existing docker/cuda/Dockerfile image (same stock vLLM + Runtime environment).
# slimserve-dependencies is a BuildKit named context containing requirements.lock
# and its complete SHA256-locked wheelhouse; see README.md.
ARG CUDA_BUILD_IMAGE
FROM ${CUDA_BUILD_IMAGE}
ARG CUDA_BUILD_IMAGE
ARG TORCH_CUDA_ARCH_LIST
ARG MAX_JOBS=4

RUN python3 -c "import importlib.metadata as m; assert m.version('vllm') == '0.25.0'; m.distribution('sovereign-runtime')" && \
    test -x /usr/local/bin/run-sovereign-runtime && \
    test -x /usr/local/bin/sovereign-runtime-healthcheck && \
    command -v nvcc && command -v git && command -v c++

COPY pyproject.toml constraints.txt README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/sovereign/
COPY lazarus /opt/sovereign/lazarus
COPY docker/slimserve /opt/sovereign-build/
RUN --mount=type=bind,from=slimserve-dependencies,target=/opt/slimserve-dependencies,ro \
    python3 /opt/sovereign-build/build.py --target cuda \
      --runtime-source /opt/sovereign --dependencies /opt/slimserve-dependencies \
      --base-image "${CUDA_BUILD_IMAGE}" --cuda-arch-list "${TORCH_CUDA_ARCH_LIST}" \
      --max-jobs "${MAX_JOBS}" && \
    python3 -m pip install --no-cache-dir --no-deps /opt/sovereign-slimserve/wheels/sovereign_runtime-*.whl

# Do NOT add the isolated bin directory to image-wide PATH: the normal Runtime
# launcher and stock vLLM interpreter remain the default serving environment.
ENV SOVEREIGN_PROFILE=cuda-x86_64 \
    VLLM_BACKEND=cuda
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/run-sovereign-runtime"]
