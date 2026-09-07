import textwrap
from pathlib import Path

import pytest

VALID_CONFIG = textwrap.dedent(
    """
    schema_version: "1.2"

    runtime:
      listen_address: 0.0.0.0
      port: 8000
      api_key_env: SOVEREIGN_RUNTIME_API_KEY
      profile: cpu-arm64

    startup:
      smoke_test_on_start: true
      remain_alive_on_configuration_error: true

    roles:
      generation:
        enabled: true
        task: generate
        source: huggingface
        model: Qwen/Qwen3-0.6B
        served_model_name: assistant-dev
        max_model_len: 8192
        priority: high
        memory_weight: 82
        max_concurrent_requests: 4

      embedding:
        enabled: true
        task: embed
        source: huggingface
        model: intfloat/e5-small-v2
        served_model_name: embedding-custom
        priority: low
        memory_weight: 18
        max_concurrent_requests: 2
        throttle_when_generation_queue_above: 2
        pooling: last
        normalization: l2
    """
)


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "runtime.yaml"
    path.write_text(VALID_CONFIG)
    return path


@pytest.fixture()
def managed_cuda_config_file(tmp_path: Path) -> Path:
    # Stack's curated CUDA generation profile after boundedProfile and
    # ApplyCandidate render the staged artifact and ordered placement.
    path = tmp_path / "runtime.yaml"
    path.write_text(
        textwrap.dedent("""\
        schema_version: "1.2"
        runtime:
          listen_address: 0.0.0.0
          port: 8000
          api_key_env: SOVEREIGN_RUNTIME_API_KEY
          profile: cuda-x86_64
        startup:
          smoke_test_on_start: true
          remain_alive_on_configuration_error: true
          fail_process_on_generation_error: false
          fail_process_on_embedding_error: false
        roles:
          generation:
            enabled: true
            task: generate
            source: local
            model: /models/staged/gemma-4-E2B-it
            revision: 9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf
            served_model_name: assistant-large
            max_model_len: 2048
            priority: high
            memory_weight: 52
            max_concurrent_requests: 1
            enforce_eager: true
            tool_call_parser: gemma4_native
            reasoning_parser: "off"
            accelerator_device_ids:
              - GPU-00000000-0000-0000-0000-000000000002
              - GPU-00000000-0000-0000-0000-000000000001
            tensor_parallel_size: 2
          embedding:
            enabled: false
            task: embed
        observability:
          prometheus: true
          structured_logs: true
          otlp_endpoint: http://otel-collector:4317
        privacy:
          prompt_logging: false
          response_logging: false
          full_trace: false
        """)
    )
    return path
