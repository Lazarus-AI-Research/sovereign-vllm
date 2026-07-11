import textwrap
from pathlib import Path

import pytest

VALID_CONFIG = textwrap.dedent(
    """
    schema_version: "1.1"

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
        model: Qwen/Qwen3-Embedding-0.6B
        served_model_name: embedding-dev
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
