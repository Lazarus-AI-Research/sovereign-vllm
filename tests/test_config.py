import pytest
import yaml

from lazarus.appliance.config import ConfigError, load_config


def test_valid_config_parses(config_file):
    config = load_config(config_file)
    assert config.roles.generation.served_model_name == "assistant-dev"
    assert config.alias_to_role() == {
        "assistant-dev": "generation",
        "embedding-custom": "embedding",
    }
    assert config.roles.embedding.throttle_when_generation_queue_above == 2


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/runtime.yaml")


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("roles: [broken")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_missing_roles_raises(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text('schema_version: "1.2"\nruntime:\n  port: 8000\n')
    with pytest.raises(ConfigError, match="roles"):
        load_config(path)


def test_enabled_role_requires_model(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    del data["roles"]["generation"]["model"]
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="roles.generation.model"):
        load_config(path)


def test_unknown_key_rejected(tmp_path, config_file):
    path = tmp_path / "broken.yaml"
    path.write_text(config_file.read_text() + "\nunknown_section: {}\n")
    with pytest.raises(ConfigError, match="unknown_section"):
        load_config(path)


def test_disabled_role_needs_no_model(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    data["roles"]["embedding"] = {"enabled": False, "task": "embed"}
    path = tmp_path / "single-role.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    assert "embedding" not in config.enabled_roles()


def test_embedding_role_can_be_omitted(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    del data["roles"]["embedding"]
    path = tmp_path / "generation-only.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    assert config.roles.embedding is None
    assert list(config.enabled_roles()) == ["generation"]


def test_tool_parser_field(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    data["roles"]["generation"]["tool_call_parser"] = "off"
    path = tmp_path / "tools-off.yaml"
    path.write_text(yaml.safe_dump(data))
    assert load_config(path).roles.generation.tool_call_parser == "off"


def test_generation_placement_requires_exact_unique_uuid_cardinality(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    data["runtime"]["profile"] = "cuda-x86_64"
    role = data["roles"]["generation"]
    first = "GPU-00000000-0000-0000-0000-000000000001"
    second = "GPU-00000000-0000-0000-0000-000000000002"
    role["accelerator_device_ids"] = [first, second]
    role["tensor_parallel_size"] = 2
    path = tmp_path / "multi-gpu.yaml"
    path.write_text(yaml.safe_dump(data))
    parsed = load_config(path)
    assert parsed.roles.generation.accelerator_device_ids == [first, second]
    assert parsed.roles.generation.tensor_parallel_size == 2

    for devices, size in (([first], 2), ([first, first], 2), (["0", second], 2)):
        role["accelerator_device_ids"] = devices
        role["tensor_parallel_size"] = size
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ConfigError, match="roles.generation"):
            load_config(path)


def test_embedding_cannot_claim_generation_placement(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    data["roles"]["embedding"]["tensor_parallel_size"] = 2
    path = tmp_path / "embedding-placement.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="roles.embedding"):
        load_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine_args", ["--tensor-parallel-size", "4"]),
        ("env", {"CUDA_VISIBLE_DEVICES": "0,1"}),
        ("environment", {"CUDA_VISIBLE_DEVICES": "0,1"}),
    ],
)
def test_raw_engine_controls_are_rejected(tmp_path, config_file, field, value):
    data = yaml.safe_load(config_file.read_text())
    data["roles"]["generation"][field] = value
    path = tmp_path / "raw-engine-controls.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=field):
        load_config(path)


@pytest.mark.parametrize("profile", ["cuda-x86_64", "cuda-arm64-dgx-spark"])
@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("number_type", [int, float])
def test_managed_cuda_config_preserves_rank_order(
    managed_cuda_config_file, profile, count, number_type
):
    data = yaml.safe_load(managed_cuda_config_file.read_text())
    data["runtime"]["profile"] = profile
    devices = [f"GPU-00000000-0000-0000-0000-{rank:012x}" for rank in range(count, 0, -1)]
    data["roles"]["generation"]["accelerator_device_ids"] = devices
    data["roles"]["generation"]["tensor_parallel_size"] = number_type(count)
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    role = load_config(managed_cuda_config_file).roles.generation
    assert role.accelerator_device_ids == devices
    assert role.tensor_parallel_size == count
    assert role.enforce_eager is True


@pytest.mark.parametrize(
    "placement",
    [
        {"accelerator_device_ids": []},
        {"accelerator_device_ids": [], "tensor_parallel_size": 1},
        {"accelerator_device_ids": ["GPU-00000000-0000-0000-0000-000000000001"]},
        {"tensor_parallel_size": 1},
        {"tensor_parallel_size": 2},
        {"tensor_parallel_size": 4},
        {"accelerator_device_ids": ["GPU-not-a-uuid"], "tensor_parallel_size": 1},
        {
            "accelerator_device_ids": ["GPU-AAAAAAAA-bbbb-cccc-dddd-eeeeeeeeeeee"],
            "tensor_parallel_size": 1,
        },
        {
            "accelerator_device_ids": ["gpu-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
            "tensor_parallel_size": 1,
        },
        {
            "accelerator_device_ids": ["GPU-00000000-0000-0000-0000-000000000001"],
            "tensor_parallel_size": True,
        },
        {
            "accelerator_device_ids": ["GPU-00000000-0000-0000-0000-000000000001"],
            "tensor_parallel_size": "1",
        },
        {
            "accelerator_device_ids": ["GPU-00000000-0000-0000-0000-000000000001"],
            "tensor_parallel_size": 1.5,
        },
        {
            "accelerator_device_ids": [
                f"GPU-00000000-0000-0000-0000-{rank:012x}" for rank in range(3)
            ],
            "tensor_parallel_size": 3,
        },
    ],
)
def test_managed_placement_rejects_noncanonical_shapes(managed_cuda_config_file, placement):
    data = yaml.safe_load(managed_cuda_config_file.read_text())
    role = data["roles"]["generation"]
    del role["accelerator_device_ids"]
    del role["tensor_parallel_size"]
    role.update(placement)
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="roles.generation"):
        load_config(managed_cuda_config_file)


@pytest.mark.parametrize("profile", ["metal-arm64", "cpu-arm64", "cpu-x86_64", "mock"])
def test_non_cuda_profiles_reject_managed_placement(managed_cuda_config_file, profile):
    data = yaml.safe_load(managed_cuda_config_file.read_text())
    data["runtime"]["profile"] = profile
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="CUDA runtime profile"):
        load_config(managed_cuda_config_file)


@pytest.mark.parametrize(
    "placement",
    [
        {"accelerator_device_ids": []},
        {"tensor_parallel_size": 1},
        {
            "accelerator_device_ids": ["GPU-00000000-0000-0000-0000-000000000001"],
            "tensor_parallel_size": 1,
        },
    ],
)
def test_disabled_embedding_rejects_placement_presence(managed_cuda_config_file, placement):
    data = yaml.safe_load(managed_cuda_config_file.read_text())
    data["roles"]["embedding"].update(placement)
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="roles.embedding"):
        load_config(managed_cuda_config_file)


@pytest.mark.parametrize("eager", ["true", "false", 0, 1, None])
def test_enforce_eager_requires_a_boolean(managed_cuda_config_file, eager):
    data = yaml.safe_load(managed_cuda_config_file.read_text())
    data["roles"]["generation"]["enforce_eager"] = eager
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="enforce_eager"):
        load_config(managed_cuda_config_file)


@pytest.mark.parametrize("profile", ["cuda-x86_64", "metal-arm64"])
def test_legacy_config_omits_placement_fields(config_file, profile):
    data = yaml.safe_load(config_file.read_text())
    data["runtime"]["profile"] = profile
    config_file.write_text(yaml.safe_dump(data))
    role = load_config(config_file).roles.generation
    assert role.accelerator_device_ids == []
    assert role.tensor_parallel_size == 1
