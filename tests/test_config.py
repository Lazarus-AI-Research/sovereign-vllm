import pytest
import yaml

from lazarus.appliance.config import ConfigError, load_config


def test_valid_config_parses(config_file):
    config = load_config(config_file)
    assert config.roles.generation.served_model_name == "assistant-dev"
    assert config.alias_to_role() == {"assistant-dev": "generation", "embedding-omni-default": "embedding"}
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
    path.write_text('schema_version: "1.1"\nruntime:\n  port: 8000\n')
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


def test_tool_parser_field(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    data["roles"]["generation"]["tool_call_parser"] = "off"
    path = tmp_path / "tools-off.yaml"
    path.write_text(yaml.safe_dump(data))
    assert load_config(path).roles.generation.tool_call_parser == "off"
