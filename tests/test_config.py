"""Unit tests for configuration management."""

from pitch_predictor.config import get_project_root, load_config


def test_project_root_exists():
    root = get_project_root()
    assert root.exists()
    assert (root / "config" / "config.yaml").exists()


def test_load_config_values():
    config = load_config()
    assert "databricks" in config
    assert "automl" in config
    assert config["databricks"]["catalog"] == "main"
    assert config["automl"]["target_col"] == "pitch_type"
