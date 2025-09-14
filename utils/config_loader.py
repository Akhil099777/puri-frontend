import os
from typing import Any, Dict
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Raises FileNotFoundError with a helpful message if missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. Please create it as per sample 'config/config.yaml'."
        )
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def get_models_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    models_cfg: Dict[str, Dict[str, Any]] = config.get("models", {})
    return models_cfg


def get_default_threshold(config: Dict[str, Any]) -> float:
    thresholds_cfg = config.get("thresholds", {})
    return float(thresholds_cfg.get("default", 0.5))


def ensure_uploads_dir(config: Dict[str, Any]) -> str:
    io_cfg = config.get("io", {})
    uploads_dir = io_cfg.get("uploads_dir", "uploads")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir, exist_ok=True)
    return uploads_dir
