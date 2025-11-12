"""Config utilities: load YAML configs and environment variables safely."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary.
    Args:
        path: Path to a YAML file.
    Returns:
        Parsed configuration dictionary.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def env(key: str, default: Any | None = None) -> Any:
    """Fetch environment variable with a default.
    Args:
        key: Environment variable name.
        default: Fallback value if not set.
    """
    return os.getenv(key, default)

