"""
Config loader for bio-med-rag.

Merges configs/config.yaml (base) with configs/config.local.yaml (local
overrides, gitignored) and resolves env vars from .env.

Usage:
    from src.config.loader import get_config
    cfg = get_config()
    print(cfg.encoder.model_name)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

# Project root is two levels up from this file (src/config/loader.py)
ROOT = Path(__file__).resolve().parents[2]


def get_config() -> DictConfig:
    """Load and merge base + local YAML configs, then inject env vars."""

    # 1. Load .env into os.environ (no-op if .env does not exist)
    load_dotenv(ROOT / ".env", override=False)

    # 2. Load base config (always present)
    base_cfg_path = ROOT / "configs" / "config.yaml"
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
    cfg = OmegaConf.load(base_cfg_path)

    # 3. Merge local override if it exists
    local_cfg_path = ROOT / "configs" / "config.local.yaml"
    if local_cfg_path.exists():
        local_cfg = OmegaConf.load(local_cfg_path)
        cfg = OmegaConf.merge(cfg, local_cfg)

    # 4. Override logging level from env if set
    log_level = os.getenv("LOG_LEVEL")
    if log_level:
        cfg.logging.level = log_level

    return cfg
