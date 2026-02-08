"""Mock modules for PRIME-RL local simulation and security analysis."""

import json
from pathlib import Path
from typing import Optional

from prime_rl.mock.setup_env import setup_mock_environment , get_all_rank_configs, RankConfig
from prime_rl.mock.distributed_cpu import setup_torch_distributed_cpu
from prime_rl.mock.device import get_device, patch_cuda_for_cpu

CONFIG_PATH = Path(__file__).parent / "simulation_config.json"

__all__ = [
    "setup_mock_environment",
    "get_all_rank_configs",
    "RankConfig",
    "setup_torch_distributed_cpu",
    "get_device",
    "patch_cuda_for_cpu",
    "load_simulation_config",
]


def load_simulation_config(config_path: Optional[Path] = None) -> dict:
    """Load simulation configuration from JSON file."""
    path = config_path or CONFIG_PATH
    with open(path) as f:
        return json.load(f)
