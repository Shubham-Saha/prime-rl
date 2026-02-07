"""
Mock modules for PRIME-RL security analysis.

Required modules (Minimal):
- setup_env: Environment variable configuration
- distributed_cpu: GLOO-based distributed setup
- device: CPU device handling
- mock_inference_server: HTTP API server
- mock_verifier: Verification simulation
- run_security_analysis: Main entry point

Optional modules (Full Training):
- data_mock: Fake training data generation
- local_sandbox: Local code execution sandbox
- local_train: Mock training loop
"""

from prime_rl.mock.setup_env import setup_simulated_cluster, get_all_rank_configs
from prime_rl.mock.distributed_cpu import setup_torch_distributed_cpu
from prime_rl.mock.device import get_device, patch_cuda_for_cpu

__all__ = [
    "setup_simulated_cluster",
    "get_all_rank_configs", 
    "setup_torch_distributed_cpu",
    "get_device",
    "patch_cuda_for_cpu",
]