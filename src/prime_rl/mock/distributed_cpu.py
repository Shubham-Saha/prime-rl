"""
CPU-compatible distributed setup using GLOO backend.

This replaces the NCCL-based setup in src/prime_rl/trainer/utils.py
for systems without NVIDIA GPUs.

The ONLY changes from original:
1. Remove torch.cuda.set_device() 
2. Force backend="gloo"
"""
import logging
import torch
import torch.distributed as dist
from datetime import timedelta

logger = logging.getLogger(__name__)

# Original default from prime_rl/trainer/utils.py
DEFAULT_TIMEOUT = timedelta(seconds=600)


def setup_torch_distributed_cpu(
    timeout: timedelta = DEFAULT_TIMEOUT,
    enable_gloo: bool = True  # Always True for CPU
) -> None:
    """
    Initialize torch.distributed with GLOO backend for CPU.
    
    This is the CPU-compatible version of setup_torch_distributed from
    src/prime_rl/trainer/utils.py
    
    Original code (line 47-56):
        def setup_torch_distributed(timeout, enable_gloo):
            torch.cuda.set_device(get_world().local_rank)  # <-- REMOVED for CPU
            backend = None  # by default nccl
            if enable_gloo:
                backend = "cpu:gloo,cuda:nccl"
            dist.init_process_group(backend=backend, timeout=timeout)
    
    Our CPU version:
        - Skips torch.cuda.set_device()
        - Uses backend="gloo" (pure CPU)
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available on this system")

    # Use GLOO backend - works on CPU without CUDA
    backend = "gloo"
    
    dist.init_process_group(backend=backend, timeout=timeout)
    
    logger.info(f"Distributed initialized with backend={backend}, "
                f"rank={dist.get_rank()}, world_size={dist.get_world_size()}")


def is_distributed_initialized() -> bool:
    """Check if distributed is initialized."""
    return dist.is_initialized()


def get_rank() -> int:
    """Get current rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get world size."""
    return dist.get_world_size() if dist.is_initialized() else 1