"""CPU-compatible distributed setup using GLOO backend."""

import logging
from datetime import timedelta

import torch.distributed as dist

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = timedelta(seconds=600)


def setup_torch_distributed_cpu(
    timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
    """Initialize torch.distributed with GLOO backend for CPU."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available on this system")

    dist.init_process_group(backend="gloo", timeout=timeout)

    logger.info(
        "Distributed initialized: backend=gloo, rank=%d, world_size=%d",
        dist.get_rank(),
        dist.get_world_size(),
    )


def is_distributed_initialized() -> bool:
    return dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1
