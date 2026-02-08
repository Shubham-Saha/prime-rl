"""Device abstraction for CPU-only execution."""

import logging

import torch

logger = logging.getLogger(__name__)

_DEVICE: str = "cpu"


def set_device(device: str) -> None:
    global _DEVICE
    _DEVICE = device


def get_device() -> str:
    return _DEVICE


def get_torch_device() -> torch.device:
    return torch.device(_DEVICE)


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(_DEVICE)


class MockCuda:
    """Mock torch.cuda for CPU-only systems."""

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0

    @staticmethod
    def set_device(device: int) -> None:
        pass

    @staticmethod
    def reset_peak_memory_stats() -> None:
        pass

    @staticmethod
    def max_memory_reserved() -> int:
        return 0


def patch_cuda_for_cpu():
    """Patch torch.cuda to work on CPU-only systems."""
    torch.cuda.is_available = MockCuda.is_available
    torch.cuda.device_count = MockCuda.device_count
    torch.cuda.set_device = MockCuda.set_device
    torch.cuda.reset_peak_memory_stats = MockCuda.reset_peak_memory_stats
    torch.cuda.max_memory_reserved = MockCuda.max_memory_reserved
    logger.info("torch.cuda patched for CPU-only execution")
