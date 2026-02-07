"""
Device abstraction for CPU-only execution.

In the real train.py, tensors are moved to "cuda" device.
This module provides a way to redirect to "cpu" instead.

Usage:
    from prime_rl.mock.device import get_device
    tensor = tensor.to(get_device())  # Returns "cpu" instead of "cuda"
"""
import torch
from typing import Union

# Global device setting
_DEVICE: str = "cpu"


def set_device(device: str) -> None:
    """Set the global device."""
    global _DEVICE
    _DEVICE = device


def get_device() -> str:
    """Get the current device string."""
    return _DEVICE


def get_torch_device() -> torch.device:
    """Get a torch.device object."""
    return torch.device(_DEVICE)


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to the configured device."""
    return tensor.to(_DEVICE)


# Mock cuda functions for CPU execution
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
        pass  # No-op
    
    @staticmethod
    def reset_peak_memory_stats() -> None:
        pass  # No-op
    
    @staticmethod
    def max_memory_reserved() -> int:
        return 0


def patch_cuda_for_cpu():
    """
    Patch torch.cuda to work on CPU-only systems.
    Call this before importing train.py if you want to avoid CUDA errors.
    """
    import torch
    torch.cuda.is_available = MockCuda.is_available
    torch.cuda.device_count = MockCuda.device_count
    torch.cuda.set_device = MockCuda.set_device
    torch.cuda.reset_peak_memory_stats = MockCuda.reset_peak_memory_stats
    torch.cuda.max_memory_reserved = MockCuda.max_memory_reserved
    print("[MOCK] torch.cuda patched for CPU-only execution")