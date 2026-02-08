"""
Mock data loader for security analysis.

Generates training data structures matching the real DataLoader output
without requiring rollouts, GPU tensors, or model inference.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional, TypedDict

import torch

logger = logging.getLogger(__name__)


class TensorMicroBatch(TypedDict):
    """Matches the real micro batch structure from trainer/rl/data.py."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    advantages: torch.Tensor
    inference_logprobs: torch.Tensor
    teacher_logprobs: Optional[torch.Tensor]
    loss_mask: torch.Tensor
    temperatures: torch.Tensor
    lora_num_tokens: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]


@dataclass
class MockRollout:
    """Simulated rollout data."""

    prompt: str
    response: str
    reward: float
    prompt_ids: list[int]
    response_ids: list[int]
    logprobs: list[float]


class MockDataLoader:
    """Mock data loader that generates realistic training batches."""

    def __init__(
        self,
        seq_len: int = 512,
        batch_size: int = 2,
        vocab_size: int = 151936,
        max_runs: int = 1,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_runs = max_runs
        self.seed = seed
        self.current_step = 0

        self.sample_prompts = [
            "What is 2 + 2?",
            "Write a Python function to calculate factorial.",
            "Explain the theory of relativity.",
            "What is the capital of France?",
            "Solve: 3x + 5 = 20",
        ]

        logger.info(
            "MockDataLoader initialized: seq_len=%d, batch_size=%d, vocab_size=%d, seed=%d",
            seq_len, batch_size, vocab_size, seed,
        )

    def wait_for_batch(self) -> None:
        """Simulate waiting for data from orchestrator."""
        logger.debug("Step %d: wait_for_batch()", self.current_step)
        time.sleep(0.01)

    def get_batch(self) -> list[TensorMicroBatch]:
        """Generate a batch of mock training data matching the real DataLoader output."""
        logger.debug("Step %d: get_batch()", self.current_step)

        micro_batches = []
        for i in range(self.batch_size):
            batch_seed = self.seed + self.current_step * 100 + i
            micro_batch = self._generate_micro_batch(seed=batch_seed)
            micro_batches.append(micro_batch)

        self.current_step += 1

        logger.info(
            "Generated batch: %d micro batches, seq_len=%d",
            len(micro_batches), self.seq_len,
        )
        return micro_batches

    def _generate_micro_batch(self, seed: int = 0) -> TensorMicroBatch:
        """Generate a single micro batch with realistic structure."""
        torch.manual_seed(seed)
        random.seed(seed)

        input_ids = torch.randint(0, self.vocab_size, (1, self.seq_len))
        position_ids = torch.arange(self.seq_len).unsqueeze(0)
        advantages = torch.randn(1, self.seq_len) * 0.5
        inference_logprobs = torch.randn(1, self.seq_len) * 0.5 - 3.0

        # First 30% is prompt (masked out), rest is response
        prompt_len = int(self.seq_len * 0.3)
        loss_mask = torch.zeros(1, self.seq_len, dtype=torch.bool)
        loss_mask[0, prompt_len:] = True

        temperatures = torch.ones(1, self.seq_len)

        lora_num_tokens = torch.zeros(self.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = self.seq_len

        return TensorMicroBatch(
            input_ids=input_ids,
            position_ids=position_ids,
            advantages=advantages,
            inference_logprobs=inference_logprobs,
            teacher_logprobs=None,
            loss_mask=loss_mask,
            temperatures=temperatures,
            lora_num_tokens=lora_num_tokens,
            pixel_values=None,
            image_grid_thw=None,
        )

    def get_batch_info(self) -> dict[str, object]:
        """Get information about batch structure for security analysis."""
        sample_batch = self._generate_micro_batch(seed=self.seed)

        info = {}
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                info[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "size_bytes": value.element_size() * value.numel(),
                }
            elif value is None:
                info[key] = None
            else:
                info[key] = {"type": type(value).__name__}

        return info

    def log_batch_structure(self) -> None:
        """Log the batch structure for analysis."""
        info = self.get_batch_info()

        total_bytes = 0
        lines = ["Training batch structure:"]
        for key, value in info.items():
            if isinstance(value, dict) and "shape" in value:
                size = value.get("size_bytes", 0)
                total_bytes += size
                lines.append(
                    f"  {key}: shape={value['shape']}, dtype={value['dtype']}, size={size:,}B"
                )
            elif value is None:
                lines.append(f"  {key}: None")

        lines.append(f"  Total per micro batch: {total_bytes:,}B")
        lines.append(f"  Total per batch: {total_bytes * self.batch_size:,}B")
        logger.info("\n".join(lines))
