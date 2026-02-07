"""
Mock Data Loader for Security Analysis.

This generates realistic training data structures that match
what the real DataLoader produces, without needing:
- Real rollouts from orchestrator
- GPU tensors
- Actual model inference

Use this to analyze:
- Data format and structure
- Serialization patterns
- What information flows between components
"""
import torch
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockDataLoader")


class TensorMicroBatch(TypedDict):
    """
    Matches the real micro batch structure from src/prime_rl/trainer/rl/data.py
    """
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
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float]


class MockDataLoader:
    """
    Mock data loader that generates realistic training batches.
    
    The real DataLoader:
    1. Receives TrainingBatch from orchestrator via ZMQ/FileSystem
    2. Packs samples into micro batches
    3. Returns TensorMicroBatch for training
    
    This mock:
    1. Generates synthetic rollouts
    2. Converts to same tensor format
    3. Allows analysis of data structures
    """
    
    def __init__(
        self,
        seq_len: int = 512,
        batch_size: int = 2,
        vocab_size: int = 151936,  # Qwen vocab size
        max_runs: int = 1,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_runs = max_runs
        self.current_step = 0
        
        # Sample prompts for generating mock data
        self.sample_prompts = [
            "What is 2 + 2?",
            "Write a Python function to calculate factorial.",
            "Explain the theory of relativity.",
            "What is the capital of France?",
            "Solve: 3x + 5 = 20",
        ]
        
        logger.info(f"MockDataLoader initialized: seq_len={seq_len}, "
                   f"batch_size={batch_size}, vocab_size={vocab_size}")
    
    def wait_for_batch(self) -> None:
        """
        In real DataLoader, this waits for data from orchestrator.
        Here we just log and return immediately.
        """
        logger.debug(f"[Step {self.current_step}] wait_for_batch() called")
        # Simulate small delay
        import time
        time.sleep(0.01)
    
    def get_batch(self) -> List[TensorMicroBatch]:
        """
        Generate a batch of mock training data.
        
        Returns the same structure as real DataLoader.
        """
        logger.debug(f"[Step {self.current_step}] get_batch() called")
        
        micro_batches = []
        
        for i in range(self.batch_size):
            micro_batch = self._generate_micro_batch(seed=self.current_step * 100 + i)
            micro_batches.append(micro_batch)
        
        self.current_step += 1
        
        logger.info(f"Generated batch with {len(micro_batches)} micro batches, "
                   f"seq_len={self.seq_len}")
        
        return micro_batches
    
    def _generate_micro_batch(self, seed: int = 0) -> TensorMicroBatch:
        """Generate a single micro batch with realistic structure."""
        
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Generate input IDs (random tokens)
        input_ids = torch.randint(0, self.vocab_size, (1, self.seq_len))
        
        # Position IDs (0 to seq_len-1)
        position_ids = torch.arange(self.seq_len).unsqueeze(0)
        
        # Advantages (normalized rewards, typically -1 to 1)
        advantages = torch.randn(1, self.seq_len) * 0.5
        
        # Inference logprobs (typically negative, around -2 to -5)
        inference_logprobs = torch.randn(1, self.seq_len) * 0.5 - 3.0
        
        # Loss mask (True for response tokens, False for prompt)
        # Assume first 30% is prompt
        prompt_len = int(self.seq_len * 0.3)
        loss_mask = torch.zeros(1, self.seq_len, dtype=torch.bool)
        loss_mask[0, prompt_len:] = True
        
        # Temperatures (usually 1.0, sometimes varied)
        temperatures = torch.ones(1, self.seq_len)
        
        # LoRA num tokens (for multi-LoRA training)
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
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get information about batch structure for security analysis."""
        sample_batch = self._generate_micro_batch()
        
        info = {}
        for key, value in sample_batch.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    info[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "device": str(value.device),
                        "size_bytes": value.element_size() * value.numel(),
                    }
                else:
                    info[key] = {"type": type(value).__name__}
            else:
                info[key] = None
        
        return info
    
    def print_batch_structure(self):
        """Print the batch structure for documentation."""
        info = self.get_batch_info()
        
        print("\n" + "="*60)
        print("TRAINING BATCH STRUCTURE")
        print("="*60)
        
        total_bytes = 0
        for key, value in info.items():
            if value is not None and isinstance(value, dict) and "shape" in value:
                size = value.get("size_bytes", 0)
                total_bytes += size
                print(f"  {key}:")
                print(f"    shape: {value['shape']}")
                print(f"    dtype: {value['dtype']}")
                print(f"    size:  {size:,} bytes")
            elif value is None:
                print(f"  {key}: None")
        
        print(f"\n  Total size per micro batch: {total_bytes:,} bytes")
        print(f"  Total size per batch: {total_bytes * self.batch_size:,} bytes")
        print("="*60)