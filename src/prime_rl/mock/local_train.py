"""
Local Mock Training Script.

This runs a simplified training loop that:
1. Uses MockDataLoader for data
2. Uses MockVerifier for scoring
3. Uses GLOO for distributed communication
4. Logs all operations for security analysis

This does NOT require GPUs or real models.
"""
import os
import sys
import time
import json
import logging
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from prime_rl.mock.setup_env import setup_simulated_cluster
from prime_rl.mock.distributed_cpu import setup_torch_distributed_cpu
from prime_rl.mock.data_mock import MockDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("LocalTrainer")


@dataclass
class TrainingConfig:
    """Configuration for local mock training."""
    max_steps: int = 5
    batch_size: int = 2
    seq_len: int = 256
    learning_rate: float = 1e-5
    output_dir: Path = Path("outputs/local_training")
    log_interval: int = 1
    
    # Distributed settings
    world_size: int = 4
    use_distributed: bool = True


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    step: int
    loss: float
    step_time: float
    data_load_time: float
    forward_time: float
    backward_time: float


class LocalMockTrainer:
    """
    Local trainer for security analysis.
    
    This simulates the training loop from src/prime_rl/trainer/rl/train.py
    but runs entirely on CPU with mock data.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[TrainingMetrics] = []
        self.current_step = 0
        
        # Initialize components
        self._setup_environment()
        self._setup_distributed()
        self._setup_data_loader()
        self._setup_model()
        
    def _setup_environment(self):
        """Setup environment variables for distributed training."""
        # Default to rank 0 for single-process testing
        rank = int(os.environ.get("RANK", "0"))
        world_size = self.config.world_size if self.config.use_distributed else 1
        
        setup_simulated_cluster(
            rank=rank,
            world_size=world_size,
            local_rank=rank % 2,
            local_world_size=min(2, world_size),
        )
        
        self.rank = rank
        self.world_size = world_size
        self.is_master = rank == 0
        
    def _setup_distributed(self):
        """Initialize distributed training with GLOO."""
        if not self.config.use_distributed or self.world_size == 1:
            logger.info("Running in single-process mode (no distributed)")
            return
            
        try:
            setup_torch_distributed_cpu()
            logger.info(f"Distributed initialized: rank={self.rank}/{self.world_size}")
        except Exception as e:
            logger.warning(f"Distributed initialization failed: {e}")
            logger.warning("This is expected if running a single process with world_size > 1.")
            logger.warning("Falling back to single-process mode (use_distributed=False).")
            self.config.use_distributed = False
    
    def _setup_data_loader(self):
        """Setup mock data loader."""
        self.data_loader = MockDataLoader(
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
        )
        
        # Print batch structure for documentation
        if self.is_master:
            self.data_loader.print_batch_structure()
    
    def _setup_model(self):
        """Setup a minimal mock model for testing."""
        # Simple linear model for testing the training loop
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.config.seq_len, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Mock model initialized: {param_count:,} parameters")
    
    def train_step(self) -> TrainingMetrics:
        """Execute one training step."""
        step_start = time.perf_counter()
        
        # 1. Load data
        data_start = time.perf_counter()
        self.data_loader.wait_for_batch()
        micro_batches = self.data_loader.get_batch()
        data_time = time.perf_counter() - data_start
        
        # 2. Forward pass
        forward_start = time.perf_counter()
        total_loss = 0.0
        
        for micro_batch in micro_batches:
            # Use advantages as a simple training signal
            x = micro_batch["advantages"].float()
            
            # Forward
            output = self.model(x)
            loss = output.mean()  # Simplified loss
            total_loss += loss.item()
        
        forward_time = time.perf_counter() - forward_start
        
        # 3. Backward pass
        backward_start = time.perf_counter()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (like real trainer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        backward_time = time.perf_counter() - backward_start
        
        # 4. Collect metrics
        step_time = time.perf_counter() - step_start
        avg_loss = total_loss / len(micro_batches)
        
        metrics = TrainingMetrics(
            step=self.current_step,
            loss=avg_loss,
            step_time=step_time,
            data_load_time=data_time,
            forward_time=forward_time,
            backward_time=backward_time,
        )
        
        self.metrics.append(metrics)
        self.current_step += 1
        
        return metrics
    
    def train(self):
        """Run the full training loop."""
        logger.info("="*60)
        logger.info("STARTING LOCAL MOCK TRAINING")
        logger.info("="*60)
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Sequence length: {self.config.seq_len}")
        logger.info(f"World size: {self.world_size}")
        logger.info(f"Rank: {self.rank}")
        logger.info("="*60)
        
        for step in range(self.config.max_steps):
            metrics = self.train_step()
            
            if step % self.config.log_interval == 0 and self.is_master:
                logger.info(
                    f"Step {metrics.step:4d} | "
                    f"Loss: {metrics.loss:.4f} | "
                    f"Time: {metrics.step_time:.3f}s | "
                    f"Data: {metrics.data_load_time:.3f}s | "
                    f"Fwd: {metrics.forward_time:.3f}s | "
                    f"Bwd: {metrics.backward_time:.3f}s"
                )
        
        # Save results
        if self.is_master:
            self._save_results()
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
    
    def _save_results(self):
        """Save training results for analysis."""
        results = {
            "config": asdict(self.config),
            "metrics": [asdict(m) for m in self.metrics],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Convert Path to string for JSON
        results["config"]["output_dir"] = str(self.config.output_dir)
        
        output_file = self.config.output_dir / "training_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Main entry point for local training."""
    config = TrainingConfig(
        max_steps=5,
        batch_size=2,
        seq_len=256,
        world_size=1,  # Single process for simple testing
        use_distributed=False,
    )
    
    trainer = LocalMockTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()