"""
Local mock training script.

Runs a simplified training loop using MockDataLoader and GLOO distributed
backend. Does NOT require GPUs or real models.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from prime_rl.mock.setup_env import setup_mock_environment
from prime_rl.mock.distributed_cpu import setup_torch_distributed_cpu
from prime_rl.mock.data_mock import MockDataLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "simulation_config.json"


@dataclass
class TrainingConfig:
    """Configuration for local mock training."""

    max_steps: int = 5
    batch_size: int = 2
    seq_len: int = 256
    learning_rate: float = 1e-5
    output_dir: Path = Path("outputs/local_training")
    log_interval: int = 1
    world_size: int = 4
    use_distributed: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        training = data.get("training", {})
        return cls(
            max_steps=training.get("max_steps", cls.max_steps),
            batch_size=training.get("batch_size", cls.batch_size),
            seq_len=training.get("seq_len", cls.seq_len),
            learning_rate=training.get("learning_rate", cls.learning_rate),
            output_dir=Path(training.get("output_dir", str(cls.output_dir))),
            log_interval=training.get("log_interval", cls.log_interval),
            world_size=data.get("cluster", {}).get("num_nodes", 2)
            * data.get("cluster", {}).get("gpus_per_node", 2),
            seed=data.get("seed", cls.seed),
        )


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
    """Local trainer that simulates the training loop from trainer/rl/train.py on CPU."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: list[TrainingMetrics] = []
        self.current_step = 0

        self._setup_environment()
        self._setup_distributed()
        self._setup_data_loader()
        self._setup_model()

    def _setup_environment(self):
        rank = int(os.environ.get("RANK", "0"))
        world_size = self.config.world_size if self.config.use_distributed else 1

        setup_mock_environment(
            rank=rank,
            world_size=world_size,
            local_rank=rank % 2,
            local_world_size=min(2, world_size),
            sandbox_url="http://localhost:8000",
            inference_url="http://localhost:8080/v1"
        )

        self.rank = rank
        self.world_size = world_size
        self.is_master = rank == 0

    def _setup_distributed(self):
        if not self.config.use_distributed or self.world_size == 1:
            logger.info("Running in single-process mode (no distributed)")
            return

        try:
            setup_torch_distributed_cpu()
            logger.info("Distributed initialized: rank=%d/%d", self.rank, self.world_size)
        except Exception as e:
            logger.warning("Distributed initialization failed: %s", e)
            logger.warning("Falling back to single-process mode")
            self.config.use_distributed = False

    def _setup_data_loader(self):
        self.data_loader = MockDataLoader(
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
        )
        if self.is_master:
            self.data_loader.log_batch_structure()

    def _setup_model(self):
        torch.manual_seed(self.config.seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.config.seq_len, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("Mock model initialized: %s parameters", f"{param_count:,}")

    def train_step(self) -> TrainingMetrics:
        step_start = time.perf_counter()

        data_start = time.perf_counter()
        self.data_loader.wait_for_batch()
        micro_batches = self.data_loader.get_batch()
        data_time = time.perf_counter() - data_start

        forward_start = time.perf_counter()
        total_loss = 0.0
        for micro_batch in micro_batches:
            x = micro_batch["advantages"].float()
            output = self.model(x)
            loss = output.mean()
            total_loss += loss.item()
        forward_time = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        backward_time = time.perf_counter() - backward_start

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
        logger.info(
            "Starting training: max_steps=%d, batch_size=%d, seq_len=%d, "
            "world_size=%d, rank=%d",
            self.config.max_steps, self.config.batch_size,
            self.config.seq_len, self.world_size, self.rank,
        )

        for step in range(self.config.max_steps):
            metrics = self.train_step()

            if step % self.config.log_interval == 0 and self.is_master:
                logger.info(
                    "Step %4d | Loss: %.4f | Time: %.3fs | "
                    "Data: %.3fs | Fwd: %.3fs | Bwd: %.3fs",
                    metrics.step, metrics.loss, metrics.step_time,
                    metrics.data_load_time, metrics.forward_time,
                    metrics.backward_time,
                )

        if self.is_master:
            self._save_results()

        logger.info("Training complete")

    def _save_results(self):
        results = {
            "metadata": {
                "seed": self.config.seed,
                "timestamp": datetime.now().isoformat(),
            },
            "config": asdict(self.config),
            "metrics": [asdict(m) for m in self.metrics],
        }
        results["config"]["output_dir"] = str(self.config.output_dir)

        output_file = self.config.output_dir / "training_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Results saved to %s", output_file)


def load_config(config_path: Optional[Path] = None) -> TrainingConfig:
    path = config_path or CONFIG_PATH
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return TrainingConfig.from_dict(data)
    return TrainingConfig()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()
    config.world_size = 1
    config.use_distributed = False

    trainer = LocalMockTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
