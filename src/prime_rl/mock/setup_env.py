"""
Environment setup for local multi-process simulation.

Sets environment variables so prime_rl believes it is running on a distributed cluster.
Must be called BEFORE importing any prime_rl modules.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RankConfig:
    """Configuration for a single rank in the simulated cluster."""

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    node_id: int
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    def as_env_vars(self) -> dict[str, str]:
        return {
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }


def setup_mock_environment(
    rank: int = 0,
    world_size: int = 8,
    local_rank: int = 0,
    local_world_size: int = 2,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    sandbox_url: str = "http://localhost:8000",
    inference_url: str = "http://localhost:8080/v1",
) -> None:
    """Set environment variables to simulate a distributed cluster (see trainer/envs.py)."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PRIME_RL_MOCK_MODE"] = "1"

    # Primary Sandbox URL redirect
    os.environ["PRIME_SANDBOX_API_URL"] = sandbox_url
    os.environ["SANDBOX_API_URL"] = sandbox_url
    os.environ["SANDBOX_BASE_URL"] = sandbox_url
    
    # Fake API keys to pass library validation
    os.environ["PRIME_API_KEY"] = "mock-security-analysis-key"
    os.environ["SANDBOX_API_KEY"] = "mock-security-analysis-key"
    
    # Inference Server redirect (for LLM calls)
    os.environ["OPENAI_BASE_URL"] = inference_url
    os.environ["OPENAI_API_KEY"] = "mock-inference-key"

    logger.info("=" * 60)
    logger.info("MOCK ENVIRONMENT CONFIGURED")
    logger.info(
        "   Identity: Rank %d/%d (Node %d)",
        rank, world_size, rank // local_world_size,
    )
    logger.info("   Network:  Sandbox -> %s", sandbox_url)
    logger.info("   Network:  Inference -> %s", inference_url)
    logger.info("=" * 60)


def get_all_rank_configs(
    num_nodes: int = 4,
    gpus_per_node: int = 2,
) -> list[RankConfig]:
    """Generate a RankConfig for every rank in the simulated cluster."""
    world_size = num_nodes * gpus_per_node
    configs = []

    for node_id in range(num_nodes):
        for local_rank in range(gpus_per_node):
            global_rank = node_id * gpus_per_node + local_rank
            configs.append(
                RankConfig(
                    rank=global_rank,
                    world_size=world_size,
                    local_rank=local_rank,
                    local_world_size=gpus_per_node,
                    node_id=node_id,
                )
            )

    return configs


def log_cluster_topology(num_nodes: int = 4, gpus_per_node: int = 2) -> None:
    """Log the simulated cluster topology."""
    configs = get_all_rank_configs(num_nodes, gpus_per_node)

    lines = ["Simulated cluster topology:"]
    current_node = -1
    for cfg in configs:
        if cfg.node_id != current_node:
            lines.append(f"  Node {cfg.node_id} (10.0.0.{cfg.node_id + 1})")
            current_node = cfg.node_id
        role = "MASTER" if cfg.rank == 0 else "WORKER"
        lines.append(f"    Rank {cfg.rank} (local_rank={cfg.local_rank}) [{role}]")

    logger.info("\n".join(lines))
