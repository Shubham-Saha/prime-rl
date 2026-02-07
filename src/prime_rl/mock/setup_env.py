"""
Environment Setup for Local Multi-Process Simulation.

This script sets environment variables to simulate a distributed cluster.
Run this BEFORE importing any prime_rl modules.

Usage:
    from prime_rl.mock.setup_env import setup_simulated_cluster
    setup_simulated_cluster(rank=0, world_size=4, local_rank=0, local_world_size=2)
    
    # Now import prime_rl - it will think it's running on a cluster
    from prime_rl.trainer.world import get_world
"""
import os
from typing import List, Dict


def setup_simulated_cluster(
    rank: int = 0,
    world_size: int = 8,
    local_rank: int = 0,
    local_world_size: int = 2,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    """
    Set environment variables to simulate a distributed cluster.
    
    The real PRIME-RL reads these from environment (see src/prime_rl/trainer/envs.py).
    By setting them, the original World class works without modification.
    
    Args:
        rank: Global rank (0 to world_size-1)
        world_size: Total number of processes
        local_rank: Rank within this node (0 to local_world_size-1)
        local_world_size: Number of processes per node
        master_addr: IP of master node
        master_port: Port for distributed rendezvous
        
    Example configurations:
        - 4 nodes × 2 GPUs = 8 ranks total
        - Node 0: rank=0,1 with local_rank=0,1
        - Node 1: rank=2,3 with local_rank=0,1
        - etc.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    
    # Disable CUDA to force CPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    print(f"[SIMULATED CLUSTER] rank={rank}/{world_size}, "
          f"local_rank={local_rank}/{local_world_size}, "
          f"master={master_addr}:{master_port}")


def get_all_rank_configs(
    num_nodes: int = 4,
    gpus_per_node: int = 2
) -> List[Dict]:
    """
    Generate environment configurations for all ranks.
    Use this to spawn multiple processes, each with different rank.
    
    Returns list of dicts, each containing env vars for one rank.
    """
    configs = []
    world_size = num_nodes * gpus_per_node
    
    for node_id in range(num_nodes):
        for local_rank in range(gpus_per_node):
            global_rank = node_id * gpus_per_node + local_rank
            configs.append({
                "RANK": str(global_rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(local_rank),
                "LOCAL_WORLD_SIZE": str(gpus_per_node),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "node_id": node_id,  # Extra info for logging
            })
    
    return configs


def print_cluster_topology(num_nodes: int = 4, gpus_per_node: int = 2):
    """Print the simulated cluster topology."""
    print("\n" + "="*60)
    print("SIMULATED CLUSTER TOPOLOGY")
    print("="*60)
    
    configs = get_all_rank_configs(num_nodes, gpus_per_node)
    
    current_node = -1
    for cfg in configs:
        node_id = cfg["node_id"]
        if node_id != current_node:
            print(f"\n  Node {node_id} (10.0.0.{node_id+1})")
            current_node = node_id
        
        rank = cfg["RANK"]
        local_rank = cfg["LOCAL_RANK"]
        role = "MASTER" if rank == "0" else "WORKER"
        print(f"    └── Rank {rank} (local_rank={local_rank}) [{role}]")
    
    print("\n" + "="*60)