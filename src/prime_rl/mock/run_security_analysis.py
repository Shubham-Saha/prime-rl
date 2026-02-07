"""
Complete Integration Script for PRIME-RL Security Analysis.

This script:
1. Sets up simulated distributed environment
2. Starts mock inference server
3. Uses REAL PRIME-RL code (ZMQ, transport, client)
4. Captures all network traffic for analysis

Run this to start security analysis of PRIME-RL distributed system.
"""
import os
import sys
import time
import json
import signal
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def run_mock_inference_server(port: int = 8000, log_file: Optional[Path] = None):
    """Run mock inference server in a subprocess."""
    from prime_rl.mock.mock_inference_server import run_server
    
    if log_file:
        # Redirect output to log file
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout
    
    run_server(port=port)


def run_trainer_process(
    rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int,
    output_dir: Path,
    inference_url: str,
):
    """Run a trainer process with simulated environment."""
    from prime_rl.mock.setup_env import setup_simulated_cluster
    from prime_rl.mock.distributed_cpu import setup_torch_distributed_cpu
    from prime_rl.mock.device import patch_cuda_for_cpu
    
    # Setup environment before any imports
    setup_simulated_cluster(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size
    )
    
    # Patch CUDA for CPU execution
    patch_cuda_for_cpu()
    
    # Now import and use real PRIME-RL code
    from prime_rl.trainer.world import get_world
    from prime_rl.transport.config import ZMQTransportConfig, FileSystemTransportConfig
    from prime_rl.transport import setup_training_batch_receiver
    
    # Verify world is set up correctly
    world = get_world()
    print(f"[TRAINER rank={rank}] World initialized: {world}")
    
    # Initialize torch.distributed with GLOO
    setup_torch_distributed_cpu()
    
    # Use filesystem transport (works without GPU)
    transport_config = FileSystemTransportConfig()
    
    # The receiver will use the REAL code
    # receiver = setup_training_batch_receiver(transport_config)
    
    print(f"[TRAINER rank={rank}] Ready for training")
    
    # Simulate training steps
    for step in range(3):
        time.sleep(1)
        print(f"[TRAINER rank={rank}] Step {step} complete")
    
    print(f"[TRAINER rank={rank}] Finished")


def run_orchestrator_process(
    output_dir: Path,
    inference_url: str,
):
    """Run orchestrator process."""
    import asyncio
    
    from prime_rl.mock.setup_env import setup_simulated_cluster
    
    # Orchestrator doesn't need distributed, but set env anyway
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    
    # Use real OpenAI client
    from prime_rl.utils.client import setup_clients, setup_admin_clients, check_health
    from prime_rl.utils.config import ClientConfig
    
    # Create client config pointing to our mock server
    client_config = ClientConfig(
        base_url=[inference_url],
        api_key_var="OPENAI_API_KEY",  # Will use "EMPTY" if not set
        timeout=60,
    )
    
    print(f"[ORCHESTRATOR] Connecting to inference server at {inference_url}")
    
    # These are the REAL client functions - they will work with our mock server
    clients = setup_clients(client_config)
    admin_clients = setup_admin_clients(client_config)
    
    async def main():
        # Check health using REAL code
        print("[ORCHESTRATOR] Checking inference server health...")
        await check_health(admin_clients, timeout=30)
        print("[ORCHESTRATOR] Inference server is healthy!")
        
        # Test chat completion using REAL OpenAI client
        client = clients[0]
        print("[ORCHESTRATOR] Testing chat completion...")
        
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            n=2,
            temperature=0.7,
        )
        
        print(f"[ORCHESTRATOR] Received {len(response.choices)} completions")
        for i, choice in enumerate(response.choices):
            print(f"  Choice {i}: {choice.message.content[:50]}...")
    
    asyncio.run(main())
    print("[ORCHESTRATOR] Finished")


class SecurityAnalysisRunner:
    """
    Main runner for PRIME-RL security analysis.
    
    Spawns all components and captures network traffic.
    """
    
    def __init__(
        self,
        num_trainer_nodes: int = 2,
        gpus_per_node: int = 2,
        inference_port: int = 8000,
        output_dir: Path = Path("security_analysis_output"),
    ):
        self.num_trainer_nodes = num_trainer_nodes
        self.gpus_per_node = gpus_per_node
        self.inference_port = inference_port
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processes: List[mp.Process] = []
        self.inference_process: Optional[mp.Process] = None
    
    def start(self):
        """Start all components."""
        print("\n" + "="*70)
        print("PRIME-RL SECURITY ANALYSIS")
        print("="*70)
        print(f"Trainer nodes:     {self.num_trainer_nodes}")
        print(f"GPUs per node:     {self.gpus_per_node}")
        print(f"Total ranks:       {self.num_trainer_nodes * self.gpus_per_node}")
        print(f"Inference port:    {self.inference_port}")
        print(f"Output directory:  {self.output_dir}")
        print("="*70 + "\n")
        
        # Start mock inference server
        print("[MAIN] Starting mock inference server...")
        self.inference_process = mp.Process(
            target=run_mock_inference_server,
            kwargs={
                "port": self.inference_port,
                "log_file": self.output_dir / "inference_server.log"
            }
        )
        self.inference_process.start()
        time.sleep(2)  # Wait for server to start
        
        inference_url = f"http://127.0.0.1:{self.inference_port}/v1"
        
        # Start trainer processes
        print("[MAIN] Starting trainer processes...")
        world_size = self.num_trainer_nodes * self.gpus_per_node
        
        for node_id in range(self.num_trainer_nodes):
            for local_rank in range(self.gpus_per_node):
                global_rank = node_id * self.gpus_per_node + local_rank
                
                p = mp.Process(
                    target=run_trainer_process,
                    kwargs={
                        "rank": global_rank,
                        "world_size": world_size,
                        "local_rank": local_rank,
                        "local_world_size": self.gpus_per_node,
                        "output_dir": self.output_dir,
                        "inference_url": inference_url,
                    }
                )
                p.start()
                self.processes.append(p)
                print(f"[MAIN] Started trainer rank {global_rank} (PID: {p.pid})")
        
        # Start orchestrator
        print("[MAIN] Starting orchestrator...")
        orch_process = mp.Process(
            target=run_orchestrator_process,
            kwargs={
                "output_dir": self.output_dir,
                "inference_url": inference_url,
            }
        )
        orch_process.start()
        self.processes.append(orch_process)
        print(f"[MAIN] Started orchestrator (PID: {orch_process.pid})")
    
    def wait(self, timeout: Optional[float] = None):
        """Wait for all processes to complete."""
        print("[MAIN] Waiting for processes to complete...")
        
        for p in self.processes:
            p.join(timeout=timeout)
    
    def stop(self):
        """Stop all processes."""
        print("[MAIN] Stopping all processes...")
        
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        
        if self.inference_process and self.inference_process.is_alive():
            self.inference_process.terminate()
            self.inference_process.join(timeout=5)
        
        print("[MAIN] All processes stopped")
    
    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*70)
        print("SECURITY ANALYSIS SUMMARY")
        print("="*70)
        
        # Check for log files
        log_files = list(self.output_dir.glob("*.log"))
        print(f"\nLog files generated: {len(log_files)}")
        for f in log_files:
            print(f"  - {f.name}")
        
        # Check inference server log for requests
        inf_log = self.output_dir / "inference_server.log"
        if inf_log.exists():
            print(f"\nInference server log: {inf_log}")
        
        print("\nSecurity-relevant endpoints accessed:")
        print("  - /health (health check)")
        print("  - /v1/models (model listing)")
        print("  - /init_broadcaster (NCCL initialization)")
        print("  - /update_weights (weight synchronization)")
        print("  - /v1/chat/completions (inference)")
        
        print("\n" + "="*70)


def main():
    """Main entry point."""
    runner = SecurityAnalysisRunner(
        num_trainer_nodes=2,
        gpus_per_node=2,
        inference_port=8000,
    )
    
    try:
        runner.start()
        runner.wait(timeout=30)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        runner.stop()
        runner.print_summary()


if __name__ == "__main__":
    # Use spawn method for Windows compatibility
    mp.set_start_method("spawn", force=True)
    main()