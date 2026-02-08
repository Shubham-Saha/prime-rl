#!/usr/bin/env python
"""
Integration script for PRIME-RL security analysis.

Spawns:
1. Mock Sandbox Server (for verifiers HTTP traffic analysis)
2. Mock Inference Server (for OpenAI-compatible API)
3. Trainer processes (simulated distributed training)
4. Orchestrator process (coordination)

Captures all network traffic for security analysis.
"""

import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MOCK_DIR = Path(__file__).parent
CONFIG_PATH = MOCK_DIR / "simulation_config.json"
RESULTS_DIR = Path("results")


def _setup_subprocess_logging(label: str) -> None:
    """Configure logging for a spawned subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{label}] %(levelname)s: %(message)s",
    )


def run_mock_sandbox_server(port: int = 8000, log_file: str = "results/sandbox_requests.jsonl") -> None:
    """Run mock sandbox server (blocking, intended for subprocess)."""
    _setup_subprocess_logging("SandboxServer")

    from prime_rl.mock.sandbox_server import MockSandboxServer

    server = MockSandboxServer(
        port=port,
        log_file=log_file,
        execute_locally=False,
    )
    server.run()


def run_mock_inference_server(port: int = 8080) -> None:
    """Run mock inference server (blocking, intended for subprocess)."""
    _setup_subprocess_logging("InferenceServer")

    from prime_rl.mock.mock_inference_server import run_server
    run_server(port=port)


def run_trainer_process(
    rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int,
    output_dir: Path,
    inference_url: str,
    sandbox_url: str,
) -> None:
    """Run a trainer process with simulated environment."""
    _setup_subprocess_logging(f"Trainer-{rank}")

    from prime_rl.mock.setup_env import setup_mock_environment
    setup_mock_environment(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        sandbox_url=sandbox_url,
        inference_url=inference_url,
    )
    
    from prime_rl.mock.device import patch_cuda_for_cpu

    patch_cuda_for_cpu()

    from prime_rl.trainer.world import get_world

    world = get_world()
    logger.info("Trainer rank=%d: world initialized: %s", rank, world)

    for step in range(3):
        time.sleep(1)
        logger.info("Trainer rank=%d: step %d complete", rank, step)
    
    logger.info("Trainer rank=%d: finished", rank)


def run_orchestrator_process(
    output_dir: Path,
    inference_url: str,
    sandbox_url: str,
) -> None:
    """Run orchestrator process."""
    import asyncio

    _setup_subprocess_logging("Orchestrator")

    from prime_rl.mock.setup_env import setup_mock_environment
    setup_mock_environment(
        rank=0,
        world_size=1,
        local_rank=0,
        local_world_size=1,
        sandbox_url=sandbox_url,
        inference_url=inference_url,
    )
    
    from prime_rl.utils.logger import setup_logger
    setup_logger(log_level="INFO", tag="Orchestrator")

    from prime_rl.utils.client import setup_clients, setup_admin_clients, check_health
    from prime_rl.utils.config import ClientConfig

    client_config = ClientConfig(
        base_url=[inference_url],
        api_key_var="OPENAI_API_KEY",
        timeout=60,
    )
    
    logger.info("Orchestrator connecting to inference: %s", inference_url)
    logger.info("Orchestrator sandbox endpoint: %s", sandbox_url)
    
    clients = setup_clients(client_config)
    admin_clients = setup_admin_clients(client_config)
    
    async def main():
        logger.info("Checking inference server health...")
        await check_health(admin_clients, timeout=30)
        logger.info("Inference server is healthy")
        
        client = clients[0]
        logger.info("Testing chat completion...")
        
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            n=2,
            temperature=0.7,
        )
        
        logger.info("Received %d completions", len(response.choices))
        for i, choice in enumerate(response.choices):
            content = choice.message.content or ""
            logger.info("  Choice %d: %s...", i, content[:50] if len(content) > 50 else content)
        
        # ============================================================
        # ACTUALLY TRIGGER VERIFIERS SANDBOX CALLS
        # ============================================================
        await test_verifiers_sandbox(sandbox_url, client)
    
    asyncio.run(main())
    logger.info("Orchestrator finished")


async def test_verifiers_sandbox(sandbox_url: str, openai_client) -> None:
    """
    Actually trigger verifiers to make HTTP calls to the sandbox server.
    
    This is what populates your sandbox_requests.jsonl log file.
    """
    logger.info("=" * 50)
    logger.info("TESTING VERIFIERS SANDBOX INTEGRATION")
    logger.info("=" * 50)
    
    try:
        import verifiers as vf
        from verifiers import load_environment
        
        logger.info("Verifiers library loaded successfully")
        logger.info("Sandbox requests will go to: %s", sandbox_url)
        
        # Try to load a code execution environment
        # This environment type triggers sandbox calls
        try:
            # Option 1: Try math-python (requires code execution)
            logger.info("Attempting to load 'math-python' environment...")
            env = load_environment("math-python")
            logger.info("Environment loaded: %s", type(env).__name__)
            
            # Create a sample rollout that would trigger code execution
            sample_example = {
                "prompt": [{"role": "user", "content": "Write Python code to calculate 2+2 and print the result."}],
                "example_id": 1,
            }
            
            # This should trigger an HTTP POST to your sandbox server
            logger.info("Triggering rollout (this should call sandbox)...")
            
            rollout_input = vf.RolloutInput(**sample_example)
            
            # Run a single rollout - this triggers the sandbox
            result = await env.run_rollout(
                rollout_input,
                client=openai_client,
                model="Qwen/Qwen3-0.6B",
                sampling_args={"temperature": 0.7, "max_tokens": 256},
            )
            
            logger.info("Rollout complete!")
            logger.info("  Reward: %s", result.get("reward", "N/A"))
            logger.info("  Error: %s", result.get("error", "None"))
            
        except Exception as e:
            logger.warning("Could not run full environment test: %s", e)
            logger.info("Falling back to direct sandbox API test...")
            
            # Fallback: Directly call the sandbox API to prove it works
            await test_sandbox_directly(sandbox_url)
            
    except ImportError as e:
        logger.warning("Verifiers library not installed: %s", e)
        logger.info("Falling back to direct sandbox API test...")
        await test_sandbox_directly(sandbox_url)


async def test_sandbox_directly(sandbox_url: str) -> None:
    """
    Directly call the sandbox API to verify connectivity.
    
    This bypasses verifiers and directly tests your mock server.
    """
    import aiohttp
    
    logger.info("Testing sandbox API directly...")
    
    # Test 1: Health check
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{sandbox_url}/health") as resp:
                data = await resp.json()
                logger.info("Health check response: %s", data)
        except Exception as e:
            logger.error("Health check failed: %s", e)
        
        # Test 2: Code execution endpoint
        try:
            payload = {
                "code": "print('Hello from security analysis!')\nresult = 2 + 2\nprint(f'Result: {result}')",
                "language": "python",
                "timeout": 30,
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer mock-api-key",
                "X-API-Key": "mock-api-key",
            }
            
            async with session.post(
                f"{sandbox_url}/v1/sandbox/run",
                json=payload,
                headers=headers,
            ) as resp:
                data = await resp.json()
                logger.info("Code execution response: %s", data)
                
        except Exception as e:
            logger.error("Code execution test failed: %s", e)
        
        # Test 3: Try alternative endpoint (discover what verifiers uses)
        try:
            async with session.post(
                f"{sandbox_url}/run",
                json=payload,
                headers=headers,
            ) as resp:
                data = await resp.json()
                logger.info("Alternative endpoint (/run) response: %s", data)
        except Exception as e:
            logger.error("Alternative endpoint failed: %s", e)
    
    logger.info("Direct sandbox tests complete - check sandbox_requests.jsonl")


class SecurityAnalysisRunner:
    """Spawns all components and captures network traffic for security analysis."""
    
    def __init__(
        self,
        num_trainer_nodes: int = 2,
        gpus_per_node: int = 2,
        inference_port: int = 8080,
        sandbox_port: int = 8000,
        output_dir: Path = Path("security_analysis_output"),
    ):
        self.num_trainer_nodes = num_trainer_nodes
        self.gpus_per_node = gpus_per_node
        self.inference_port = inference_port
        self.sandbox_port = sandbox_port
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processes: list[mp.Process] = []
        self.inference_process: mp.Process | None = None
        self.sandbox_process: mp.Process | None = None

        self.inference_url = f"http://127.0.0.1:{self.inference_port}/v1"
        self.sandbox_url = f"http://127.0.0.1:{self.sandbox_port}"
    
    @classmethod
    def from_config(cls, config_path: Path | None = None) -> "SecurityAnalysisRunner":
        path = config_path or CONFIG_PATH
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            sa = data.get("security_analysis", {})
            return cls(
                num_trainer_nodes=sa.get("num_trainer_nodes", 2),
                gpus_per_node=sa.get("gpus_per_node", 2),
                inference_port=sa.get("inference_port", 8080),
                sandbox_port=sa.get("sandbox_port", 8000),
                output_dir=Path(sa.get("output_dir", "security_analysis_output")),
            )
        return cls()
    
    def start(self):
        total_ranks = self.num_trainer_nodes * self.gpus_per_node
        
        logger.info("=" * 70)
        logger.info("SECURITY ANALYSIS CONFIGURATION")
        logger.info("=" * 70)
        logger.info(
            "  Cluster: %d nodes x %d GPUs = %d ranks",
            self.num_trainer_nodes, self.gpus_per_node, total_ranks,
        )
        logger.info("  Sandbox Server:   %s", self.sandbox_url)
        logger.info("  Inference Server: %s", self.inference_url)
        logger.info("  Output Directory: %s", self.output_dir)
        logger.info("=" * 70)
        
        logger.info("[1/4] Starting mock sandbox server...")
        sandbox_log = str(self.output_dir / "sandbox_requests.jsonl")
        self.sandbox_process = mp.Process(
            target=run_mock_sandbox_server,
            kwargs={"port": self.sandbox_port, "log_file": sandbox_log},
        )
        self.sandbox_process.start()
        logger.info("      PID: %s -> %s", self.sandbox_process.pid, self.sandbox_url)
        time.sleep(1)

        logger.info("[2/4] Starting mock inference server...")
        self.inference_process = mp.Process(
            target=run_mock_inference_server,
            kwargs={"port": self.inference_port},
        )
        self.inference_process.start()
        logger.info("      PID: %s -> %s", self.inference_process.pid, self.inference_url)
        time.sleep(2)

        logger.info("[3/4] Starting %d trainer processes...", total_ranks)
        world_size = total_ranks

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
                        "inference_url": self.inference_url,
                        "sandbox_url": self.sandbox_url,
                    },
                )
                p.start()
                self.processes.append(p)
                logger.info(
                    "      Rank %d (Node %d, Local %d) PID: %s",
                    global_rank, node_id, local_rank, p.pid,
                )

        logger.info("[4/4] Starting orchestrator...")
        orch_process = mp.Process(
            target=run_orchestrator_process,
            kwargs={
                "output_dir": self.output_dir,
                "inference_url": self.inference_url,
                "sandbox_url": self.sandbox_url,
            },
        )
        orch_process.start()
        self.processes.append(orch_process)
        logger.info("      PID: %s", orch_process.pid)
    
    def wait(self, timeout: float | None = None):
        logger.info("Waiting for processes to complete...")
        for p in self.processes:
            p.join(timeout=timeout)
    
    def stop(self):
        logger.info("Stopping all processes...")
        
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        
        if self.inference_process and self.inference_process.is_alive():
            self.inference_process.terminate()
            self.inference_process.join(timeout=5)
        
        if self.sandbox_process and self.sandbox_process.is_alive():
            self.sandbox_process.terminate()
            self.sandbox_process.join(timeout=5)
        
        logger.info("All processes stopped")
    
    def log_summary(self):
        """Generate security analysis summary."""
        log_files = list(self.output_dir.glob("*.log"))
        sandbox_log = self.output_dir / "sandbox_requests.jsonl"
        
        # Count sandbox requests
        sandbox_requests = 0
        sandbox_endpoints: set[str] = set()
        if sandbox_log.exists():
            with open(sandbox_log) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        sandbox_requests += 1
                        sandbox_endpoints.add(entry.get("path", "unknown"))
                    except json.JSONDecodeError:
                        pass
        
        lines = [
            "",
            "=" * 70,
            "SECURITY ANALYSIS SUMMARY",
            "=" * 70,
            "",
            f"  Output directory: {self.output_dir}",
            f"  Log files generated: {len(log_files)}",
        ]
        
        for f in log_files:
            lines.append(f"    - {f.name}")
        
        lines.extend([
            "",
            "  Sandbox Server Analysis:",
            f"    - Total HTTP requests captured: {sandbox_requests}",
            f"    - Endpoints hit: {list(sandbox_endpoints) if sandbox_endpoints else ['none']}",
            f"    - Request log: {sandbox_log}",
            "",
            "  Communication Channels Analyzed:",
            "",
            f"    [HTTP] Sandbox API ({self.sandbox_url}):",
            "           ├─ POST /v1/sandbox/run (code execution)",
            "           ├─ GET  /health (health check)",
            "           └─ *    /* (catch-all for discovery)",
            "",
            f"    [HTTP] Inference API ({self.inference_url}):",
            "           ├─ GET  /health (health check)",
            "           ├─ GET  /v1/models (model listing)",
            "           ├─ POST /v1/chat/completions (inference)",
            "           ├─ POST /init_broadcaster (NCCL init)",
            "           └─ POST /update_weights (weight sync)",
            "",
            "    [ZMQ] Transport Layer:",
            "           ├─ PUSH TrainingBatchSender",
            "           ├─ PULL TrainingBatchReceiver",
            "           ├─ PUB  MicroBatchSender",
            "           └─ SUB  MicroBatchReceiver",
            "",
            "    [NCCL/Filesystem] Weight Broadcast:",
            "           ├─ FileSystemWeightBroadcast",
            "           └─ NCCLWeightBroadcast",
            "",
            "=" * 70,
        ])
        
        logger.info("\n".join(lines))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("DISTRIBUTED FEDERATED LEARNING - SECURITY ANALYSIS")
    logger.info("=" * 70)

    runner = SecurityAnalysisRunner.from_config()

    try:
        runner.start()

        logger.info("=" * 70)
        logger.info("SIMULATION RUNNING")
        logger.info("=" * 70)
        logger.info("Communication channels being analyzed:")
        logger.info("  1. Sandbox API (HTTP)    -> %s", runner.sandbox_url)
        logger.info("  2. Inference API (HTTP)  -> %s", runner.inference_url)
        logger.info("  3. ZMQ Transport (TCP)   -> configurable ports")
        logger.info("  4. Weight Broadcast      -> NCCL/Filesystem")
        logger.info("Press Ctrl+C to stop and view analysis report.")
        logger.info("=" * 70)

        runner.wait(timeout=60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        runner.stop()
        runner.log_summary()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()