# GEMINI.md

## Overview

This file provides guidelines for Gemini AI to assist with **Mock Simulation of Distributed Federated Learning (DFL)** and **Network Security**.
The context is a PhD project analyzing **Topology Attacks**, **Network Security**, and **Algorithmic Exploitation** in decentralized GPU clusters.

---

## Code Guidelines

### General Principles

- **Avoid try/except blocks** unless it's really necessary.  It's fine that a program fails if something goes wrong as this helps us to catch non-obvious bugs and unforeseen side-effects earlier. You can add try catch on code that explicitly aims to be fault tolerant like adding retry mechanisms or explicit and intentional robustness.
- **Do not add unnecessary comments.** Especially do not try to explain code change that reflect your work process, do not refer to old code. "The code used to do that but now we are doing this" is not a pattern we want. Instead prefer to use targeted comments sparingly to explain ambiguous code.

- **Keep code self-documenting** through clear naming conventions and logical structure.

### Zen of Python

Remember the Zen of Python when writing code:

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Errors should never pass silently.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

---

## Operational Instructions (Critical)

### Running Code

- All code should be runnable with `uv run` or `uv run <command>`.
- All dependencies should already be installed and pinned in the lock file. If not, add it to `pyproject.toml` and run `uv sync --all-extras` to install it.

### CLI Usage (Repo Specifics)

- **Config files use `@` syntax**: `uv run sft @ path/to/config.toml`
- **Multi-GPU with torchrun**: `uv run torchrun --nproc-per-node 2 src/prime_rl/trainer/sft/train.py @ path/to/config.toml`
- **Boolean flags don't need `true`**: use `--model.optim_cpu_offload` not `--model.optim_cpu_offload true`, use `--no-model.optim_cpu_offload` to pass False.
- **Override config values with CLI flags**: `--model.name Qwen/Qwen3-0.6B --training.max_steps 100`

### Testing

Write tests as plain functions with pytest fixtures. Don't use class-based tests.

### Git

Branch prefixes: `feature/`, `fix/`, `chore/`, `security/`, `simulation/`

---

## Mock Simulation Development

### Structure & Organization

- **Separate concerns**: Keep simulation logic, data models, and I/O operations in distinct modules.
- **Use factories**: Create factory functions/classes for generating mock objects consistently.
- **Configuration-driven**: Simulations should be configurable via external files (YAML, TOML, JSON).

### Mock Data Best Practices

```python
# Prefer dataclasses for mock entities
from dataclasses import dataclass, field
from typing import Optional
import uuid

@dataclass
class MockEntity:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    state: dict = field(default_factory=dict)
```

### Simulation Patterns

- **Deterministic by default**: Use seeded random generators for reproducibility.
- **Time abstraction**: Use a simulation clock, not real time.
- **Event-driven**: Prefer event queues over tight polling loops.
- **State snapshots**: Implement checkpoint/restore for debugging.

---

### Network Fidelity (Crucial for Phase 1)
  - Simulate **latency** (not just logical ticks, but milliseconds).
  - Simulate **packet loss** and **connection drops**.
  - Abstract the **Topology** (Who connects to whom?) using graph structures (e.g., adjacency lists).

```python
import random
from dataclasses import dataclass, field

@dataclass
class NetworkSimulation:
    seed: int = 1
    tick: int = 0
    latency_ms: tuple[int, int] = (10, 100)  # min, max
    packet_loss_rate: float = 0.01
    
    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.events = []
    
    def step(self):
        self.tick += 1
        # Process events for this tick
    
    def simulate_latency(self) -> int:
        return self.rng.randint(*self.latency_ms)
    
    def should_drop_packet(self) -> bool:
        return self.rng.random() < self.packet_loss_rate
```

### Testing Simulations

- Write **property-based tests** for simulation invariants.
- Use **pytest fixtures** for common simulation setups.
- Test **edge cases**: empty states, maximum loads, boundary conditions.

---

## Security Analysis

### Distributed AI Security Patterns

When generating or reviewing code, check for these specific AI/Network risks:

#### 1. Deserialization Safety

| | Pattern |
|---|---------|
| ❌ BAD | Using `pickle.load()` on data from the network (RCE risk) |
| ✅ GOOD | Use `torch.load(..., weights_only=True)` or `safetensors` |
| ✅ GOOD | Verify signatures/hashes *before* deserializing objects |

#### 2. Network Protocol & Logic

| | Pattern |
|---|---------|
| ❌ BAD | Trusting node IDs sent by the peer (Sybil risk) |
| ✅ GOOD | Verify identity via cryptographic signatures |
| ❌ BAD | Infinite waits on network sockets (DoS risk) |
| ✅ GOOD | Strict timeouts on all network operations (ZMQ/TCP) |

#### 3. Data Poisoning & Validation

| | Pattern |
|---|---------|
| ❌ BAD | Aggregating model updates without range checks (Poisoning risk) |
| ✅ GOOD | Validate gradient norms and shapes before aggregation |

### Security Analysis Workflow

1. **Static Analysis**: Run `bandit`, `semgrep`, or `safety`
2. **Dependency Audit**: Check for known vulnerabilities in dependencies
3. **Secret Scanning**: Ensure no secrets are committed
4. **Threat Modeling**: Identify attack vectors and trust boundaries

---

## PhD Project Context

### Simulation Domain

| Component | Description |
|-----------|-------------|
| **Architecture** | Hierarchical Distributed Federated Learning (DFL) |
| **Nodes** | Trainers (Clients), Aggregators (Orchestrators), Inference Servers (vLLM) |
| **Protocols** | Gloo/NCCL (Simulated via TCP), ZMQ (Control plane) |
| **Topology** | P2P Mesh with dynamic neighbor selection |

### Research Phases (PhD Roadmap)

| Phase | Focus | Target |
|-------|-------|--------|
| **Phase 1** | Network | Topology Poisoning & Eclipse Attacks (Targeting peer tables) |
| **Phase 2** | Algorithm | Adversarial Exploitation of PSO/Genetic Algorithms (Targeting fitness functions) |
| **Phase 3** | Privacy | Inter-DC Traffic Analysis & Inference Side-Channels |
| **Phase 4** | Defense | Trust-Aware Hierarchical Aggregation Protocol |

### Research Output Guidelines

- **Metrics over Logs**: Instead of printing text, log structured metrics (`Step`, `Accuracy`, `Latency`, `AttackSuccess`).
- **Artifacts**: Simulations must output data in analysis-friendly formats (CSV, JSONL, Parquet) to the `results/` directory.
- **Reproducibility**: Every output file must include the `seed` and `config` used to generate it in its metadata.
---

```python
import json
from dataclasses import dataclass, asdict

@dataclass
class ExperimentMetadata:
    seed: int
    config_path: str
    timestamp: str
    git_commit: str

def save_results(data: list[dict], metadata: ExperimentMetadata, path: str):
    output = {
        "metadata": asdict(metadata),
        "results": data
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
```

---

### Code Review Checklist

When reviewing code for security:

1. **Input Validation**
   - [ ] All external inputs are validated and sanitized
   - [ ] Type checking is enforced
   - [ ] Length/size limits are applied

2. **Authentication & Authorization**
   - [ ] Authentication is required for sensitive operations
   - [ ] Authorization checks use principle of least privilege
   - [ ] Session management is secure

3. **Data Protection**
   - [ ] Sensitive data is encrypted at rest and in transit
   - [ ] Secrets are not hardcoded (use environment variables or secret managers)
   - [ ] PII is handled according to compliance requirements

4. **Error Handling**
   - [ ] Errors don't leak sensitive information
   - [ ] Stack traces are not exposed to users
   - [ ] Logging doesn't include secrets

---


```bash
# Security scanning commands
uv run bandit -r src/
uv run safety check
uv run semgrep --config auto src/
```

---

## Running Code

- Use `uv run` or `uv run <command>` to execute code.
- Manage dependencies in `pyproject.toml` and sync with `uv sync --all-extras`.

---

## Testing

- Write tests as **plain functions** with pytest fixtures.
- **No class-based tests** unless necessary for shared state.
- Aim for high coverage on security-critical paths.

```python
import pytest

@pytest.fixture
def mock_simulation():
    return Simulation(seed=42)

def test_simulation_deterministic(mock_simulation):
    result1 = mock_simulation.step()
    mock_simulation_2 = Simulation(seed=42)
    result2 = mock_simulation_2.step()
    assert result1 == result2
```

---

## Git Workflow

- **Branch prefixes**: `feature/`, `fix/`, `security/`, `simulation/`
- **Commit messages**: Use conventional commits (feat:, fix:, security:, docs:)
- **Never commit**: Secrets, API keys, credentials, or sensitive test data

---




## Quick Reference

| Task | Command |
|------|---------|
| Run Script | `uv run script.py` |
| Run with Config | `uv run script.py @ config.toml` |
| Run Distributed | `uv run torchrun --nproc-per-node 2 script.py @ config.toml` |
| Run Tests | `uv run pytest` |
| Security Scan | `uv run bandit -r src/` |
| Dependency Check | `uv run safety check` |
| Format Code | `uv run ruff format .` |
| Lint Code | `uv run ruff check .` |
| Sync Dependencies | `uv sync --all-extras` |