"""
Mock Verifier for Security Analysis.

The real verifier in PRIME-RL:
1. Receives completions from orchestrator
2. Runs code/checks answers in a sandbox
3. Returns rewards (0.0 to 1.0)

For security analysis, this shows:
- What data flows to the verifier
- How the sandbox would be called
- Where remote API calls would happen
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockVerifier")


@dataclass
class VerificationResult:
    """Result from verifying a completion."""
    completion_id: str
    reward: float
    is_correct: bool
    execution_output: Optional[str] = None
    error: Optional[str] = None


class MockVerifier:
    """
    Mock verifier that simulates the real verification process.
    
    In production, this would:
    1. Connect to Prime Intellect's sandbox API
    2. Execute code in isolated environment
    3. Check mathematical/logical correctness
    
    For security analysis, we log what WOULD be sent to the sandbox.
    """
    
    def __init__(self, env_type: str = "math"):
        self.env_type = env_type
        self.verification_log: List[Dict] = []
        
    def verify(self, completion: str, problem: Dict[str, Any]) -> VerificationResult:
        """
        Verify a single completion.
        
        Args:
            completion: The model's response
            problem: The original problem with expected answer
            
        Returns:
            VerificationResult with reward
        """
        # Log what would be sent to remote sandbox
        log_entry = {
            "completion": completion[:200] + "..." if len(completion) > 200 else completion,
            "problem_type": problem.get("type", "unknown"),
            "would_execute": self.env_type == "code",
        }
        self.verification_log.append(log_entry)
        
        logger.info(f"[VERIFY] Checking completion ({len(completion)} chars)")
        
        if self.env_type == "code":
            return self._verify_code(completion, problem)
        elif self.env_type == "math":
            return self._verify_math(completion, problem)
        else:
            return self._verify_generic(completion, problem)
    
    def _verify_code(self, completion: str, problem: Dict) -> VerificationResult:
        """Verify code completion - would execute in sandbox."""
        
        # SECURITY NOTE: In production, this would:
        # 1. Send code to remote sandbox (Prime Intellect API)
        # 2. Execute in isolated container
        # 3. Check test case results
        
        logger.warning("[SECURITY] Code verification would send to remote sandbox!")
        logger.warning(f"[SECURITY] Sandbox endpoint: https://api.prime.intellect/sandbox")
        logger.warning(f"[SECURITY] Code length: {len(completion)} chars")
        
        # Simulate random result
        is_correct = random.random() > 0.5
        reward = 1.0 if is_correct else 0.0
        
        return VerificationResult(
            completion_id=f"comp_{random.randint(1000, 9999)}",
            reward=reward,
            is_correct=is_correct,
            execution_output="[MOCK] Test passed" if is_correct else "[MOCK] Test failed",
        )
    
    def _verify_math(self, completion: str, problem: Dict) -> VerificationResult:
        """Verify math answer."""
        
        expected = problem.get("answer", "")
        
        # Simple check: is the expected answer in the completion?
        is_correct = str(expected).lower() in completion.lower()
        reward = 1.0 if is_correct else 0.0
        
        logger.info(f"[VERIFY] Math check: expected={expected}, found={is_correct}")
        
        return VerificationResult(
            completion_id=f"comp_{random.randint(1000, 9999)}",
            reward=reward,
            is_correct=is_correct,
        )
    
    def _verify_generic(self, completion: str, problem: Dict) -> VerificationResult:
        """Generic verification with random reward."""
        reward = random.uniform(0, 1)
        
        return VerificationResult(
            completion_id=f"comp_{random.randint(1000, 9999)}",
            reward=reward,
            is_correct=reward > 0.5,
        )
    
    def get_verification_log(self) -> List[Dict]:
        """Get all verification attempts for security analysis."""
        return self.verification_log


# Integration with orchestrator
async def mock_verify_rollouts(
    rollouts: List[Dict],
    verifier: MockVerifier,
) -> List[float]:
    """
    Verify a batch of rollouts and return rewards.
    
    This is what the orchestrator calls after getting completions.
    """
    rewards = []
    
    for rollout in rollouts:
        completion = rollout.get("response", "")
        problem = rollout.get("problem", {})
        
        result = verifier.verify(completion, problem)
        rewards.append(result.reward)
    
    logger.info(f"[VERIFY] Batch complete: {len(rewards)} rewards, "
                f"avg={sum(rewards)/len(rewards):.2f}")
    
    return rewards