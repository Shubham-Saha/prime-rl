"""
Local sandbox replacement for verifier environments.
Executes code locally with security restrictions instead of remote sandboxes.

This module intercepts calls that would normally go to Prime Intellect's
paid sandbox infrastructure and executes them locally.
"""
import sys
import io
import ast
import traceback
import resource
import signal
import multiprocessing as mp
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
import json
import socket


@dataclass
class SandboxConfig:
    """Configuration for local sandbox execution."""
    timeout_seconds: int = 30
    max_memory_mb: int = 256  # Conservative for 8GB RAM
    max_output_bytes: int = 1024 * 1024  # 1MB
    allow_imports: List[str] = field(default_factory=lambda: [
        "math", "random", "string", "collections", "itertools",
        "functools", "operator", "re", "json", "datetime", "typing"
    ])
    blocked_builtins: List[str] = field(default_factory=lambda: [
        "exec", "eval", "compile", "open", "input", "__import__",
        "globals", "locals", "vars", "dir", "getattr", "setattr",
        "delattr", "breakpoint"
    ])


@dataclass
class SandboxResult:
    """Result from sandbox execution."""
    success: bool
    output: str
    error: Optional[str]
    execution_time_ms: float
    memory_used_mb: float
    was_terminated: bool = False
    termination_reason: Optional[str] = None


class CodeValidator:
    """Validates code before execution in sandbox."""
    
    DANGEROUS_PATTERNS = [
        "os.system", "subprocess", "socket", "requests", "urllib",
        "shutil.rmtree", "__builtins__", "importlib", "ctypes",
        "pickle", "marshal"
    ]
    
    @classmethod
    def validate(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code for dangerous patterns.
        Returns (is_safe, error_message).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for dangerous imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not cls._is_safe_import(alias.name):
                        return False, f"Blocked import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and not cls._is_safe_import(node.module):
                    return False, f"Blocked import: {node.module}"
        
        # Check for dangerous patterns in code
        code_lower = code.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.lower() in code_lower:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, None
    
    @staticmethod
    def _is_safe_import(module_name: str) -> bool:
        """Check if an import is in the allowed list."""
        safe_modules = [
            "math", "random", "string", "collections", "itertools",
            "functools", "operator", "re", "json", "datetime", "typing",
            "dataclasses", "enum", "abc", "copy", "heapq", "bisect"
        ]
        base_module = module_name.split(".")[0]
        return base_module in safe_modules


class LocalSandbox:
    """
    Local sandbox for code execution.
    Replaces Prime Intellect's remote sandbox with local restricted execution.
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.execution_count = 0
        self._active_connections: Dict[str, Dict] = {}
        
    def execute(self, code: str, test_code: Optional[str] = None) -> SandboxResult:
        """
        Execute code in a restricted environment.
        
        Args:
            code: The code to execute
            test_code: Optional test code to run after main code
            
        Returns:
            SandboxResult with execution details
        """
        start_time = time.perf_counter()
        
        # Validate code first
        is_safe, error = CodeValidator.validate(code)
        if not is_safe:
            return SandboxResult(
                success=False,
                output="",
                error=error,
                execution_time_ms=0,
                memory_used_mb=0,
                was_terminated=True,
                termination_reason="Code validation failed"
            )
        
        # Execute in subprocess for isolation
        result = self._execute_in_subprocess(code, test_code)
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        result.execution_time_ms = execution_time_ms
        
        self.execution_count += 1
        return result
    
    def _execute_in_subprocess(
        self, 
        code: str, 
        test_code: Optional[str]
    ) -> SandboxResult:
        """Execute code in an isolated subprocess."""
        
        # Create a queue for results
        result_queue = mp.Queue()
        
        # Create subprocess
        process = mp.Process(
            target=self._subprocess_executor,
            args=(code, test_code, self.config, result_queue)
        )
        
        process.start()
        process.join(timeout=self.config.timeout_seconds)
        
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            
            return SandboxResult(
                success=False,
                output="",
                error="Execution timeout",
                execution_time_ms=self.config.timeout_seconds * 1000,
                memory_used_mb=0,
                was_terminated=True,
                termination_reason="Timeout exceeded"
            )
        
        # Get result from queue
        try:
            result = result_queue.get_nowait()
            return SandboxResult(**result)
        except:
            return SandboxResult(
                success=False,
                output="",
                error="Failed to retrieve execution result",
                execution_time_ms=0,
                memory_used_mb=0
            )
    
    @staticmethod
    def _subprocess_executor(
        code: str,
        test_code: Optional[str],
        config: SandboxConfig,
        result_queue: mp.Queue
    ):
        """Executor function that runs in subprocess."""
        try:
            # Set resource limits (Unix only)
            if sys.platform != "win32":
                # Memory limit
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(
                    resource.RLIMIT_AS, 
                    (config.max_memory_mb * 1024 * 1024, hard)
                )
                
                # CPU time limit
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (config.timeout_seconds, config.timeout_seconds + 5)
                )
            
            # Create restricted globals
            safe_builtins = {
                k: v for k, v in __builtins__.items()
                if k not in config.blocked_builtins
            } if isinstance(__builtins__, dict) else {
                k: getattr(__builtins__, k) 
                for k in dir(__builtins__)
                if k not in config.blocked_builtins and not k.startswith('_')
            }
            
            restricted_globals = {
                "__builtins__": safe_builtins,
                "__name__": "__sandbox__",
            }
            
            # Import allowed modules
            for module_name in config.allow_imports:
                try:
                    restricted_globals[module_name] = __import__(module_name)
                except ImportError:
                    pass
            
            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            
            try:
                # Execute main code
                exec(code, restricted_globals)
                
                # Execute test code if provided
                if test_code:
                    exec(test_code, restricted_globals)
                
                success = True
                error = None
            except Exception as e:
                success = False
                error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
            
            output = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                output += f"\nSTDERR:\n{stderr_capture.getvalue()}"
            
            # Truncate output if too large
            if len(output) > config.max_output_bytes:
                output = output[:config.max_output_bytes] + "\n... [truncated]"
            
            result_queue.put({
                "success": success,
                "output": output,
                "error": error,
                "execution_time_ms": 0,  # Will be set by parent
                "memory_used_mb": 0,  # TODO: Measure actual memory
                "was_terminated": False,
                "termination_reason": None
            })
            
        except Exception as e:
            result_queue.put({
                "success": False,
                "output": "",
                "error": f"Sandbox error: {str(e)}",
                "execution_time_ms": 0,
                "memory_used_mb": 0,
                "was_terminated": True,
                "termination_reason": "Subprocess error"
            })
    
    def get_sandbox_info(self) -> Dict[str, Any]:
        """Get information about the sandbox that would be displayed in terminal."""
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "127.0.0.1"
            
        return {
            "type": "local",
            "hostname": hostname,
            "ip_address": local_ip,
            "location": "LOCAL_MACHINE",
            "execution_count": self.execution_count,
            "config": {
                "timeout_seconds": self.config.timeout_seconds,
                "max_memory_mb": self.config.max_memory_mb,
                "allowed_imports": self.config.allow_imports
            },
            "note": "Running locally instead of Prime Intellect cloud sandbox"
        }
    
    def print_sandbox_status(self):
        """Print sandbox status to terminal."""
        info = self.get_sandbox_info()
        print("\n" + "="*60)
        print("LOCAL SANDBOX STATUS")
        print("="*60)
        print(f"Type:              {info['type']}")
        print(f"Hostname:          {info['hostname']}")
        print(f"IP Address:        {info['ip_address']}")
        print(f"Location:          {info['location']}")
        print(f"Executions:        {info['execution_count']}")
        print(f"Timeout:           {info['config']['timeout_seconds']}s")
        print(f"Max Memory:        {info['config']['max_memory_mb']}MB")
        print("-"*60)
        print(f"NOTE: {info['note']}")
        print("="*60 + "\n")


class SandboxInterceptor:
    """
    Intercepts calls to remote sandbox and redirects to local execution.
    This is the main class you'll use to replace the verifier sandbox.
    """
    
    def __init__(self):
        self.local_sandbox = LocalSandbox()
        self.intercepted_count = 0
        self.remote_endpoints_intercepted: List[str] = []
        
    def intercept_remote_call(
        self,
        endpoint: str,
        code: str,
        test_code: Optional[str] = None,
        **kwargs
    ) -> SandboxResult:
        """
        Intercept a call that would go to a remote sandbox.
        
        Args:
            endpoint: The remote endpoint that was being called
            code: Code to execute
            test_code: Optional test code
            **kwargs: Additional arguments (ignored)
            
        Returns:
            SandboxResult from local execution
        """
        self.intercepted_count += 1
        self.remote_endpoints_intercepted.append(endpoint)
        
        print(f"\n[SANDBOX INTERCEPT] Redirecting call to: {endpoint}")
        print(f"[SANDBOX INTERCEPT] Executing locally instead...")
        
        result = self.local_sandbox.execute(code, test_code)
        
        print(f"[SANDBOX INTERCEPT] Execution complete: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"[SANDBOX INTERCEPT] Time: {result.execution_time_ms:.2f}ms")
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get interceptor status."""
        return {
            "intercepted_calls": self.intercepted_count,
            "unique_endpoints": list(set(self.remote_endpoints_intercepted)),
            "sandbox_info": self.local_sandbox.get_sandbox_info()
        }


# Global interceptor instance
_SANDBOX_INTERCEPTOR: Optional[SandboxInterceptor] = None


def get_sandbox_interceptor() -> SandboxInterceptor:
    """Get or create the global sandbox interceptor."""
    global _SANDBOX_INTERCEPTOR
    if _SANDBOX_INTERCEPTOR is None:
        _SANDBOX_INTERCEPTOR = SandboxInterceptor()
    return _SANDBOX_INTERCEPTOR


def execute_locally(code: str, test_code: Optional[str] = None) -> SandboxResult:
    """Convenience function to execute code locally."""
    return get_sandbox_interceptor().local_sandbox.execute(code, test_code)