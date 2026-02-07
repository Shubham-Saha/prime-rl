"""
Mock Inference Server implementing the EXACT same HTTP API as vLLM.

This server responds to all endpoints that src/prime_rl/utils/client.py calls:
- GET  /health                    -> check_health()
- GET  /v1/models                 -> maybe_check_has_model()
- POST /update_weights            -> update_weights()
- POST /reload_weights            -> reload_weights()
- POST /load_lora_adapter         -> load_lora_adapter()
- POST /v1/unload_lora_adapter    -> unload_lora_adapter()
- POST /init_broadcaster          -> init_nccl_broadcast()
- POST /v1/chat/completions       -> OpenAI chat API

The real client code in client.py will work UNCHANGED against this server.
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Use standard library for minimal dependencies
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockInferenceServer")


@dataclass
class ServerState:
    """Tracks server state for security analysis."""
    current_model: str = "Qwen/Qwen3-0.6B"
    loaded_adapters: Dict[str, str] = field(default_factory=dict)
    nccl_initialized: bool = False
    request_log: List[Dict] = field(default_factory=list)
    

# Global state
state = ServerState()


class MockInferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler implementing vLLM-compatible API."""
    
    def log_message(self, format: str, *args) -> None:
        """Override to use our logger."""
        logger.debug(f"{self.address_string()} - {format % args}")
    
    def _send_json_response(self, data: Dict, status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _read_json_body(self) -> Dict:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        return {}
    
    def _log_request(self, endpoint: str, data: Optional[Dict] = None) -> None:
        """Log request for security analysis."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers),
            "body": data,
            "client": self.client_address,
        }
        state.request_log.append(entry)
        logger.info(f"[{self.command}] {endpoint}")
    
    # =========================================================================
    # GET Endpoints
    # =========================================================================
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        
        if self.path == "/health":
            # Used by check_health() in client.py
            self._log_request("/health")
            self._send_json_response({"status": "healthy"})
            
        elif self.path == "/v1/models":
            # Used by maybe_check_has_model() in client.py
            self._log_request("/v1/models")
            
            models = [
                {
                    "id": state.current_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "organization",
                }
            ]
            
            # Add loaded LoRA adapters as models
            for name, path in state.loaded_adapters.items():
                models.append({
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "organization",
                    "parent": state.current_model,
                })
            
            self._send_json_response({
                "object": "list",
                "data": models
            })
            
        elif self.path == "/debug/requests":
            # Debug endpoint to view all logged requests
            self._send_json_response({"requests": state.request_log})
            
        elif self.path == "/debug/state":
            # Debug endpoint to view server state
            self._send_json_response(asdict(state))
            
        else:
            self.send_error(404, f"Not found: {self.path}")
    
    # =========================================================================
    # POST Endpoints
    # =========================================================================
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        body = self._read_json_body()
        
        if self.path == "/update_weights":
            # Used by update_weights() in client.py
            # Real vLLM loads weights from weight_dir
            self._log_request("/update_weights", body)
            
            weight_dir = body.get("weight_dir")
            logger.info(f"[WEIGHT UPDATE] weight_dir={weight_dir}")
            
            self._send_json_response({
                "status": "success",
                "weight_dir": weight_dir
            })
            
        elif self.path == "/reload_weights":
            # Used by reload_weights() in client.py
            # Resets to base model
            self._log_request("/reload_weights", body)
            
            state.loaded_adapters.clear()
            logger.info("[WEIGHT RELOAD] Reset to base model")
            
            self._send_json_response({"status": "success"})
            
        elif self.path == "/load_lora_adapter":
            # Used by load_lora_adapter() in client.py
            self._log_request("/load_lora_adapter", body)
            
            lora_name = body.get("lora_name")
            lora_path = body.get("lora_path")
            
            state.loaded_adapters[lora_name] = lora_path
            logger.info(f"[LORA LOAD] name={lora_name}, path={lora_path}")
            
            self._send_json_response({
                "status": "success",
                "lora_name": lora_name
            })
            
        elif self.path == "/v1/unload_lora_adapter":
            # Used by unload_lora_adapter() in client.py
            self._log_request("/v1/unload_lora_adapter", body)
            
            lora_name = body.get("lora_name")
            if lora_name in state.loaded_adapters:
                del state.loaded_adapters[lora_name]
            logger.info(f"[LORA UNLOAD] name={lora_name}")
            
            self._send_json_response({"status": "success"})
            
        elif self.path == "/init_broadcaster":
            # Used by init_nccl_broadcast() in client.py
            # This is where NCCL gets initialized between trainer and inference
            self._log_request("/init_broadcaster", body)
            
            host = body.get("host")
            port = body.get("port")
            server_rank = body.get("server_rank")
            num_servers = body.get("num_inference_server")
            timeout = body.get("timeout")
            
            state.nccl_initialized = True
            logger.info(f"[NCCL INIT] host={host}:{port}, "
                       f"rank={server_rank}/{num_servers}, timeout={timeout}s")
            
            # This is a security-relevant endpoint!
            logger.warning("[SECURITY] NCCL broadcaster initialization requested")
            logger.warning(f"[SECURITY] Network config: {host}:{port}")
            
            self._send_json_response({"status": "success"})
            
        elif self.path == "/v1/chat/completions":
            # OpenAI-compatible chat completions API
            # Used by orchestrator for generating rollouts
            self._log_request("/v1/chat/completions", {
                "model": body.get("model"),
                "n": body.get("n", 1),
                "temperature": body.get("temperature", 1.0),
                "num_messages": len(body.get("messages", [])),
            })
            
            model = body.get("model", state.current_model)
            messages = body.get("messages", [])
            n = body.get("n", 1)
            temperature = body.get("temperature", 1.0)
            logprobs = body.get("logprobs", False)
            
            # Generate mock completions
            choices = []
            for i in range(n):
                mock_content = (
                    f"<think>Let me analyze this step by step.</think>\n"
                    f"Based on my analysis, the answer is {42 + i}."
                )
                
                choice = {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": mock_content
                    },
                    "finish_reason": "stop",
                }
                
                if logprobs:
                    choice["logprobs"] = {"content": []}
                
                choices.append(choice)
            
            response = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50 * n,
                    "total_tokens": 100 + 50 * n,
                }
            }
            
            self._send_json_response(response)
            
        else:
            self.send_error(404, f"Not found: {self.path}")


class MockInferenceServer:
    """Mock inference server that can be started/stopped."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the server in a background thread."""
        self.server = HTTPServer((self.host, self.port), MockInferenceHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Mock inference server started on http://{self.host}:{self.port}")
        logger.info("Endpoints:")
        logger.info("  GET  /health")
        logger.info("  GET  /v1/models")
        logger.info("  POST /update_weights")
        logger.info("  POST /reload_weights")
        logger.info("  POST /load_lora_adapter")
        logger.info("  POST /v1/unload_lora_adapter")
        logger.info("  POST /init_broadcaster")
        logger.info("  POST /v1/chat/completions")
        logger.info("  GET  /debug/requests")
        logger.info("  GET  /debug/state")
    
    def stop(self) -> None:
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            logger.info("Mock inference server stopped")
    
    def get_request_log(self) -> List[Dict]:
        """Get all logged requests for security analysis."""
        return state.request_log
    
    def save_request_log(self, path: Path) -> None:
        """Save request log to file."""
        with open(path, "w") as f:
            for entry in state.request_log:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Request log saved to {path}")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the mock inference server (blocking)."""
    server = HTTPServer((host, port), MockInferenceHandler)
    logger.info(f"Starting mock inference server on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    run_server()