"""
Mock inference server implementing the same HTTP API as vLLM.

Endpoints (matching src/prime_rl/utils/client.py):
  GET  /health                  -> check_health()
  GET  /v1/models               -> maybe_check_has_model()
  POST /update_weights          -> update_weights()
  POST /reload_weights          -> reload_weights()
  POST /load_lora_adapter       -> load_lora_adapter()
  POST /v1/unload_lora_adapter  -> unload_lora_adapter()
  POST /init_broadcaster        -> init_nccl_broadcast()
  POST /v1/chat/completions     -> OpenAI chat API
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    """Tracks server state for security analysis."""

    current_model: str = "Qwen/Qwen3-0.6B"
    loaded_adapters: dict[str, str] = field(default_factory=dict)
    nccl_initialized: bool = False
    request_log: list[dict] = field(default_factory=list)


class MockInferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler implementing vLLM-compatible API."""

    def log_message(self, format: str, *args) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)

    @property
    def state(self) -> ServerState:
        return self.server.state

    def _send_json_response(self, data: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        return {}

    def _log_request(self, endpoint: str, data: Optional[dict] = None) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers),
            "body": data,
            "client": self.client_address,
        }
        self.state.request_log.append(entry)
        logger.info("[%s] %s", self.command, endpoint)

    # --- GET ---

    def do_GET(self) -> None:
        if self.path == "/health":
            self._log_request("/health")
            self._send_json_response({"status": "healthy"})

        elif self.path == "/v1/models":
            self._log_request("/v1/models")
            models = [
                {
                    "id": self.state.current_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "organization",
                }
            ]
            for name in self.state.loaded_adapters:
                models.append({
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "organization",
                    "parent": self.state.current_model,
                })
            self._send_json_response({"object": "list", "data": models})

        elif self.path == "/debug/requests":
            self._send_json_response({"requests": self.state.request_log})

        elif self.path == "/debug/state":
            self._send_json_response(asdict(self.state))

        else:
            self.send_error(404, f"Not found: {self.path}")

    # --- POST ---

    def do_POST(self) -> None:
        body = self._read_json_body()

        if self.path == "/update_weights":
            self._log_request("/update_weights", body)
            weight_dir = body.get("weight_dir")
            logger.info("Weight update: weight_dir=%s", weight_dir)
            self._send_json_response({"status": "success", "weight_dir": weight_dir})

        elif self.path == "/reload_weights":
            self._log_request("/reload_weights", body)
            self.state.loaded_adapters.clear()
            logger.info("Weight reload: reset to base model")
            self._send_json_response({"status": "success"})

        elif self.path == "/load_lora_adapter":
            self._log_request("/load_lora_adapter", body)
            lora_name = body.get("lora_name")
            lora_path = body.get("lora_path")
            self.state.loaded_adapters[lora_name] = lora_path
            logger.info("LoRA load: name=%s, path=%s", lora_name, lora_path)
            self._send_json_response({"status": "success", "lora_name": lora_name})

        elif self.path == "/v1/unload_lora_adapter":
            self._log_request("/v1/unload_lora_adapter", body)
            lora_name = body.get("lora_name")
            self.state.loaded_adapters.pop(lora_name, None)
            logger.info("LoRA unload: name=%s", lora_name)
            self._send_json_response({"status": "success"})

        elif self.path == "/init_broadcaster":
            self._log_request("/init_broadcaster", body)
            host = body.get("host")
            port = body.get("port")
            server_rank = body.get("server_rank")
            num_servers = body.get("num_inference_server")
            timeout = body.get("timeout")
            self.state.nccl_initialized = True
            logger.info(
                "NCCL init: host=%s:%s, rank=%s/%s, timeout=%ss",
                host, port, server_rank, num_servers, timeout,
            )
            logger.warning("NCCL broadcaster initialization — security-relevant endpoint")
            self._send_json_response({"status": "success"})

        elif self.path == "/v1/chat/completions":
            self._log_request("/v1/chat/completions", {
                "model": body.get("model"),
                "n": body.get("n", 1),
                "temperature": body.get("temperature", 1.0),
                "num_messages": len(body.get("messages", [])),
            })
            self._handle_chat_completions(body)

        else:
            self.send_error(404, f"Not found: {self.path}")

    def _handle_chat_completions(self, body: dict) -> None:
        model = body.get("model", self.state.current_model)
        n = body.get("n", 1)
        logprobs = body.get("logprobs", False)

        choices = []
        for i in range(n):
            content = (
                f"<think>Let me analyze this step by step.</think>\n"
                f"Based on my analysis, the answer is {42 + i}."
            )
            choice: dict = {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            if logprobs:
                choice["logprobs"] = {"content": []}
            choices.append(choice)

        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50 * n,
                "total_tokens": 100 + 50 * n,
            },
        }
        self._send_json_response(response)


class MockInferenceServer:
    """Mock inference server that can be started/stopped."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        model_name: str = "Qwen/Qwen3-0.6B",
    ):
        self.host = host
        self.port = port
        self.state = ServerState(current_model=model_name)
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.server = HTTPServer((self.host, self.port), MockInferenceHandler)
        self.server.state = self.state
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info("Mock inference server started on http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            logger.info("Mock inference server stopped")

    def get_request_log(self) -> list[dict]:
        return self.state.request_log

    def save_request_log(self, path: Path) -> None:
        with open(path, "w") as f:
            for entry in self.state.request_log:
                f.write(json.dumps(entry) + "\n")
        logger.info("Request log saved to %s", path)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the mock inference server (blocking)."""
    state = ServerState()
    server = HTTPServer((host, port), MockInferenceHandler)
    server.state = state
    logger.info("Starting mock inference server on http://%s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
