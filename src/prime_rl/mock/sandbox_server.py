"""
Mock Sandbox HTTP Server

This server mimics PrimeIntellect's Remote Sandbox API.
The REAL RemoteSandbox class will make HTTP requests HERE.

This gives you:
- Real HTTP communication channel to analyze
- Request/response logging for security research
- Ability to inject attacks (malformed responses, delays, etc.)
"""

import json
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class RequestLog:
    """Log entry for each API request - for security analysis."""
    timestamp: str
    method: str
    path: str
    headers: dict
    body: dict | str | None
    client_ip: str
    response_code: int
    response_body: dict | None = None


class MockSandboxServer:
    """
    Mock server that mimics PrimeIntellect's Sandbox API.
    
    The real RemoteSandbox makes HTTP POST requests to execute code.
    This server receives those requests and:
    1. Logs everything for security analysis
    2. Returns fake (but valid) responses
    3. Optionally executes code locally for realistic behavior
    """
    
    def __init__(
        self, 
        host: str = "0.0.0.0",
        port: int = 8000,
        log_file: str = "sandbox_requests.jsonl",
        execute_locally: bool = False  # Set True to actually run code
    ):
        self.host = host
        self.port = port
        self.log_file = log_file
        self.execute_locally = execute_locally
        self.request_logs: list[RequestLog] = []
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes to mimic PrimeIntellect's sandbox API."""
        
        # Main code execution endpoint
        self.app.router.add_post('/v1/sandbox/run', self.handle_run)
        self.app.router.add_post('/v1/execute', self.handle_run)  # Alternative endpoint
        self.app.router.add_post('/run', self.handle_run)  # Simple endpoint
        
        # Health check
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/v1/health', self.handle_health)
        
        # Catch-all for discovering undocumented endpoints
        self.app.router.add_route('*', '/{path:.*}', self.handle_catchall)
    
    async def handle_run(self, request: web.Request) -> web.Response:
        """
        Handle code execution requests.
        
        This is where RemoteSandbox sends code to be executed.
        """
        # Log the raw request for security analysis
        body_raw = await request.read()
        try:
            body = json.loads(body_raw) if body_raw else {}
        except json.JSONDecodeError:
            body = body_raw.decode('utf-8', errors='replace')
        
        log_entry = RequestLog(
            timestamp=datetime.now().isoformat(),
            method=request.method,
            path=str(request.path),
            headers=dict(request.headers),
            body=body,
            client_ip=request.remote or "unknown",
            response_code=200
        )
        
        # Security analysis logging
        logger.info("=" * 60)
        logger.info("SANDBOX EXECUTION REQUEST")
        logger.info("=" * 60)
        logger.info("Client IP: %s", log_entry.client_ip)
        logger.info("Path: %s", log_entry.path)
        logger.info("Headers: %s", json.dumps(log_entry.headers, indent=2))

        auth_header = request.headers.get('Authorization', '')
        api_key = request.headers.get('X-API-Key', '')
        logger.info("Auth Header: %s", f"{auth_header[:20]}..." if auth_header else "MISSING")
        logger.info("API Key: %s", f"{api_key[:10]}..." if api_key else "MISSING")

        if isinstance(body, dict):
            code = body.get('code', body.get('source', 'NO CODE FOUND'))
            language = body.get('language', 'python')
            timeout = body.get('timeout', 30)
            logger.info("Language: %s", language)
            logger.info("Timeout: %ss", timeout)
            code_str = str(code)
            logger.info("Code:\n%s", code_str[:500] + "..." if len(code_str) > 500 else code_str)
        
        # Generate response
        if self.execute_locally and isinstance(body, dict):
            # Actually execute the code (optional)
            result = await self._execute_code(body.get('code', ''))
        else:
            # Fake successful response
            result = {
                "stdout": "Mock execution successful\n",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
                "execution_time_ms": 42
            }
        
        log_entry.response_code = 200
        log_entry.response_body = result
        self.request_logs.append(log_entry)
        self._write_log(log_entry)
        
        logger.info("Response: %s", json.dumps(result, indent=2))
        logger.info("=" * 60)
        
        return web.json_response(result)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        logger.debug("Health check from %s", request.remote)
        return web.json_response({"status": "healthy", "mock": True})
    
    async def handle_catchall(self, request: web.Request) -> web.Response:
        """
        Catch-all handler for discovering undocumented API endpoints.
        
        This is useful for security analysis - you'll see what endpoints
        the real code tries to hit.
        """
        body_raw = await request.read()
        
        logger.warning("=" * 60)
        logger.warning("UNKNOWN ENDPOINT HIT")
        logger.warning("=" * 60)
        logger.warning("Method: %s", request.method)
        logger.warning("Path: %s", request.path)
        logger.warning("Headers: %s", dict(request.headers))
        logger.warning("Body: %s", body_raw.decode('utf-8', errors='replace')[:1000])
        logger.warning("=" * 60)
        
        log_entry = RequestLog(
            timestamp=datetime.now().isoformat(),
            method=request.method,
            path=str(request.path),
            headers=dict(request.headers),
            body=body_raw.decode('utf-8', errors='replace'),
            client_ip=request.remote or "unknown",
            response_code=404
        )
        self.request_logs.append(log_entry)
        self._write_log(log_entry)
        
        return web.json_response(
            {"error": "Not found", "path": request.path},
            status=404
        )
    
    async def _execute_code(self, code: str) -> dict:
        """Actually execute code locally (optional)."""
        import subprocess
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            result = subprocess.run(
                ["python", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "timed_out": False
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Execution timed out",
                "exit_code": 124,
                "timed_out": True
            }
        finally:
            temp_path.unlink(missing_ok=True)
    
    def _write_log(self, entry: RequestLog):
        """Write log entry to JSONL file for later analysis."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
    
    def get_security_report(self) -> dict:
        """Generate a security analysis report from collected logs."""
        return {
            "total_requests": len(self.request_logs),
            "endpoints_hit": list(set(r.path for r in self.request_logs)),
            "unique_clients": list(set(r.client_ip for r in self.request_logs)),
            "auth_headers_seen": [
                r.headers.get('Authorization', 'NONE')[:20] 
                for r in self.request_logs
            ],
            "requests": [asdict(r) for r in self.request_logs]
        }
    
    async def start(self):
        """Start the mock server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info("Mock Sandbox Server running on http://%s:%d", self.host, self.port)
        logger.info("Logs will be written to: %s", self.log_file)
        return runner
    
    def run(self):
        """Run server (blocking)."""
        web.run_app(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("MOCK SANDBOX SERVER - Security Analysis Mode")
    logger.info("=" * 60)
    logger.info("This server mimics PrimeIntellect's Sandbox API.")
    logger.info("Point RemoteSandbox to http://localhost:8000")
    logger.info("All requests will be logged for security analysis.")
    logger.info("=" * 60)

    server = MockSandboxServer(
        port=8000,
        log_file="results/sandbox_requests.jsonl",
        execute_locally=False,
    )
    server.run()