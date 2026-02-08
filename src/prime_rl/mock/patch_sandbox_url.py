"""
Fallback: Patch the sandbox URL directly in Python memory.

Use this if environment variables don't work.
"""

import logging
from contextlib import ExitStack
from unittest.mock import patch

logger = logging.getLogger(__name__)

_PATCH_TARGETS = [
    "verifiers.sandbox.SANDBOX_API_URL",
    "verifiers.sandbox.BASE_URL",
    "verifiers.sandbox.RemoteSandbox.base_url",
    "prime_sandboxes.SANDBOX_API_URL",
    "prime_sandboxes.sandbox.SANDBOX_API_URL",
]


def patch_sandbox_url(target_url: str = "http://localhost:8000"):
    """Context manager to force sandbox URL across known module locations.

    Usage:
        with patch_sandbox_url():
            env = load_environment("math-python")
            await env.verify(...)
    """
    patches = [patch(target, target_url) for target in _PATCH_TARGETS]

    class PatchContext:
        def __enter__(self):
            self.stack = ExitStack()
            for p in patches:
                try:
                    self.stack.enter_context(p)
                except (AttributeError, ModuleNotFoundError):
                    logger.debug("Patch target not found, skipping: %s", p)
            return self

        def __exit__(self, *args):
            self.stack.close()

    return PatchContext()