"""
LLM Sandbox Client SDK

This package provides both synchronous and asynchronous client SDKs for interacting
with the LLM Sandbox API server.

Example usage:

    # Async context manager
    async with AsyncSandboxClient("http://localhost:8000") as client:
        session = await client.create_session()
        result = await client.run_code(session.session_id, "print('Hello World')")

    # Sync context manager
    with SandboxClient("http://localhost:8000") as client:
        session = client.create_session()
        result = client.run_code(session.session_id, "print('Hello World')")
"""

from .async_client import *
from .sync_client import *
from .models import *
from .exceptions import *

__all__ = [
    "AsyncSandboxClient",
    "SandboxClient",
    # Exception classes
    "SandboxClientError",
    "SessionNotFoundError",
    "ServerError",
    "NetworkError",
    "ValidationError",
    # Model classes
    "SessionInfo",
    "CodeResult",
    "ArtifactResult",
    "CommandResult",
    "FileUploadResult",
    "FileDownloadResult",
    "LibraryInstallResult",
    # async context manager
    "async_sandbox_session",
    # sync context manager
    "sandbox_session",
]
