"""
Asynchronous client for the LLM Sandbox API

This client provides async/await methods for interacting with the sandbox API
and supports async context manager for proper resource cleanup.
"""

import asyncio
import json
import logging
from typing import List, Optional, Union
import aiohttp
from contextlib import asynccontextmanager

from .base_client import BaseClient
from .exceptions import NetworkError, SandboxClientError
from .models import (
    SessionInfo,
    CodeResult,
    ArtifactResult,
    CommandResult,
    LibraryInstallResult,
    FileUploadResult,
    FileDownloadResult,
    SessionList,
    FileContent,
)


logger = logging.getLogger(__name__)


class AsyncSandboxClient(BaseClient):
    """Asynchronous client for LLM Sandbox API"""

    def __init__(self, base_url: str, timeout: float = 30.0):
        super().__init__(base_url, timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._owned_session = False

    async def __aenter__(self):
        """Async context manager entry"""
        # await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    # async def _ensure_session(self):
    #     """Ensure we have an aiohttp session"""
    #     if self._session is None or self._session.closed:
    #         timeout = aiohttp.ClientTimeout(
    #             total=self.timeout, connect=10, sock_read=30
    #         )
    #         connector = aiohttp.TCPConnector(
    #             limit=100,
    #             limit_per_host=30,
    #             ttl_dns_cache=300,
    #             use_dns_cache=True,
    #             keepalive_timeout=120,
    #             enable_cleanup_closed=True,
    #         )
    #         self._session = aiohttp.ClientSession(
    #             connector=connector,
    #             timeout=timeout,
    #             headers=self._headers,
    #             raise_for_status=False,
    #         )

    #         self._owned_session = True

    async def close(self):
        """Close the HTTP session"""
        if self._session and self._owned_session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._owned_session = False

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an HTTP request"""
        # await self._ensure_session()

        url = f"{self.base_url}{endpoint}"

        try:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.request(method, url, **kwargs) as response:
                    response_text = await response.text()

                    if response.status >= 400:
                        self._handle_error_response(response.status, response_text)

                    if response_text:
                        return json.loads(response_text)
                    return {}

        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error: {e}")
        except json.JSONDecodeError as e:
            raise SandboxClientError(f"Invalid JSON response: {e}")

    async def health_check(self) -> bool:
        """Check if the server is healthy"""
        try:
            await self._request("GET", "/v1/ping")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def create_session(
        self,
        language: str = "python",
        timeout: int = 300,
        cpu_mem_limit_mb: int = 256,
        cpu_core_limit: int = 1,
    ) -> SessionInfo:
        """Create a new sandbox session"""
        payload = {
            "language": language,
            "timeout": timeout,
            "cpu_mem_limit_mb": cpu_mem_limit_mb,
            "cpu_core_limit": cpu_core_limit,
        }

        data = await self._request("POST", "/sessions/create_session", json=payload)

        return self._parse_session_info(data)

    async def get_session(self, session_id: str) -> SessionInfo:
        """Get information about a specific session"""
        data = await self._request("GET", f"/sessions/{session_id}/get_session_status")

        return self._parse_session_info(data)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        data = await self._request("DELETE", f"/sessions/{session_id}/delete_session")

        return data.get("delete_status") == "success"

    async def list_sessions(self, status_filter: Optional[str] = None) -> SessionList:
        """List all sessions"""
        params = {}
        if status_filter:
            params["status_filter"] = status_filter

        data = await self._request("GET", "/sessions/list_sessions", params=params)

        return self._parse_session_list(data)

    async def run_code(
        self, session_id: str, code: str, libraries: Optional[List[str]] = None
    ) -> CodeResult:
        """Run code in a session"""
        payload = {"session_id": session_id, "code": code, "libraries": libraries or []}

        data = await self._request(
            "POST", f"/sessions/{session_id}/run_code", json=payload
        )

        return self._parse_code_result(data)

    async def run_code_with_artifacts(
        self, session_id: str, code: str, libraries: Optional[List[str]] = None
    ) -> ArtifactResult:
        """Run code and capture artifacts"""
        payload = {"session_id": session_id, "code": code, "libraries": libraries or []}

        data = await self._request(
            "POST", f"/sessions/{session_id}/run_code_with_artifacts", json=payload
        )

        return self._parse_artifact_result(data)

    async def execute_command(self, session_id: str, command: str) -> CommandResult:
        """Execute a shell command in a session"""
        payload = {"session_id": session_id, "command": command}

        data = await self._request(
            "POST", f"/sessions/{session_id}/execute_command", json=payload
        )

        return self._parse_command_result(data)

    async def install_libraries(
        self, session_id: str, libraries: List[str]
    ) -> LibraryInstallResult:
        """Install libraries in a session"""
        payload = {"session_id": session_id, "libraries": libraries}

        data = await self._request(
            "POST", f"/sessions/{session_id}/install_libraries", json=payload
        )

        return self._parse_library_result(data)

    async def upload_files(
        self, session_id: str, files: List[FileContent]
    ) -> FileUploadResult:
        """Upload files to a session"""
        file_data = []
        for file_content in files:
            file_data.append(
                {"content_base64": file_content.to_base64(), "dest": file_content.path}
            )

        payload = {"session_id": session_id, "files": file_data}

        data = await self._request(
            "POST", f"/sessions/{session_id}/upload_files", json=payload
        )

        return self._parse_upload_result(data)

    async def download_files(
        self, session_id: str, file_paths: List[str]
    ) -> FileDownloadResult:
        """Download files from a session"""
        payload = {"session_id": session_id, "files": file_paths}

        data = await self._request(
            "GET", f"/sessions/{session_id}/download_files", json=payload
        )

        return self._parse_download_result(data)

    async def list_files(self, session_id: str, root: str = "/sandbox") -> str:
        """List files in a session"""
        payload = {"session_id": session_id, "root": root}

        data = await self._request(
            "GET", f"/sessions/{session_id}/list_all_files", json=payload
        )

        return data.get("listed_file_paths", "")


# Convenience async context manager
@asynccontextmanager
async def async_sandbox_session(
    base_url: str,
    language: str = "python",
    timeout: int = 300,
    cpu_mem_limit_mb: int = 256,
    cpu_core_limit: int = 1,
):
    """
    Async context manager that creates a client and session, then cleans up both

    Usage:
        async with async_sandbox_session("http://localhost:8000") as (client, session):
            result = await client.run_code(session.session_id, "print('Hello')")
    """
    async with AsyncSandboxClient(base_url) as client:
        session = await client.create_session(
            language=language,
            timeout=timeout,
            cpu_mem_limit_mb=cpu_mem_limit_mb,
            cpu_core_limit=cpu_core_limit,
        )
        try:
            yield client, session
        finally:
            try:
                await client.delete_session(session.session_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session.session_id}: {e}")
