"""
Synchronous client for the LLM Sandbox API

This client provides synchronous methods for interacting with the sandbox API
and supports context manager for proper resource cleanup.
"""

import json
import logging
from typing import List, Optional
import requests
from contextlib import contextmanager

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
    SessionCreateRequest,
)


logger = logging.getLogger(__name__)


class SandboxClient(BaseClient):
    """Synchronous client for LLM Sandbox API"""

    def __init__(self, base_url: str, timeout: float = 30.0):
        super().__init__(base_url, timeout)
        self._session = requests.Session()
        self._session.headers.update(self._headers)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def close(self):
        """Close the HTTP session"""
        if self._session:
            self._session.close()

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an HTTP request"""
        url = f"{self.base_url}{endpoint}"

        # Set timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        try:
            response = self._session.request(method, url, **kwargs)

            if response.status_code >= 400:
                self._handle_error_response(response.status_code, response.text)

            if response.text:
                return response.json()
            return {}

        except requests.RequestException as e:
            raise NetworkError(f"Network error: {e}")
        except json.JSONDecodeError as e:
            raise SandboxClientError(f"Invalid JSON response: {e}")

    def health_check(self) -> bool:
        """Check if the server is healthy"""
        try:
            self._request("GET", "/v1/ping")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def create_session(
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

        data = self._request("POST", "/sessions/create_session", json=payload)

        return self._parse_session_info(data)

    def get_session(self, session_id: str) -> SessionInfo:
        """Get information about a specific session"""
        data = self._request("GET", f"/sessions/{session_id}/get_session_status")

        return self._parse_session_info(data)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        data = self._request("DELETE", f"/sessions/{session_id}/delete_session")

        return data.get("delete_status") == "success"

    def list_sessions(self, status_filter: Optional[str] = None) -> SessionList:
        """List all sessions"""
        params = {}
        if status_filter:
            params["status_filter"] = status_filter

        data = self._request("GET", "/sessions/list_sessions", params=params)

        return self._parse_session_list(data)

    def run_code(
        self, session_id: str, code: str, libraries: Optional[List[str]] = None
    ) -> CodeResult:
        """Run code in a session"""
        payload = {"session_id": session_id, "code": code, "libraries": libraries or []}

        data = self._request("POST", f"/sessions/{session_id}/run_code", json=payload)

        return self._parse_code_result(data)

    def run_code_with_artifacts(
        self, session_id: str, code: str, libraries: Optional[List[str]] = None
    ) -> ArtifactResult:
        """Run code and capture artifacts"""
        payload = {"session_id": session_id, "code": code, "libraries": libraries or []}

        data = self._request(
            "POST", f"/sessions/{session_id}/run_code_with_artifacts", json=payload
        )

        return self._parse_artifact_result(data)

    def execute_command(self, session_id: str, command: str) -> CommandResult:
        """Execute a shell command in a session"""
        payload = {"session_id": session_id, "command": command}

        data = self._request(
            "POST", f"/sessions/{session_id}/execute_command", json=payload
        )

        return self._parse_command_result(data)

    def install_libraries(
        self, session_id: str, libraries: List[str]
    ) -> LibraryInstallResult:
        """Install libraries in a session"""
        payload = {"session_id": session_id, "libraries": libraries}

        data = self._request(
            "POST", f"/sessions/{session_id}/install_libraries", json=payload
        )

        return self._parse_library_result(data)

    def upload_files(
        self, session_id: str, files: List[FileContent]
    ) -> FileUploadResult:
        """Upload files to a session"""
        file_data = []
        for file_content in files:
            file_data.append(
                {"content_base64": file_content.to_base64(), "dest": file_content.path}
            )

        payload = {"session_id": session_id, "files": file_data}

        data = self._request(
            "POST", f"/sessions/{session_id}/upload_files", json=payload
        )

        return self._parse_upload_result(data)

    def download_files(
        self, session_id: str, file_paths: List[str]
    ) -> FileDownloadResult:
        """Download files from a session"""
        payload = {"session_id": session_id, "files": file_paths}

        data = self._request(
            "GET", f"/sessions/{session_id}/download_files", json=payload
        )

        return self._parse_download_result(data)

    def list_files(self, session_id: str, root: str = "/sandbox") -> str:
        """List files in a session"""
        payload = {"session_id": session_id, "root": root}

        data = self._request(
            "GET", f"/sessions/{session_id}/list_all_files", json=payload
        )

        return data.get("listed_file_paths", "")


# Convenience context manager
@contextmanager
def sandbox_session(
    base_url: str,
    language: str = "python",
    timeout: int = 300,
    cpu_mem_limit_mb: int = 256,
    cpu_core_limit: int = 1,
):
    """
    Context manager that creates a client and session, then cleans up both

    Usage:
        with sandbox_session("http://localhost:8000") as (client, session):
            result = client.run_code(session.session_id, "print('Hello')")
    """
    with SandboxClient(base_url) as client:
        session = client.create_session(
            language=language,
            timeout=timeout,
            cpu_mem_limit_mb=cpu_mem_limit_mb,
            cpu_core_limit=cpu_core_limit,
        )
        try:
            yield client, session
        finally:
            try:
                client.delete_session(session.session_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session.session_id}: {e}")
