"""
Base client class with common functionality for both sync and async clients
"""

import json
import logging
from typing import Dict, Any, Optional, List
from .exceptions import (
    SandboxClientError,
    ServerError,
    NetworkError,
    SessionNotFoundError,
)
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


class BaseClient:
    """Base client with common functionality"""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_error_response(self, status_code: int, response_text: str):
        """Handle error responses from the server"""
        try:
            error_data = json.loads(response_text)
            detail = error_data.get("detail", "Unknown error")
        except json.JSONDecodeError:
            detail = response_text

        if status_code == 404:
            if "session" in detail.lower():
                raise SessionNotFoundError(detail)

        raise ServerError(
            message=f"Server error: {status_code}",
            status_code=status_code,
            detail=detail,
        )

    def _parse_session_info(self, data: Dict[str, Any]) -> SessionInfo:
        """Parse session info from API response"""
        return SessionInfo(
            session_id=data["session_id"],
            status=data["status"],
            created_at=data["created_at"],
        )

    def _parse_code_result(self, data: Dict[str, Any]) -> CodeResult:
        """Parse code execution result from API response"""
        output = data["output"]
        return CodeResult(
            session_id=data["session_id"],
            stdout=output["stdout"],
            stderr=output["stderr"],
            return_code=output["exit_code"],
            run_time=data["run_time"],
        )

    def _parse_artifact_result(self, data: Dict[str, Any]) -> ArtifactResult:
        """Parse artifact execution result from API response"""
        output = data["output"]
        return ArtifactResult(
            session_id=data["session_id"],
            stdout=output["stdout"],
            stderr=output["stderr"],
            return_code=output["exit_code"],
            files=output.get("plots", []),
            run_time=data["run_time"],
        )

    def _parse_command_result(self, data: Dict[str, Any]) -> CommandResult:
        """Parse command execution result from API response"""
        output = data["output"]
        return CommandResult(
            session_id=data["session_id"],
            stdout=output["stdout"],
            stderr=output["stderr"],
            return_code=output["exit_code"],
            run_time=data["run_time"],
        )

    def _parse_library_result(self, data: Dict[str, Any]) -> LibraryInstallResult:
        """Parse library installation result from API response"""
        output = data["output"]
        return LibraryInstallResult(
            session_id=data["session_id"],
            stdout=output["stdout"],
            stderr=output["stderr"],
            return_code=output["exit_code"],
            run_time=data["run_time"],
        )

    def _parse_upload_result(self, data: Dict[str, Any]) -> FileUploadResult:
        """Parse file upload result from API response"""
        upload_results = data.get("upload_results", [])
        uploaded_files = []
        failed_files = []

        for result in upload_results:
            if result.get("exit_code", 0) == 0:
                # Success - would need to track which file this corresponds to
                pass
            else:
                # Failed - would need to track which file this corresponds to
                pass

        return FileUploadResult(
            session_id=data["session_id"],
            uploaded_files=uploaded_files,
            failed_files=failed_files,
            upload_results=upload_results,
        )

    def _parse_download_result(self, data: Dict[str, Any]) -> FileDownloadResult:
        """Parse file download result from API response"""
        downloaded_files = data.get("downloaded_files", [])
        files = []

        for file_data in downloaded_files:
            file_content = FileContent.from_base64(
                path=file_data["dest"], content_base64=file_data["content_base64"]
            )
            files.append(file_content)

        return FileDownloadResult(
            session_id=data["session_id"],
            files=files,
            download_results=data.get("download_results", []),
        )

    def _parse_session_list(self, data: Dict[str, Any]) -> SessionList:
        """Parse session list from API response"""
        sessions = [
            self._parse_session_info(session_data) for session_data in data["sessions"]
        ]

        return SessionList(
            sessions=sessions,
            total_count=data["total_count"],
            active_count=data["active_count"],
        )
