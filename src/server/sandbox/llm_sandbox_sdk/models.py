"""
Data models for the LLM Sandbox Client SDK
"""

from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime
import base64


@dataclass
class SessionInfo:
    """Information about a sandbox session"""

    session_id: str
    status: str
    created_at: str

    @property
    def created_datetime(self) -> datetime:
        """Parse created_at string into datetime object"""
        return datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))


@dataclass
class CodeResult:
    """Result of code execution"""

    session_id: str
    stdout: str
    stderr: str
    return_code: int
    run_time: float


@dataclass
class ArtifactResult:
    """Result of code execution with artifacts"""

    session_id: str
    stdout: str
    stderr: str
    return_code: int
    files: List[str]
    run_time: float


@dataclass
class CommandResult:
    """Result of command execution"""

    session_id: str
    stdout: str
    stderr: str
    return_code: int
    run_time: float


@dataclass
class LibraryInstallResult:
    """Result of library installation"""

    session_id: str
    stdout: str
    stderr: str
    return_code: int
    run_time: float


@dataclass
class FileContent:
    """File content for upload/download operations"""

    path: str
    content: bytes

    def to_base64(self) -> str:
        """Convert content to base64 string"""
        return base64.b64encode(self.content).decode("utf-8")

    @classmethod
    def from_base64(cls, path: str, content_base64: str) -> "FileContent":
        """Create FileContent from base64 string"""
        content = base64.b64decode(content_base64)
        return cls(path=path, content=content)


@dataclass
class FileUploadResult:
    """Result of file upload operation"""

    session_id: str
    uploaded_files: List[str]
    failed_files: List[str]
    upload_results: List[Dict[str, Any]]


@dataclass
class FileDownloadResult:
    """Result of file download operation"""

    session_id: str
    files: List[FileContent]
    download_results: List[Dict[str, Any]]


@dataclass
class SessionList:
    """List of sessions"""

    sessions: List[SessionInfo]
    total_count: int
    active_count: int


@dataclass
class SessionCreateRequest:
    """Request to create a new session"""

    language: str = "python"
    timeout: int = 300
    cpu_mem_limit_mb: int = 256
    cpu_core_limit: int = 1
