"""
Exception classes for the LLM Sandbox Client SDK
"""


class SandboxClientError(Exception):
    """Base exception for all sandbox client errors"""

    pass


class SessionNotFoundError(SandboxClientError):
    """Raised when a session is not found on the server"""

    pass


class ServerError(SandboxClientError):
    """Raised when the server returns an error response"""

    def __init__(self, message: str, status_code: int = None, detail: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class NetworkError(SandboxClientError):
    """Raised when there's a network connection issue"""

    pass


class ValidationError(SandboxClientError):
    """Raised when request validation fails"""

    pass


class TimeoutError(SandboxClientError):
    """Raised when an operation times out"""

    pass


class ResourceLimitError(SandboxClientError):
    """Raised when resource limits are exceeded"""

    pass
