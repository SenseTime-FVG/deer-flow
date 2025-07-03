"""
LLM Sandbox MCP Adapter

This module provides an MCP (Model Context Protocol) adapter for the LLM Sandbox SDK.
It wraps the AsyncSandboxClient and exposes sandbox functionality as MCP tools.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from mcp.server.fastmcp import FastMCP

# Configure logging
logger = logging.getLogger(__name__)


# Import the LLM Sandbox SDK components dynamically
def _import_sandbox_sdk():
    """Dynamically import sandbox SDK components"""
    try:
        # Try relative imports first (when used as a module)
        try:
            from .llm_sandbox_sdk.async_client import AsyncSandboxClient
            from .llm_sandbox_sdk.models import (
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
            from .llm_sandbox_sdk.exceptions import (
                SandboxClientError,
                SessionNotFoundError,
                ServerError,
                NetworkError,
                ValidationError,
                ResourceLimitError,
            )
        except ImportError:
            # Fallback to absolute imports (when running as script)
            import sys
            import os

            sys.path.insert(0, os.path.dirname(__file__))

            from llm_sandbox_sdk.async_client import AsyncSandboxClient
            from llm_sandbox_sdk.models import (
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
            from llm_sandbox_sdk.exceptions import (
                SandboxClientError,
                SessionNotFoundError,
                ServerError,
                NetworkError,
                ValidationError,
                ResourceLimitError,
            )

        return {
            "AsyncSandboxClient": AsyncSandboxClient,
            "SessionInfo": SessionInfo,
            "CodeResult": CodeResult,
            "ArtifactResult": ArtifactResult,
            "CommandResult": CommandResult,
            "LibraryInstallResult": LibraryInstallResult,
            "FileUploadResult": FileUploadResult,
            "FileDownloadResult": FileDownloadResult,
            "SessionList": SessionList,
            "FileContent": FileContent,
            "SandboxClientError": SandboxClientError,
            "SessionNotFoundError": SessionNotFoundError,
            "ServerError": ServerError,
            "NetworkError": NetworkError,
            "ValidationError": ValidationError,
            "ResourceLimitError": ResourceLimitError,
        }
    except ImportError as e:
        logger.error(f"Failed to import sandbox SDK: {e}")
        raise RuntimeError(f"Cannot import sandbox SDK components: {e}")


# Import SDK components
SDK = None


def _ensure_sdk():
    """Ensure SDK is loaded"""
    global SDK
    if SDK is None:
        try:
            SDK = _import_sandbox_sdk()
        except Exception as e:
            raise RuntimeError(f"Failed to import sandbox SDK: {e}")
    return SDK


class SandboxMCPAdapter:
    """MCP Adapter for LLM Sandbox SDK"""

    def __init__(self, sandbox_base_url: str, timeout: float = 30.0):
        self.sandbox_base_url = sandbox_base_url
        self.timeout = timeout
        self._client = None

    async def _get_client(self):
        """Get or create the sandbox client"""
        if self._client is None:
            sdk = _ensure_sdk()
            self._client = sdk["AsyncSandboxClient"](
                self.sandbox_base_url, self.timeout
            )
        return self._client

    async def _handle_sandbox_error(
        self, e: Exception, operation: str
    ) -> Dict[str, Any]:
        """Handle sandbox errors and return standardized error response"""
        sdk = _ensure_sdk()

        if isinstance(e, sdk["SessionNotFoundError"]):
            error_response = {
                "success": False,
                "error_type": "session_not_found",
                "error_message": f"Session not found during {operation}",
                "detail": str(e),
            }
        elif isinstance(e, sdk["NetworkError"]):
            error_response = {
                "success": False,
                "error_type": "network_error",
                "error_message": f"Network error during {operation}",
                "detail": str(e),
            }
        elif isinstance(e, sdk["ServerError"]):
            error_response = {
                "success": False,
                "error_type": "server_error",
                "error_message": f"Server error during {operation}",
                "detail": str(e),
                "status_code": getattr(e, "status_code", None),
            }
        elif isinstance(e, sdk["ValidationError"]):
            error_response = {
                "success": False,
                "error_type": "validation_error",
                "error_message": f"Validation error during {operation}",
                "detail": str(e),
            }
        elif isinstance(e, sdk["ResourceLimitError"]):
            error_response = {
                "success": False,
                "error_type": "resource_limit_error",
                "error_message": f"Resource limit exceeded during {operation}",
                "detail": str(e),
            }
        elif isinstance(e, sdk["SandboxClientError"]):
            error_response = {
                "success": False,
                "error_type": "sandbox_client_error",
                "error_message": f"Sandbox client error during {operation}",
                "detail": str(e),
            }
        else:
            error_response = {
                "success": False,
                "error_type": "unknown_error",
                "error_message": f"Unknown error during {operation}",
                "detail": str(e),
            }

        logger.error(f"Sandbox error during {operation}: {error_response}")
        return error_response


# Global adapter instance
_adapter: Optional[SandboxMCPAdapter] = None


def get_adapter() -> SandboxMCPAdapter:
    """Get the global adapter instance"""
    if _adapter is None:
        raise RuntimeError("Adapter not initialized. Call setup_adapter() first.")
    return _adapter


def setup_adapter(sandbox_base_url: str, timeout: float = 30.0):
    """Setup the global adapter instance"""
    global _adapter
    _adapter = SandboxMCPAdapter(sandbox_base_url, timeout)


# MCP Tool Functions


async def health_check() -> str:
    """
    Check if the sandbox service is healthy and available.

    Returns:
        JSON string with health status
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        is_healthy = await client.health_check()

        result = {
            "success": True,
            "healthy": is_healthy,
            "message": (
                "Sandbox service is healthy"
                if is_healthy
                else "Sandbox service is not available"
            ),
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "health_check")
        return json.dumps(error_result, ensure_ascii=False)


async def create_session(
    language: str = "python",
    timeout: int = 300,
    cpu_mem_limit_mb: int = 256,
    cpu_core_limit: int = 1,
) -> str:
    """
    Create a new sandbox session.

    Args:
        language: Programming language (default: python)
        timeout: Session timeout in seconds (default: 300)
        cpu_mem_limit_mb: Memory limit in MB (default: 256)
        cpu_core_limit: CPU core limit (default: 1)

    Returns:
        JSON string with session information
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        session = await client.create_session(
            language=language,
            timeout=timeout,
            cpu_mem_limit_mb=cpu_mem_limit_mb,
            cpu_core_limit=cpu_core_limit,
        )

        result = {
            "success": True,
            "session": {
                "session_id": session.session_id,
                "status": session.status,
                "created_at": session.created_at,
            },
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "create_session")
        return json.dumps(error_result, ensure_ascii=False)


async def get_session(session_id: str) -> str:
    """
    Get information about a specific session.

    Args:
        session_id: The session ID to query

    Returns:
        JSON string with session information
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        session = await client.get_session(session_id)

        result = {
            "success": True,
            "session": {
                "session_id": session.session_id,
                "status": session.status,
                "created_at": session.created_at,
            },
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "get_session")
        return json.dumps(error_result, ensure_ascii=False)


async def delete_session(session_id: str) -> str:
    """
    Delete a sandbox session.

    Args:
        session_id: The session ID to delete

    Returns:
        JSON string with deletion status
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        success = await client.delete_session(session_id)

        result = {"success": True, "deleted": success, "session_id": session_id}
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "delete_session")
        return json.dumps(error_result, ensure_ascii=False)


async def list_sessions(status_filter: Optional[str] = None) -> str:
    """
    List all sandbox sessions.

    Args:
        status_filter: Filter sessions by status (optional)

    Returns:
        JSON string with sessions list
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        session_list = await client.list_sessions(status_filter)

        result = {
            "success": True,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "status": session.status,
                    "created_at": session.created_at,
                }
                for session in session_list.sessions
            ],
            "total_count": session_list.total_count,
            "active_count": session_list.active_count,
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "list_sessions")
        return json.dumps(error_result, ensure_ascii=False)


async def run_code(
    session_id: str, code: str, libraries: Optional[List[str]] = None
) -> str:
    """
    Run Python code in a sandbox session.

    Args:
        session_id: The session ID to run code in
        code: The Python code to execute
        libraries: Optional list of libraries to install before execution

    Returns:
        JSON string with execution results
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        result = await client.run_code(session_id, code, libraries)

        response = {
            "success": True,
            "result": {
                "session_id": result.session_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "run_time": result.run_time,
            },
        }
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "run_code")
        return json.dumps(error_result, ensure_ascii=False)


async def run_code_with_artifacts(
    session_id: str, code: str, libraries: Optional[List[str]] = None
) -> str:
    """
    Run Python code and capture artifacts (plots, images, etc.).

    Args:
        session_id: The session ID to run code in
        code: The Python code to execute
        libraries: Optional list of libraries to install before execution

    Returns:
        JSON string with execution results and artifacts
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        result = await client.run_code_with_artifacts(session_id, code, libraries)

        response = {
            "success": True,
            "result": {
                "session_id": result.session_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "run_time": result.run_time,
                "artifacts": result.files,
            },
        }
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "run_code_with_artifacts")
        return json.dumps(error_result, ensure_ascii=False)


async def execute_command(session_id: str, command: str) -> str:
    """
    Execute a shell command in a sandbox session.

    Args:
        session_id: The session ID to execute command in
        command: The shell command to execute

    Returns:
        JSON string with command execution results
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        result = await client.execute_command(session_id, command)

        response = {
            "success": True,
            "result": {
                "session_id": result.session_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "run_time": result.run_time,
            },
        }
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "execute_command")
        return json.dumps(error_result, ensure_ascii=False)


async def install_libraries(session_id: str, libraries: List[str]) -> str:
    """
    Install Python libraries in a sandbox session.

    Args:
        session_id: The session ID to install libraries in
        libraries: List of library names to install

    Returns:
        JSON string with installation results
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        result = await client.install_libraries(session_id, libraries)

        response = {
            "success": True,
            "result": {
                "session_id": result.session_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "run_time": result.run_time,
                "libraries": libraries,
            },
        }
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "install_libraries")
        return json.dumps(error_result, ensure_ascii=False)


async def upload_files(session_id: str, files: List[Dict[str, str]]) -> str:
    """
    Upload files to a sandbox session.

    Args:
        session_id: The session ID to upload files to
        files: List of file dictionaries with 'path' and 'content_base64' keys

    Returns:
        JSON string with upload results
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()

        # Convert file dictionaries to FileContent objects
        file_contents = []
        for file_dict in files:
            sdk = _ensure_sdk()
            file_content = sdk["FileContent"].from_base64(
                path=file_dict["path"], content_base64=file_dict["content_base64"]
            )
            file_contents.append(file_content)

        result = await client.upload_files(session_id, file_contents)

        response = {
            "success": True,
            "result": {
                "session_id": result.session_id,
                "uploaded_files": result.uploaded_files,
                "failed_files": result.failed_files,
                "upload_results": result.upload_results,
            },
        }
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "upload_files")
        return json.dumps(error_result, ensure_ascii=False)


async def download_files(session_id: str, file_paths: List[str]) -> str:
    """
    Download files from a sandbox session.

    Args:
        session_id: The session ID to download files from
        file_paths: List of file paths to download

    Returns:
        JSON string with download results
    """
    adapter = get_adapter()
    try:
        client = await adapter._get_client()
        result = await client.download_files(session_id, file_paths)

        # Convert FileContent objects to dictionaries
        files = []
        for file_content in result.files:
            files.append(
                {"path": file_content.path, "content_base64": file_content.to_base64()}
            )

        response = {
            "success": True,
            "result": {
                "session_id": result.session_id,
                "files": files,
                "download_results": result.download_results,
            },
        }
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        error_result = await adapter._handle_sandbox_error(e, "download_files")
        return json.dumps(error_result, ensure_ascii=False)


# Main entry point
def main():
    """Main entry point for the LLM Sandbox MCP server"""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM Sandbox MCP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8016, help="Port to listen on")
    parser.add_argument(
        "--sandbox_host",
        type=str,
        default="localhost",
        help="Sandbox service host (default: localhost)",
    )
    parser.add_argument(
        "--sandbox_port",
        type=str,
        default="8000",
        help="Sandbox service port (default: 8000)",
    )
    parser.add_argument(
        "--sandbox_timeout",
        type=float,
        default=30.0,
        help="Sandbox request timeout in seconds (default: 30.0)",
    )
    args = parser.parse_args()

    # Setup the sandbox endpoint
    sandbox_base_url = f"http://{args.sandbox_host}:{args.sandbox_port}"
    setup_adapter(sandbox_base_url, args.sandbox_timeout)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting LLM Sandbox MCP server on {args.host}:{args.port}")
    logger.info(f"Connecting to sandbox service at {sandbox_base_url}")

    # Setup FastMCP server
    settings = dict(host=args.host, port=args.port)
    mcp = FastMCP("LLM_Sandbox", **settings)

    # Add all the sandbox tools
    mcp.add_tool(health_check)
    mcp.add_tool(create_session)
    mcp.add_tool(get_session)
    mcp.add_tool(delete_session)
    mcp.add_tool(list_sessions)
    mcp.add_tool(run_code)
    mcp.add_tool(run_code_with_artifacts)
    mcp.add_tool(execute_command)
    mcp.add_tool(install_libraries)
    mcp.add_tool(upload_files)
    mcp.add_tool(download_files)

    # Start the server
    try:
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
