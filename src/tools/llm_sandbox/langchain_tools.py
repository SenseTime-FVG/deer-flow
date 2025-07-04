"""
LangChain Tools based on LLM Sandbox SDK

This module provides LangChain-compatible tools that use the SDK client
instead of directly accessing the backend session. This allows for
remote API usage and better separation of concerns.
"""

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from typing import List, Optional, Any, Type, Union
from pydantic import BaseModel, Field, ConfigDict
import json
import base64
import logging
import asyncio
from contextlib import asynccontextmanager

from .sdk.async_client import AsyncSandboxClient
from .sdk.models import SessionInfo, FileContent
from .sdk.exceptions import SandboxClientError, SessionNotFoundError

from abc import ABC
from typing import Tuple

logger = logging.getLogger(__name__)

import sandbox_fusion


# Input schemas for SDK-based tools
class ExecuteCodeSDKInput(BaseModel):
    """Input schema for executing Python code via SDK."""

    code: str = Field(description="Python code to execute in the sandbox")
    libraries: Optional[List[str]] = Field(
        default=None, description="Libraries to install before execution"
    )
    session_id: Optional[str] = Field(
        description="Session ID to use. If not provided, the default session will be used.",
        default=None,
    )


class ExecuteCodeWithArtifactsSDKInput(BaseModel):
    """Input schema for executing Python code with artifacts via SDK"""

    code: str = Field(description="Python code to execute in the sandbox")
    libraries: Optional[List[str]] = Field(
        description="Libraries to install before execution"
    )
    session_id: Optional[str] = Field(
        description="Session ID to use. If not provided, the default session will be used.",
        default=None,
    )


class InstallLibrariesSDKInput(BaseModel):
    """Input schema for installing Python libraries via SDK."""

    libraries: List[str] = Field(description="List of Python library names to install")
    session_id: Optional[str] = Field(
        description="Session ID to use. If not provided, the default session will be used.",
        default=None,
    )


class UploadFileSDKInput(BaseModel):
    """Input schema for uploading files to sandbox via SDK."""

    content: str = Field(description="File content (text or base64 encoded for binary)")
    file_path: str = Field(description="Destination file path in the sandbox")
    encoding: str = Field(
        default="utf-8", description="File encoding (utf-8 or base64 for binary)"
    )
    session_id: Optional[str] = Field(
        description="Session ID to use. If not provided, the default session will be used.",
        default=None,
    )


class DownloadFileSDKInput(BaseModel):
    """Input schema for downloading files from sandbox via SDK."""

    file_paths: List[str] = Field(
        description="List of file paths to download from the sandbox"
    )
    session_id: Optional[str] = Field(
        description="Session ID to use. If not provided, the default session will be used.",
        default=None,
    )


class ListFilesSDKInput(BaseModel):
    """Input schema for listing files in sandbox via SDK."""

    path: Optional[str] = Field(
        default="/sandbox", description="Directory path to list"
    )
    session_id: Optional[str] = Field(
        description="Session ID to use. If not provided, the default session will be used.",
        default=None,
    )


class ExecuteCommandSDKInput(BaseModel):
    """Input schema for executing shell commands via SDK."""

    command: str = Field(description="Shell command to execute")
    session_id: Optional[str] = Field(
        description="Session ID to use. If not provided, the default session will be used.",
        default=None,
    )


class CreateSessionSDKInput(BaseModel):
    """Input schema for creating a new session via SDK."""

    language: str = Field(
        default="python", description="Programming language for the session"
    )
    timeout: int = Field(default=300, description="Session timeout in seconds")
    cpu_mem_limit_mb: int = Field(default=256, description="Memory limit in MB")
    cpu_core_limit: int = Field(default=1, description="CPU core limit")


class DeleteSessionSDKInput(BaseModel):
    """Input schema for deleting a session via SDK."""

    session_id: str = Field(description="Session ID to delete")


class ListSessionsSDKInput(BaseModel):
    """Input schema for listing sessions via SDK."""

    status_filter: Optional[str] = Field(
        default=None, description="Filter sessions by status (optional)"
    )


class LLMSandboxSDKToolkit(BaseToolkit):
    """LangChain toolkit for LLM Sandbox operations using SDK."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: AsyncSandboxClient = Field(description="SDK client for API communication")
    session_id: Optional[str] = Field(
        default=None, description="Default session ID for operations"
    )
    auto_create_session: bool = Field(
        default=True, description="Automatically create a session if none exists"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        session_id: Optional[str] = None,
        auto_create_session: bool = True,
        timeout: float = 300.0,
        **kwargs,
    ):
        client = AsyncSandboxClient(base_url=base_url, timeout=timeout)

        super().__init__(
            client=client,
            session_id=session_id,
            auto_create_session=auto_create_session,
            **kwargs,
        )

        # Note: Auto-creation will be handled in async context
        self._auto_create_session = auto_create_session

    async def ahealth_check(self) -> bool:
        """Check if the sandbox server is connected in asynchrnous mode"""
        return await self.client.health_check()

    async def create_session(self, language: str = "python") -> SessionInfo:
        session_info = await self.client.create_session(language=language)
        self.session_id = session_info.session_id
        return session_info

    def get_session_id(self) -> Optional[str]:
        """Get the current session ID"""
        return self.session_id

    def health_check(self) -> bool:
        return asyncio.run(self.health_check())

    def get_tools(self) -> List[BaseTool]:
        """Return list of tools in this toolkit."""
        return [
            LLMSandboxSDKRunCodeTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKRunCodeWithArtifactsTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKInstallLibraryTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKUploadFilesTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKDownloadFilesTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKListFilesTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKExecuteCommandTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKCreateSessionTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKDeleteSessionTool(
                client=self.client, default_session_id=self.session_id
            ),
            LLMSandboxSDKListSessionsTool(
                client=self.client, default_session_id=self.session_id
            ),
        ]

    async def cleanup(self):
        """Clean up resources"""
        if self.session_id:
            try:
                await self.client.delete_session(self.session_id)
                logger.info(f"Cleaned up session: {self.session_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session {self.session_id}: {e}")

        if self.client:
            await self.client.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


class LLMSandboxSDKTool(BaseTool, ABC):
    """Base class for LLM Sandbox SDK tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: AsyncSandboxClient = Field(description="SDK client for API communication")
    default_session_id: Optional[str] = Field(
        default=None, description="Default session ID to use if none provided in input"
    )

    def __init__(
        self,
        client: AsyncSandboxClient,
        default_session_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(client=client, default_session_id=default_session_id, **kwargs)

    def _get_session_id(self, input_session_id: Optional[str]) -> str:
        """Get session ID from input or default"""
        session_id = input_session_id or self.default_session_id
        if not session_id:
            raise ValueError("No session ID provided and no default session available")
        return session_id

    def _handle_error(self, error: Exception) -> List[Union[bool, dict]]:
        """Handle errors consistently across tools. Returns [has_error, error_dict]"""
        error_dict = {
            "error": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if isinstance(error, SessionNotFoundError):
            error_dict["error_type"] = "session_not_found"
        elif isinstance(error, SandboxClientError):
            error_dict["error_type"] = "client_error"

        return [True, error_dict]


class LLMSandboxSDKRunCodeTool(LLMSandboxSDKTool):
    """Tool for executing Python code using SDK."""

    name: str = "execute_python_code_sdk"
    description: str = """Execute Python code in a sandboxed environment using the SDK client.
    Returns the stdout, stderr, and exit code of the execution.
    Use this tool to run Python scripts, analyze data, or perform computations.
    For convenientce, you could also specify libraries to install before execution.
    
    [IMPORTANT]
    1. Your default workspace is under "/sandbox". You are not permitted to operate content outside this scope.
    2. Write complete codes! The interpreter environment is isolated for each tool call.
    However, files under your workspace will be kept unchanged between calls.
    3. Use "list_sandbox_files_sdk" tool first if you're unsure about file structures.
    4. Install necessary libraries using "install_python_libraries_sdk" tool for import errors.
    5. Use "print" function to output results to stdout.
    """
    args_schema: Type[BaseModel] = ExecuteCodeSDKInput

    def _run(
        self,
        code: str,
        libraries: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Execute Python code using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(code, libraries, session_id, **kwargs))

    async def _arun(
        self,
        code: str,
        libraries: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Execute Python code using SDK. Returns [has_error, result_dict]"""
        try:
            actual_session_id = self._get_session_id(session_id)
            result = await self.client.run_code(actual_session_id, code, libraries)

            result_dict = {
                "session_id": result.session_id,
                "exit_code": result.return_code,
                "run_time": result.run_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": False,
            }

            return [False, result_dict]

        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKRunCodeWithArtifactsTool(LLMSandboxSDKTool):
    """Tool for executing Python code with artifacts using SDK."""

    name: str = "execute_python_code_with_artifacts_sdk"
    description: str = """Execute Python code in a sandboxed environment and capture artifacts using SDK.
    Returns stdout, stderr, exit code and artifacts (like plots) from execution.
    Use this tool to run Python scripts that generate visualizations.
    For convenientce, you could also specify libraries to install before execution.

    
    [IMPORTANT]
    1. Your default workspace is under "/sandbox". You are not permitted to operate content outside this scope.
    2. Write complete codes! The interpreter environment is isolated for each tool call.
    3. Supported visualization libraries include matplotlib, seaborn, plotly, etc.
    4. Use "print" function to output results to stdout.
    5. Artifacts will be captured automatically for supported plotting libraries.
    """
    args_schema: Type[BaseModel] = ExecuteCodeWithArtifactsSDKInput

    def _run(
        self,
        code: str,
        libraries: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Execute Python code with artifacts using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(code, libraries, session_id, **kwargs))

    async def _arun(
        self,
        code: str,
        libraries: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Execute Python code with artifacts using SDK. Returns [has_error, result_dict]"""
        try:
            actual_session_id = self._get_session_id(session_id)
            result = await self.client.run_code_with_artifacts(
                actual_session_id, code, libraries
            )

            result_dict = {
                "session_id": result.session_id,
                "exit_code": result.return_code,
                "run_time": result.run_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "artifacts": result.files,
                "error": False,
            }

            return [False, result_dict]

        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKInstallLibraryTool(LLMSandboxSDKTool):
    """Tool for installing Python libraries using SDK."""

    name: str = "install_python_libraries_sdk"
    description: str = """Install Python libraries in the sandbox environment using SDK.
    Takes a list of library names and installs them using pip.
    Use this tool to install dependencies before running code."""
    args_schema: Type[BaseModel] = InstallLibrariesSDKInput

    def _run(
        self, libraries: List[str], session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """Install Python libraries using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(libraries, session_id, **kwargs))

    async def _arun(
        self, libraries: List[str], session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """Install Python libraries using SDK. Returns [has_error, result_dict]"""
        try:
            actual_session_id = self._get_session_id(session_id)
            result = await self.client.install_libraries(actual_session_id, libraries)

            result_dict = {
                "session_id": result.session_id,
                "libraries": libraries,
                "exit_code": result.return_code,
                "run_time": result.run_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": False,
            }

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKUploadFilesTool(LLMSandboxSDKTool):
    """Tool for uploading files to sandbox using SDK."""

    name: str = "upload_file_to_sandbox_sdk"
    description: str = """Upload a file to the sandbox environment using SDK.
    Supports text files (with utf-8 encoding) and binary files (with base64 encoding).
    Use this tool to provide input files for your code."""
    args_schema: Type[BaseModel] = UploadFileSDKInput

    def _run(
        self,
        content: str,
        file_path: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Upload a file to sandbox using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(content, file_path, session_id, **kwargs))

    async def _arun(
        self,
        content: str,
        file_path: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Upload a file to sandbox using SDK. Returns [has_error, result_dict]"""
        try:
            actual_session_id = self._get_session_id(session_id)
            file_content_bytes = base64.b64decode(content)
            file_content = FileContent(path=file_path, content=file_content_bytes)
            result = await self.client.upload_files(actual_session_id, [file_content])
            result_dict = {
                "session_id": result.session_id,
                "file_path": file_path,
                "upload_results": result.upload_results,
                "error": False,
            }

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKDownloadFilesTool(LLMSandboxSDKTool):
    """Tool for downloading files from sandbox using SDK."""

    name: str = "download_files_from_sandbox_sdk"
    description: str = """Download files from the sandbox environment using SDK.
    Returns file contents as text (for text files) or base64 (for binary files).
    Use this tool to retrieve output files or results from your code."""
    args_schema: Type[BaseModel] = DownloadFileSDKInput

    def _run(
        self, file_paths: List[str], session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """Download files from sandbox using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(file_paths, session_id, **kwargs))

    async def _arun(
        self, file_paths: List[str], session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """Download files from sandbox using SDK. Returns [has_error, result_dict]"""
        try:
            actual_session_id = self._get_session_id(session_id)
            result = await self.client.download_files(actual_session_id, file_paths)

            # Process files and convert content to appropriate format
            files_data = []
            for file_content in result.files:
                content_b64 = base64.b64encode(file_content.content).decode("utf-8")
                files_data.append(
                    {
                        "path": file_content.path,
                        "content": content_b64,
                        "content_type": "base64",
                    }
                )

            result_dict = {
                "session_id": result.session_id,
                "files": files_data,
                "download_results": result.download_results,
                "error": False,
            }

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKListFilesTool(LLMSandboxSDKTool):
    """Tool for listing files in sandbox using SDK."""

    name: str = "list_sandbox_files_sdk"
    description: str = """List files and directories in the sandbox environment using SDK.
    Returns a structured representation of the directory tree.
    Use this tool to explore the sandbox filesystem."""
    args_schema: Type[BaseModel] = ListFilesSDKInput

    def _run(
        self, path: str = "/sandbox", session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """List files in sandbox using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(path, session_id, **kwargs))

    async def _arun(
        self, path: str = "/sandbox", session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """List files in sandbox using SDK. Returns [has_error, result_dict]"""
        try:
            actual_session_id = self._get_session_id(session_id)
            files_output = await self.client.list_files(actual_session_id, path)

            # Try to parse as JSON for structured data
            try:
                files_data = json.loads(files_output)
            except json.JSONDecodeError:
                # Fall back to raw output as string
                files_data = files_output

            result_dict = {
                "session_id": actual_session_id,
                "path": path,
                "files": files_data,
                "error": False,
            }

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKExecuteCommandTool(LLMSandboxSDKTool):
    """Tool for executing shell commands using SDK."""

    name: str = "execute_shell_command_sdk"
    description: str = """Execute shell commands in the sandbox environment using SDK.
    Returns stdout, stderr, and exit code of the command execution.
    Use this tool for file operations, system commands, or running scripts."""
    args_schema: Type[BaseModel] = ExecuteCommandSDKInput

    def _run(
        self, command: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """Execute shell command using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(command, session_id, **kwargs))

    async def _arun(
        self, command: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """Execute shell command using SDK. Returns [has_error, result_dict]"""
        try:
            actual_session_id = self._get_session_id(session_id)
            result = await self.client.execute_command(actual_session_id, command)

            result_dict = {
                "session_id": result.session_id,
                "command": command,
                "exit_code": result.return_code,
                "run_time": result.run_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": False,
            }

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKCreateSessionTool(LLMSandboxSDKTool):
    """Tool for creating new sessions using SDK."""

    name: str = "create_sandbox_session_sdk"
    description: str = """Create a new sandbox session using SDK.
    Returns session information including session ID.
    Use this tool to create sessions for before execution work."""
    args_schema: Type[BaseModel] = CreateSessionSDKInput

    def _run(
        self,
        language: str = "python",
        timeout: int = 300,
        cpu_mem_limit_mb: int = 256,
        cpu_core_limit: int = 1,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Create a new session using SDK (sync wrapper for async)."""
        return asyncio.run(
            self._arun(language, timeout, cpu_mem_limit_mb, cpu_core_limit, **kwargs)
        )

    async def _arun(
        self,
        language: str = "python",
        timeout: int = 300,
        cpu_mem_limit_mb: int = 256,
        cpu_core_limit: int = 1,
        **kwargs: Any,
    ) -> List[Union[bool, dict]]:
        """Create a new session using SDK. Returns [has_error, result_dict]"""
        try:
            session_info = await self.client.create_session(
                language=language,
                timeout=timeout,
                cpu_mem_limit_mb=cpu_mem_limit_mb,
                cpu_core_limit=cpu_core_limit,
            )

            result_dict = {
                "session_id": session_info.session_id,
                "status": session_info.status,
                "created_at": session_info.created_at,
                "language": language,
                "timeout": timeout,
                "cpu_mem_limit_mb": cpu_mem_limit_mb,
                "cpu_core_limit": cpu_core_limit,
                "error": False,
            }

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKDeleteSessionTool(LLMSandboxSDKTool):
    """Tool for deleting sessions using SDK."""

    name: str = "delete_sandbox_session_sdk"
    description: str = """Delete a sandbox session using SDK.
    Use this tool to clean up sessions that are no longer needed."""
    args_schema: Type[BaseModel] = DeleteSessionSDKInput

    def _run(self, session_id: str, **kwargs: Any) -> List[Union[bool, dict]]:
        """Delete a session using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(session_id, **kwargs))

    async def _arun(self, session_id: str, **kwargs: Any) -> List[Union[bool, dict]]:
        """Delete a session using SDK. Returns [has_error, result_dict]"""
        try:
            success = await self.client.delete_session(session_id)

            result_dict = {"session_id": session_id, "deleted": success, "error": False}

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


class LLMSandboxSDKListSessionsTool(LLMSandboxSDKTool):
    """Tool for listing sessions using SDK."""

    name: str = "list_sandbox_sessions_sdk"
    description: str = """List all sandbox sessions using SDK.
    Returns information about all active sessions.
    Use this tool to see available sessions."""
    args_schema: Type[BaseModel] = ListSessionsSDKInput

    def _run(
        self, status_filter: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """List sessions using SDK (sync wrapper for async)."""
        return asyncio.run(self._arun(status_filter, **kwargs))

    async def _arun(
        self, status_filter: Optional[str] = None, **kwargs: Any
    ) -> List[Union[bool, dict]]:
        """List sessions using SDK. Returns [has_error, result_dict]"""
        try:
            session_list = await self.client.list_sessions(status_filter=status_filter)

            # Convert sessions to dict format
            sessions_data = []
            for session in session_list.sessions:
                sessions_data.append(
                    {
                        "session_id": session.session_id,
                        "status": session.status,
                        "created_at": session.created_at,
                    }
                )

            result_dict = {
                "total_count": session_list.total_count,
                "active_count": session_list.active_count,
                "status_filter": status_filter,
                "sessions": sessions_data,
                "error": False,
            }

            return [False, result_dict]
        except Exception as e:
            return self._handle_error(e)


def create_sandbox_toolkit(
    base_url: str = "http://localhost:8000",
    http_timeout: float = 120,
    auto_create_session: bool = True,
) -> Tuple[AsyncSandboxClient, LLMSandboxSDKToolkit]:

    toolkit = LLMSandboxSDKToolkit(
        base_url=base_url,
        session_id=None,  # No default session, will create one
        auto_create_session=auto_create_session,
        timeout=http_timeout,
    )

    return toolkit


toolkit = create_sandbox_toolkit()

if __name__ == "__main__":
    # Example usage
    from langchain_core.utils.function_calling import convert_to_openai_tool

    async def main():
        async with llm_sandbox_toolkit as toolkit:
            result = await toolkit.ahealth_check()
            tools = toolkit.get_tools()

            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:100]}...")

            print("\nTool schemas:")
            for tool in tools:
                try:
                    openai_tool = convert_to_openai_tool(tool)
                    print(openai_tool)

                except Exception as e:
                    print(f"Error converting tool {tool.name}: {e}")

            print(f"Sandbox is healthy: {result}")

    asyncio.run(main())
