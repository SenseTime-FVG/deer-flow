# LLM Sandbox Client SDK

A comprehensive Python SDK for interacting with the LLM Sandbox API server. This SDK provides both synchronous and asynchronous clients with full support for session management, code execution, file operations, and resource cleanup.

## Features

- 🔄 **Dual API Support**: Both synchronous and asynchronous clients
- 🛡️ **Context Managers**: Automatic resource cleanup with `with` and `async with`
- 📝 **Type Hints**: Full type annotation support
- 🎯 **Comprehensive Error Handling**: Detailed exception hierarchy
- 📁 **File Operations**: Upload/download files with base64 encoding
- 🔧 **Library Management**: Install Python packages dynamically
- 🖥️ **Shell Commands**: Execute shell commands in sandbox
- 📊 **Session Management**: Create, monitor, and cleanup sessions
- 🎨 **Artifact Support**: Handle plots, images, and other generated files

## Installation

```bash
# Install the main package (includes the SDK)
pip install llm-sandbox-wrapper
```

## Quick Start

### Async Client Example

```python
import asyncio
from llm_sandbox_wrapper.sdk import AsyncSandboxClient, async_sandbox_session

async def main():
    # Method 1: Manual session management
    async with AsyncSandboxClient("http://localhost:8000") as client:
        session = await client.create_session()
        
        result = await client.run_code(
            session.session_id,
            "print('Hello World!')"
        )
        print(result.stdout)  # Output: Hello World!
        
        await client.delete_session(session.session_id)
    
    # Method 2: Automatic session management (recommended)
    async with async_sandbox_session("http://localhost:8000") as (client, session):
        result = await client.run_code(
            session.session_id,
            "import sys; print(sys.version)"
        )
        print(f"Python version: {result.stdout}")

asyncio.run(main())
```

### Sync Client Example

```python
from llm_sandbox_wrapper.sdk import SandboxClient, sandbox_session

# Method 1: Manual session management
with SandboxClient("http://localhost:8000") as client:
    session = client.create_session()
    
    result = client.run_code(
        session.session_id,
        "print('Hello from sync client!')"
    )
    print(result.stdout)
    
    client.delete_session(session.session_id)

# Method 2: Automatic session management (recommended)
with sandbox_session("http://localhost:8000") as (client, session):
    result = client.run_code(
        session.session_id,
        "print(f'2 + 2 = {2 + 2}')"
    )
    print(result.stdout)  # Output: 2 + 2 = 4
```

## Core Classes

### AsyncSandboxClient

Asynchronous client for high-performance applications.

```python
async with AsyncSandboxClient("http://localhost:8000") as client:
    # All operations are async
    session = await client.create_session()
    result = await client.run_code(session.session_id, "print('async')")
```

### SandboxClient

Synchronous client for simple use cases.

```python
with SandboxClient("http://localhost:8000") as client:
    # All operations are synchronous
    session = client.create_session()
    result = client.run_code(session.session_id, "print('sync')")
```

## Advanced Usage

### Code Execution with Libraries

```python
async with async_sandbox_session("http://localhost:8000") as (client, session):
    # Install libraries
    install_result = await client.install_libraries(
        session.session_id,
        ["numpy", "matplotlib", "pandas"]
    )
    
    # Run code using the libraries
    result = await client.run_code(
        session.session_id,
        """
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)
print(f"Generated {len(x)} data points")
        """
    )
```

### File Operations

```python
from llm_sandbox_wrapper.sdk import FileContent

async with async_sandbox_session("http://localhost:8000") as (client, session):
    # Upload files
    file_content = FileContent(
        path="/sandbox/data.txt",
        content=b"This is my data file content"
    )
    
    upload_result = await client.upload_files(
        session.session_id,
        [file_content]
    )
    
    # Run code that processes the file
    result = await client.run_code(
        session.session_id,
        """
with open('/sandbox/data.txt', 'r') as f:
    content = f.read()
    print(f"File content: {content}")

# Create a new file
with open('/sandbox/output.txt', 'w') as f:
    f.write("Processed data")
        """
    )
    
    # Download the output file
    download_result = await client.download_files(
        session.session_id,
        ["/sandbox/output.txt"]
    )
    
    for file in download_result.files:
        print(f"Downloaded {file.path}: {file.content}")
```

### Artifacts (Plots, Images, etc.)

```python
async with async_sandbox_session("http://localhost:8000") as (client, session):
    # Install matplotlib
    await client.install_libraries(session.session_id, ["matplotlib"])
    
    # Create a plot and save it
    artifact_result = await client.run_code_with_artifacts(
        session.session_id,
        """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.savefig('/sandbox/sine_wave.png', dpi=150, bbox_inches='tight')
plt.close()

print('Plot saved successfully')
        """
    )
    
    print(f"Artifacts created: {artifact_result.files}")
    
    # Download the plot
    download_result = await client.download_files(
        session.session_id,
        ["/sandbox/sine_wave.png"]
    )
```

### Shell Commands

```python
async with async_sandbox_session("http://localhost:8000") as (client, session):
    # Execute shell commands
    result = await client.execute_command(
        session.session_id,
        "ls -la /sandbox"
    )
    print(f"Directory listing:\n{result.stdout}")
    
    # Install system packages (if needed)
    result = await client.execute_command(
        session.session_id,
        "apt-get update && apt-get install -y curl"
    )
```

### Session Management

```python
async with AsyncSandboxClient("http://localhost:8000") as client:
    # Create multiple sessions
    session1 = await client.create_session(language="python", cpu_mem_limit_mb=512)
    session2 = await client.create_session(language="python", cpu_mem_limit_mb=1024)
    
    # List all sessions
    sessions = await client.list_sessions()
    print(f"Total sessions: {sessions.total_count}")
    print(f"Active sessions: {sessions.active_count}")
    
    for session_info in sessions.sessions:
        print(f"Session {session_info.session_id}: {session_info.status}")
    
    # Get specific session info
    info = await client.get_session(session1.session_id)
    print(f"Session created at: {info.created_at}")
    
    # Clean up
    await client.delete_session(session1.session_id)
    await client.delete_session(session2.session_id)
```

## Error Handling

The SDK provides a comprehensive exception hierarchy:

```python
from llm_sandbox_wrapper.sdk import (
    SandboxClientError, SessionNotFoundError, 
    ServerError, NetworkError, ValidationError
)

async with AsyncSandboxClient("http://localhost:8000") as client:
    try:
        session = await client.create_session()
        result = await client.run_code(session.session_id, "invalid code")
        
    except SessionNotFoundError:
        print("Session was not found")
    except ServerError as e:
        print(f"Server error {e.status_code}: {e.detail}")
    except NetworkError:
        print("Network connection failed")
    except SandboxClientError as e:
        print(f"Client error: {e}")
```

## Data Models

The SDK uses type-safe data models:

```python
from llm_sandbox_wrapper.sdk.models import (
    SessionInfo, CodeResult, ArtifactResult, 
    CommandResult, FileContent, SessionList
)

# All API responses are parsed into these models
session_info: SessionInfo = await client.get_session(session_id)
code_result: CodeResult = await client.run_code(session_id, code)
file_content: FileContent = FileContent(path="/path", content=b"data")
```

## Configuration

### Client Configuration

```python
# Configure timeouts and other options
client = AsyncSandboxClient(
    base_url="http://localhost:8000",
    timeout=60.0  # Request timeout in seconds
)

# Session configuration
session = await client.create_session(
    language="python",
    timeout=600,  # Session timeout in seconds
    cpu_mem_limit_mb=1024,  # Memory limit
    cpu_core_limit=2  # CPU core limit
)
```

### Logging

```python
import logging

# Enable debug logging for the SDK
logging.getLogger("llm_sandbox_wrapper.sdk").setLevel(logging.DEBUG)
```

## Concurrent Operations

### Multiple Sessions

```python
import asyncio

async with AsyncSandboxClient("http://localhost:8000") as client:
    # Create multiple sessions concurrently
    sessions = await asyncio.gather(*[
        client.create_session() for _ in range(5)
    ])
    
    # Run code in all sessions concurrently
    results = await asyncio.gather(*[
        client.run_code(session.session_id, f"print('Session {i}')")
        for i, session in enumerate(sessions)
    ])
    
    # Clean up all sessions
    await asyncio.gather(*[
        client.delete_session(session.session_id)
        for session in sessions
    ])
```

### Same Session Operations

Operations on the same session are automatically serialized by the server to prevent race conditions:

```python
async with async_sandbox_session("http://localhost:8000") as (client, session):
    # These operations will be serialized automatically
    tasks = [
        client.run_code(session.session_id, f"print('Operation {i}')")
        for i in range(3)
    ]
    
    results = await asyncio.gather(*tasks)
```

## Testing

The SDK includes comprehensive tests:

```bash
# Run with pytest
pip install pytest pytest-asyncio
pytest llm_sandbox_wrapper/sdk/test_sdk.py

# Or run manually
python llm_sandbox_wrapper/sdk/test_sdk.py
```

## Examples

See `llm_sandbox_wrapper/sdk/examples.py` for comprehensive usage examples.

## API Reference

### AsyncSandboxClient Methods

- `async create_session(...) -> SessionInfo`
- `async get_session(session_id) -> SessionInfo`
- `async delete_session(session_id) -> bool`
- `async list_sessions() -> SessionList`
- `async run_code(session_id, code, libraries=None) -> CodeResult`
- `async run_code_with_artifacts(session_id, code, libraries=None) -> ArtifactResult`
- `async execute_command(session_id, command) -> CommandResult`
- `async install_libraries(session_id, libraries) -> LibraryInstallResult`
- `async upload_files(session_id, files) -> FileUploadResult`
- `async download_files(session_id, file_paths) -> FileDownloadResult`
- `async list_files(session_id, root="/sandbox") -> str`
- `async health_check() -> bool`

### SandboxClient Methods

Same methods as AsyncSandboxClient but without `async`/`await`.

## Best Practices

1. **Always use context managers** for automatic resource cleanup
2. **Use automatic session management** (`async_sandbox_session`/`sandbox_session`) when possible
3. **Handle exceptions** appropriately for production code
4. **Use async client** for high-performance applications
5. **Install libraries once** per session and reuse the session
6. **Monitor session resources** and clean up when done
7. **Use file operations** for persistent data between code executions

## Contributing

Contributions are welcome! Please see the main project repository for contribution guidelines.

## License

This SDK is part of the LLM Sandbox project. See the main project for license information.
