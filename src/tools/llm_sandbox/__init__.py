from .langchain_tools import *

LLM_SANDBOX_CLIENT = AsyncSandboxClient(
    base_url="http://localhost:8000",  # sandox endpoint
    timeout=6000,  # 100 min timeout
)


## Json schema for the LLM Sandbox API
LLM_SANDBOX_EXECUTE_CODE_TOOL = {
    "name": "execute_python_code_sdk",
    "description": 'Execute Python code in a sandboxed environment using the SDK client. This method will automatically create a .py file named by random uuid under /sandbox.Returns the stdout, stderr, and exit code of the execution. Use this tool to run Python scripts, analyze data, or perform computations. For convenience, you could also specify libraries to install before execution.\n\n[IMPORTANT]\n1. Your default workspace is under "/sandbox". You are not permitted to operate content outside this scope.\n2. Write complete codes! The interpreter environment is isolated for each tool call. However, files under your workspace will be kept unchanged between calls.\n3. Use "list_sandbox_files_sdk" tool first if you\'re unsure about file structures.\n4. Install necessary libraries using "install_python_libraries_sdk" tool for import errors.\n5. Use "print" function to output results to stdout.',
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the sandbox",
            },
            "libraries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Libraries to install before execution (optional)",
            },
        },
        "required": ["code"],
    },
}

LLM_SANDBOX_EXECUTE_CODE_WITH_ARTIFACTS_TOOL = {
    "name": "execute_python_code_with_artifacts_sdk",
    "description": 'Execute Python code in a sandboxed environment and capture artifacts using SDK. Returns stdout, stderr, exit code and artifacts (like plots) from execution. Use this tool to run Python scripts that generate visualizations. For convenience, you could also specify libraries to install before execution.\n\n[IMPORTANT]\n1. Your default workspace is under "/sandbox". You are not permitted to operate content outside this scope.\n2. Write complete codes! The interpreter environment is isolated for each tool call.\n3. Supported visualization libraries include matplotlib, seaborn, plotly, etc.\n4. Use "print" function to output results to stdout.\n5. Artifacts will be captured automatically for supported plotting libraries.',
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the sandbox",
            },
            "libraries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Libraries to install before execution (optional)",
            },
        },
        "required": ["code"],
    },
}

LLM_SANDBOX_LIST_SANDBOX_FILES_TOOL = {
    "name": "list_sandbox_files_sdk",
    "description": "List files and directories in the sandbox environment using SDK. Returns a structured representation of the directory tree. Use this tool to explore the sandbox filesystem.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: /sandbox)",
            }
        },
        "required": [],
    },
}

LLM_SANDBOX_UPLOAD_FILE_TO_SANDBOX_TOOL = {
    "name": "upload_file_to_sandbox_sdk",
    "description": "Upload a file to the sandbox environment using SDK. Supports text files (with utf-8 encoding) and binary files (with base64 encoding). Use this tool to provide input files for your code.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "File content (text or base64 encoded for binary)",
            },
            "file_path": {
                "type": "string",
                "description": "Destination file path in the sandbox",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (utf-8 or base64 for binary)",
                "default": "utf-8",
            },
        },
        "required": ["content", "file_path"],
    },
}

LLM_SANDBOX_DOWNLOAD_FILES_FROM_SANDBOX_TOOL = {
    "name": "download_files_from_sandbox_sdk",
    "description": "Download files from the sandbox environment using SDK. Returns file contents as text (for text files) or base64 (for binary files). Use this tool to retrieve output files or results from your code.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to download from the sandbox",
            }
        },
        "required": ["file_paths"],
    },
}

LLM_SANDBOX_EXECUTE_COMMAND_TOOL = {
    "name": "execute_command_sdk",
    "description": "Execute a shell command in a sandbox session using SDK. Returns the stdout, stderr, and exit code of the command execution. Use this tool to run shell commands, system operations, or command-line utilities in the sandboxed environment. The root is set under /sandbox.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"}
        },
        "required": ["command"],
    },
}

LLM_SANDBOX_INSTALL_LIBRARIES_TOOL = {
    "name": "install_python_libraries_sdk",
    "description": "Install Python libraries in the sandbox environment using SDK. Takes a list of library names and installs them using pip. Use this tool to install dependencies before running code.",
    "parameters": {
        "type": "object",
        "properties": {
            "libraries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of Python library names to install",
            }
        },
        "required": ["libraries"],
    },
}
