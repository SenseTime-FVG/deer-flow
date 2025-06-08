
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import NonTextContent
from typing import Any
import os.path as osp

import base64
import filetype
import logging
logger = logging.getLogger(__name__)

def file_to_data_uri(file_path):
    """读取一个文件并将其转换成base64编码利用filetype库,在前面加上mime信息,返回data:uri格式"""
    with open(file_path, 'rb') as file:
        file_content = file.read()
    
    kind = filetype.guess(file_content)
    if kind is None:
        mime_type = 'application/octet-stream'
    else:
        mime_type = kind.mime
    base64_content = base64.b64encode(file_content).decode('utf-8')
    mime_base64 = f"data:{mime_type};base64,{base64_content}"
    return mime_base64

def add_file_upload_in_tool(tool: BaseTool):
    old_coroutine = tool.coroutine
    
    async def wrapped_call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        for k in list(arguments.keys()):
            if 'uri' == k.lower() and osp.exists(arguments[k]):
                arguments[k] = file_to_data_uri(arguments[k])
        out = await old_coroutine(**arguments)
        return out
    
    tool.coroutine = wrapped_call_tool
    return tool

class MultiServerMCPClient_wFileUpload(MultiServerMCPClient):
    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        await session.initialize()
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        server_tools = [add_file_upload_in_tool(tool) for tool in server_tools]
        self.server_name_to_tools[server_name] = server_tools
