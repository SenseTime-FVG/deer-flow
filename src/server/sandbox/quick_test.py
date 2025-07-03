import asyncio
import pytest
from contextlib import asynccontextmanager
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


class TestSandboxMCPServer:

    def get_client(self):
        mcp_servers = {
            "LLM_Sandbox": {"url": "http://localhost:8015/sse", "transport": "sse"}
        }
        client = MultiServerMCPClient(mcp_servers)
        return client

    @pytest.mark.asyncio
    async def test_health_check(self):
        client = self.get_client()
        from langchain_mcp_adapters.tools import load_mcp_tools

        async with client.session("LLM_Sandbox") as session:
            tools = await load_mcp_tools(session)
            print(tools)
