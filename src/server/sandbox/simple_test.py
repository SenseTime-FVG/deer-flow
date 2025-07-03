#!/usr/bin/env python3
"""
Simple MCP client test using native MCP client
"""

import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

async def test_basic_mcp_connection():
    """Test basic MCP connection"""
    print("Testing basic MCP connection to http://localhost:8015/sse")
    
    try:
        # Test connection
        print("Attempting to connect via SSE...")
        async with sse_client("http://localhost:8015/sse") as (read, write):
            print("✓ SSE connection established")
            
            async with ClientSession(read, write) as session:
                print("✓ ClientSession created")
                
                # Initialize
                print("Initializing session...")
                init_result = await session.initialize()
                print(f"✓ Session initialized: {init_result}")
                
                # List tools
                print("Listing tools...")
                tools_result = await session.list_tools()
                print(f"✓ Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                return True
                
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_langchain_mcp_client():
    """Test using langchain_mcp_adapters"""
    print("\nTesting langchain_mcp_adapters MultiServerMCPClient...")
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        mcp_servers = {
            "LLM_Sandbox": {"url": "http://localhost:8015/sse", "transport": "sse"}
        }
        
        async with MultiServerMCPClient(mcp_servers) as client:
            print("✓ MultiServerMCPClient connected")
            tools = client.get_tools()
            print(f"✓ Retrieved {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}")
            return True
            
    except Exception as e:
        print(f"✗ MultiServerMCPClient failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("=== MCP Connection Test ===")
    
    # Test 1: Basic MCP connection
    basic_success = await test_basic_mcp_connection()
    
    # Test 2: langchain_mcp_adapters
    langchain_success = await test_langchain_mcp_client()
    
    print(f"\n=== Results ===")
    print(f"Basic MCP connection: {'✓ SUCCESS' if basic_success else '✗ FAILED'}")
    print(f"langchain_mcp_adapters: {'✓ SUCCESS' if langchain_success else '✗ FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())
