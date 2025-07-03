#!/usr/bin/env python3
"""
Direct test of MCP server health_check tool using SSE client
"""

import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

async def test_mcp_health_check():
    """Test the MCP server health_check tool directly"""
    print("Testing MCP server health_check tool...")
    
    try:
        # Connect to the MCP server via SSE
        async with sse_client("http://localhost:8015/sse") as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                print("Initializing MCP session...")
                await session.initialize()
                
                # List available tools
                print("Listing available tools...")
                tools_result = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools_result.tools]}")
                
                # Find the health_check tool
                health_check_tool = None
                for tool in tools_result.tools:
                    if tool.name == "health_check":
                        health_check_tool = tool
                        break
                
                if not health_check_tool:
                    print("ERROR: health_check tool not found!")
                    return
                
                print(f"Found health_check tool: {health_check_tool.name}")
                print(f"Description: {health_check_tool.description}")
                
                # Call the health_check tool
                print("Calling health_check tool...")
                result = await session.call_tool(
                    name="health_check",
                    arguments={}
                )
                
                print(f"Raw result: {result}")
                print(f"Content: {result.content}")
                
                # Parse the JSON response
                if result.content and len(result.content) > 0:
                    content_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    try:
                        parsed_result = json.loads(content_text)
                        print(f"Parsed result: {json.dumps(parsed_result, indent=2)}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                        print(f"Content was: {content_text}")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_health_check())
