#!/usr/bin/env python3
"""
Fixed MCP client test using native MCP client only
"""

import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

class NativeMCPClient:
    """Native MCP client wrapper for testing"""
    
    def __init__(self, url: str):
        self.url = url
        self.session = None
        self.tools = []
    
    async def __aenter__(self):
        print(f"Connecting to {self.url}...")
        self.context = sse_client(self.url)
        self.read, self.write = await self.context.__aenter__()
        
        print("Creating session...")
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        
        print("Initializing session...")
        await self.session.initialize()
        
        print("Loading tools...")
        tools_result = await self.session.list_tools()
        self.tools = tools_result.tools
        print(f"Loaded {len(self.tools)} tools")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if hasattr(self, 'context'):
            await self.context.__aexit__(exc_type, exc_val, exc_tb)
    
    async def call_tool(self, name: str, arguments: dict = None):
        """Call a tool by name"""
        if arguments is None:
            arguments = {}
        
        result = await self.session.call_tool(name=name, arguments=arguments)
        return result

async def test_health_check():
    """Test the health_check tool"""
    print("=== Testing Native MCP Client ===")
    
    try:
        async with NativeMCPClient("http://localhost:8015/sse") as client:
            print(f"✓ Connected successfully")
            print(f"Available tools: {[tool.name for tool in client.tools]}")
            
            # Test health_check
            print("\nTesting health_check tool...")
            result = await client.call_tool("health_check")
            
            print(f"Raw result: {result}")
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    response_text = content.text
                else:
                    response_text = str(content)
                
                print(f"Response: {response_text}")
                
                try:
                    parsed = json.loads(response_text)
                    print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
                    
                    if parsed.get('success'):
                        print("✓ Health check passed!")
                    else:
                        print(f"✗ Health check failed: {parsed.get('error_message', 'Unknown error')}")
                        
                except json.JSONDecodeError as e:
                    print(f"✗ Failed to parse response as JSON: {e}")
            
            return True
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_create_session():
    """Test session creation"""
    print("\n=== Testing Session Creation ===")
    
    try:
        async with NativeMCPClient("http://localhost:8015/sse") as client:
            print("Testing create_session tool...")
            result = await client.call_tool("create_session", {
                "language": "python",
                "timeout": 300
            })
            
            if result.content and len(result.content) > 0:
                content = result.content[0]
                response_text = content.text if hasattr(content, 'text') else str(content)
                print(f"Create session response: {response_text}")
                
                try:
                    parsed = json.loads(response_text)
                    if parsed.get('success'):
                        session_id = parsed.get('session_id')
                        print(f"✓ Session created: {session_id}")
                        return session_id
                    else:
                        print(f"✗ Session creation failed: {parsed.get('error_message')}")
                except json.JSONDecodeError:
                    print(f"✗ Failed to parse response: {response_text}")
            
            return None
            
    except Exception as e:
        print(f"✗ Session creation test failed: {e}")
        return None

async def main():
    print("=== Enhanced MCP Testing ===")
    
    # Test health check
    health_success = await test_health_check()
    
    # Test session creation if health check passed
    if health_success:
        session_id = await test_create_session()
        if session_id:
            print(f"\n✓ All tests passed! Session ID: {session_id}")
        else:
            print("\n⚠ Health check passed but session creation failed")
    else:
        print("\n✗ Health check failed, skipping other tests")

if __name__ == "__main__":
    asyncio.run(main())
