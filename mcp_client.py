"""
mcp_client.py
--------------
A simple MCP CLIENT that connects to mcp_server.py and calls its tools.

This is the other half of MCP: mcp_server.py EXPOSES tools, this script
CONSUMES them. It starts mcp_server.py as a subprocess, talks to it over
stdio using the real MCP protocol (the same way Claude Desktop or any other
MCP host would), and prints the results.

No LLM, no API key -- this just demonstrates the full client-server round
trip: connect -> list tools -> call tools -> read results.

Run:
    python mcp_client.py
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # tells the client how to start the server: run "python mcp_server.py"
    server_params = StdioServerParameters(
        command=sys.executable,   # use the same python interpreter we're running with
        args=["mcp_server.py"],
    )

    print("Starting mcp_server.py as a subprocess and connecting...\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # MCP handshake
            await session.initialize()

            # 1. list available tools, proves the connection works
            tools_response = await session.list_tools()
            print("Connected. Tools exposed by the server:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # 2. make sure the knowledge base has documents in it
            print("Calling tool: ingest_documents()")
            result = await session.call_tool("ingest_documents", {})
            print(result.content[0].text)
            print()

            # 3. ask a question as a new user
            print("Calling tool: ask_question(user_id='client_demo_user', question='How much storage does the Pro plan include?')")
            result = await session.call_tool(
                "ask_question",
                {"user_id": "client_demo_user", "question": "How much storage does the Pro plan include?"},
            )
            print(result.content[0].text)
            print()

            # 4. save a long-term memory fact for that user
            print("Calling tool: remember_fact(user_id='client_demo_user', fact='Prefers email support over chat.')")
            result = await session.call_tool(
                "remember_fact",
                {"user_id": "client_demo_user", "fact": "Prefers email support over chat."},
            )
            print(result.content[0].text)
            print()

            # 5. ask again, to show long-term memory now shows up in the retrieval result
            print("Calling tool: ask_question() again, same user, different question")
            result = await session.call_tool(
                "ask_question",
                {"user_id": "client_demo_user", "question": "How do I cancel my subscription?"},
            )
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
