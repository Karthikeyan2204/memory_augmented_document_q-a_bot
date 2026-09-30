"""
mcp_server.py
--------------
Exposes the Q&A bot as MCP tools, so any MCP-compatible client (like Claude
Desktop) can use it directly.

Run standalone to test:
    python mcp_server.py

To register with Claude Desktop, add this to its config file
(claude_desktop_config.json):

{
  "mcpServers": {
    "qa-bot": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
"""

from mcp.server.fastmcp import FastMCP
from qa_bot import QABot

mcp = FastMCP("qa-bot")
bot = QABot()  # one shared bot for the life of the server


@mcp.tool()
def ask_question(user_id: str, question: str) -> str:
    """
    Ask the Q&A bot a question. Returns the short-term conversation
    history, the user's long-term memory, and matching document chunks
    with similarity scores. No LLM is used -- this is the raw retrieval
    result.
    """
    result = bot.ask(user_id, question)
    lines = [f"Returning user: {result['returning_user']}"]
    lines.append(f"\nShort-term memory:\n{result['short_term_history']}")
    lines.append(f"\nLong-term memory:\n{result['long_term_memory']}")
    lines.append("\nDocument matches:")
    for m in result["document_matches"]:
        lines.append(f"  [{m['source']}, distance={m['score']:.4f}] {m['text']}")
    return "\n".join(lines)


@mcp.tool()
def remember_fact(user_id: str, fact: str) -> str:
    """Save a fact to a user's long-term memory."""
    bot.remember(user_id, fact)
    return f"Saved for '{user_id}': {fact}"


@mcp.tool()
def ingest_documents() -> str:
    """(Re)load every .txt, .pdf, and .docx file in docs/ into the knowledge base."""
    result = bot.ingest_documents()
    msg = f"Added {result['chunks_added']} chunks to the knowledge base."
    if result["skipped"]:
        msg += "\nSkipped: " + "; ".join(result["skipped"])
    return msg


if __name__ == "__main__":
    mcp.run()
