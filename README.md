# Memory-Augmented Document Q&A Bot

A simple Q&A bot that answers questions from your documents and remembers
things about each user across runs. No API keys needed anywhere — runs
fully offline after the first download.

## What's inside

| File | What it does |
|---|---|
| `qa_bot.py` | The bot itself — chunking, embeddings, vector store, short/long-term memory, the RAG loop. Everything important is in this one file. |
| `setup.py` | Run once to load `docs/*.txt` into the knowledge base |
| `chat.py` | Simple command-line chat with the bot |
| `mcp_server.py` | Exposes the bot as MCP tools |
| `mcp_client.py` | Connects to the MCP server and calls its tools — shows the full client → server round trip |
| `docs/` | Sample documents to ask questions about |

## How it maps to the requirements

- **Short-term vs long-term memory** — `qa_bot.py`: `ConversationBufferWindowMemory` (short-term, RAM only) vs `VectorStoreRetrieverMemory` (long-term, saved to disk in ChromaDB)
- **Vector stores and embedding storage** — ChromaDB, via `langchain_chroma.Chroma`
- **Semantic similarity and chunking** — `RecursiveCharacterTextSplitter` for chunking, ChromaDB similarity search for retrieval
- **Retrieval Augmented Memory (RAG) loop** — `QABot.ask()` in `qa_bot.py`
- **Demo: agent with short + long term memory** — `chat.py`
- **Hands-on: chunking + vector search** — `qa_bot.py`'s `ingest_documents()` and `ask()`
- **Workshop: improve response with memory access** — try the same question as a new `user_id` vs. one that already has long-term memory saved
- **MCP Demo** — `mcp_server.py` (server) + `mcp_client.py` (client)

No LLM is used anywhere — there's no chat-completion call, no generated
prose. Every answer is the raw retrieval result: matching document chunks
with similarity scores, plus whatever memory was found. This is RAG's
retrieval half, without the generation half, by design.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

No `.env` file, no API key needed. First run downloads a small embedding
model (~80MB) from HuggingFace; after that everything works offline.

## Adding your own documents

Drop any mix of `.txt`, `.pdf`, or `.docx` files into the `docs/` folder,
then run `python setup.py`. Each file becomes its own set of chunks
labeled by filename, so answers tell you which document they came from.
Scanned/image-only PDFs (no selectable text) will be skipped with a
warning — there's no OCR step.

## Run it

```bash
# 1. Load the sample documents into the knowledge base (run once)
python setup.py

# 2. Chat with the bot
python chat.py
```

Try asking things like:
- "How much storage does the Pro plan include?"
- "How do I restore a deleted file?"
- "My sync is stuck, what do I do?"

Inside the chat:
- `/remember <fact>` — save a fact to your long-term memory
- `/quit` — exit

Run `python chat.py` again later with the same `user_id` to see long-term
memory persist across separate runs.

## MCP

Run the server standalone (mostly for testing — it just waits for a client):
```bash
python mcp_server.py
```

Run the client, which starts the server itself and calls its tools:
```bash
python mcp_client.py
```

To register the server with Claude Desktop instead, add this to its config
file:
```json
{
  "mcpServers": {
    "qa-bot": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

## A note on versions

`requirements.txt` caps `langchain` below `0.3.27` on purpose. LangChain
moved `ConversationBufferWindowMemory` and `VectorStoreRetrieverMemory` out
of `langchain.memory` and into a separate `langchain_classic` package
starting at that version. Staying below it keeps the simple
`from langchain.memory import ...` lines in `qa_bot.py` working without
needing an extra package.
