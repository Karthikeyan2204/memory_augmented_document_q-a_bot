"""
qa_bot.py
---------
A memory-augmented document Q&A bot. Kept deliberately simple: one class,
no LLM call (pure retrieval), no API key.

How it works, step by step:

1. CHUNKING: documents in docs/ (.txt, .pdf, or .docx -- mix and match)
   are split into overlapping text chunks using LangChain's
   RecursiveCharacterTextSplitter.

2. EMBEDDINGS + VECTOR STORE: each chunk is turned into a vector (an
   embedding) using a free local model (HuggingFace sentence-transformers,
   no API key) and stored in ChromaDB, a vector database that supports
   fast similarity search.

3. SHORT-TERM MEMORY: the last few messages in the current conversation,
   kept in plain Python memory (RAM). Uses LangChain's
   ConversationBufferWindowMemory. Gone when the program exits.

4. LONG-TERM MEMORY: facts about a specific user, also embedded and
   stored in ChromaDB (a separate collection per user), using LangChain's
   VectorStoreRetrieverMemory. Saved to disk, so it's still there next
   time you run the program -- this is what lets the bot "remember" a
   user across separate runs.

5. RAG LOOP: when you ask a question, the bot:
     a. searches the document knowledge base for matching chunks
     b. searches the user's long-term memory for matching past facts
     c. looks at the short-term conversation history
     d. combines all three into one answer
   No language model writes the answer -- the bot returns exactly what
   it retrieved, clearly labeled, which is the "Retrieval Augmented
   Memory" part of the brief without the generation part.
"""

import os
import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---- settings (change these if you want) ----
DOCS_FOLDER = "docs"
CHROMA_FOLDER = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
SHORT_TERM_TURNS = 3      # how many past exchanges to keep verbatim
TOP_K = 3                  # how many results to retrieve per search


def get_embeddings():
    """Free local embedding model. Downloads once (~80MB), then works offline."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def read_file_text(filepath):
    """
    Reads a file and returns its plain text, regardless of format.
    Supports .txt, .pdf, and .docx. Add more formats here if needed.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".pdf":
        reader = PdfReader(filepath)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if ext == ".docx":
        doc = DocxDocument(filepath)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported file type: {ext}")


class QABot:
    def __init__(self):
        self.embeddings = get_embeddings()

        # the document knowledge base (one shared ChromaDB collection)
        self.knowledge_base = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_FOLDER,
        )

        # one short-term memory + one long-term memory per user_id
        self._short_term = {}   # {user_id: ConversationBufferWindowMemory}
        self._long_term = {}    # {user_id: VectorStoreRetrieverMemory}

    # ---------------- chunking + ingestion ----------------

    def ingest_documents(self):
        """
        Chunks every .txt, .pdf, and .docx file in docs/ and embeds them
        into the knowledge base. You can mix and match formats freely --
        just drop files into docs/ and run this (or setup.py).
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # pick up all supported file types
        filepaths = []
        for pattern in ("*.txt", "*.pdf", "*.docx"):
            filepaths.extend(glob.glob(os.path.join(DOCS_FOLDER, pattern)))

        all_chunks = []
        all_sources = []
        skipped = []

        for filepath in filepaths:
            filename = os.path.basename(filepath)
            try:
                text = read_file_text(filepath)
            except Exception as e:
                skipped.append(f"{filename} ({e})")
                continue

            if not text.strip():
                skipped.append(f"{filename} (no extractable text -- might be a scanned/image-only PDF)")
                continue

            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))

        if all_chunks:
            self.knowledge_base.add_texts(
                texts=all_chunks,
                metadatas=[{"source": s} for s in all_sources],
            )
        return {"chunks_added": len(all_chunks), "skipped": skipped}

    def knowledge_base_size(self):
        return len(self.knowledge_base.get().get("ids", []))

    # ---------------- memory setup ----------------

    def _get_short_term(self, user_id):
        if user_id not in self._short_term:
            self._short_term[user_id] = ConversationBufferWindowMemory(
                k=SHORT_TERM_TURNS,
                memory_key="history",
                input_key="input",
            )
        return self._short_term[user_id]

    def _get_long_term(self, user_id):
        if user_id not in self._long_term:
            # each user gets their own ChromaDB collection, so users never
            # see each other's long-term memory
            safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
            vectorstore = Chroma(
                collection_name=f"user_memory_{safe_id}",
                embedding_function=self.embeddings,
                persist_directory=CHROMA_FOLDER,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="memory")
            memory._vectorstore = vectorstore  # keep a handle for is_returning_user()
            self._long_term[user_id] = memory
        return self._long_term[user_id]

    def is_returning_user(self, user_id):
        """True if this user already has long-term memory saved from a past run."""
        long_term = self._get_long_term(user_id)
        existing = long_term._vectorstore.get()
        return len(existing.get("ids", [])) > 0

    def remember(self, user_id, fact):
        """Explicitly save a fact to a user's long-term memory."""
        long_term = self._get_long_term(user_id)
        long_term.save_context({"input": "note"}, {"response": fact})

    # ---------------- the RAG loop ----------------

    def ask(self, user_id, question):
        """
        Runs the full retrieval loop and returns a dict with everything
        found: short-term history, long-term memory hits, and matching
        document chunks. This dict IS the bot's answer -- no LLM involved.
        """
        short_term = self._get_short_term(user_id)
        long_term = self._get_long_term(user_id)
        returning = self.is_returning_user(user_id)

        # 1. search document knowledge base
        doc_hits = self.knowledge_base.similarity_search_with_score(question, k=TOP_K)

        # 2. search this user's long-term memory
        memory_vars = long_term.load_memory_variables({"prompt": question})
        long_term_text = memory_vars.get("memory", "").strip()

        # 3. read short-term conversation history
        history_vars = short_term.load_memory_variables({})
        history_text = history_vars.get("history", "").strip()

        # 4. save this turn into both memories for next time
        short_term.save_context({"input": question}, {"response": f"(retrieved {len(doc_hits)} chunk(s))"})
        long_term.save_context({"input": question}, {"response": f"asked about: {question}"})

        return {
            "returning_user": returning,
            "short_term_history": history_text or "(no earlier turns this session)",
            "long_term_memory": long_term_text or "(no long-term memory yet for this user)",
            "document_matches": [
                {"text": doc.page_content, "source": doc.metadata.get("source", "?"), "score": score}
                for doc, score in doc_hits
            ],
        }