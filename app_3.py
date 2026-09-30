"""
app.py
------
A simple web page for the Q&A bot, using Streamlit. Same QABot underneath
as chat.py and mcp_server.py -- this just gives it a browser interface
instead of a terminal.

Run:
    streamlit run app.py

First run downloads the embedding model (~80MB, needs internet once);
after that everything works offline.
"""

import os
import streamlit as st
from qa_bot import QABot, DOCS_FOLDER

st.set_page_config(page_title="Document Q&A Bot", page_icon="🤖")


# Load the bot once and reuse it across interactions (loading the embedding
# model is the slow part, so we don't want to redo it on every click).
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_bot():
    return QABot()


bot = load_bot()

st.title("🤖 Document Q&A Bot")
st.caption("Upload documents in the sidebar, then ask questions about them. No LLM is used — answers show the raw retrieved chunks and memory.")

# ---- sidebar: user identity + knowledge base controls ----
st.sidebar.header("Settings")

user_id = st.sidebar.text_input("Your user ID", value="guest")

st.sidebar.divider()
st.sidebar.write(f"**Knowledge base:** {bot.knowledge_base_size()} chunks loaded")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files and st.sidebar.button("Add to knowledge base"):
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    for uploaded_file in uploaded_files:
        save_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    with st.spinner("Chunking and embedding documents..."):
        result = bot.ingest_documents()

    st.sidebar.success(f"Added {result['chunks_added']} chunks from {len(uploaded_files)} file(s).")
    if result["skipped"]:
        st.sidebar.warning("Skipped: " + "; ".join(result["skipped"]))
    st.rerun()

st.sidebar.divider()
fact = st.sidebar.text_input("Save a fact to your long-term memory")
if st.sidebar.button("Save fact") and fact.strip():
    bot.remember(user_id, fact.strip())
    st.sidebar.success("Saved.")

# ---- main area: chat ----
if bot.knowledge_base_size() == 0:
    st.warning("Knowledge base is empty. Upload documents using the sidebar first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input box
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result = bot.ask(user_id, question)

    with st.chat_message("assistant"):
        st.markdown(f"**Returning user:** {result['returning_user']}")

        st.markdown("**Short-term memory (this session):**")
        st.text(result["short_term_history"])

        st.markdown("**Long-term memory (saved facts about you):**")
        st.text(result["long_term_memory"])

        st.markdown("**Matching document chunks:**")
        for m in result["document_matches"]:
            with st.expander(f"{m['source']} (distance: {m['score']:.4f})"):
                st.write(m["text"])

    # save a short summary into the displayed chat history
    summary = f"Returning user: {result['returning_user']}  \nFound {len(result['document_matches'])} matching chunk(s)."
    st.session_state.messages.append({"role": "assistant", "content": summary})