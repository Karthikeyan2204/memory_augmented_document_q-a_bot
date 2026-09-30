

from qa_bot import QABot

if __name__ == "__main__":
    print("Loading embedding model (first time only, downloads ~80MB)...")
    bot = QABot()

    print("Chunking and embedding documents from docs/ ...")
    result = bot.ingest_documents()

    print(f"Done. Added {result['chunks_added']} chunks.")
    if result["skipped"]:
        print("\nSkipped (could not extract text):")
        for item in result["skipped"]:
            print(f"  - {item}")

    print(f"\nKnowledge base now has {bot.knowledge_base_size()} chunks total.")
