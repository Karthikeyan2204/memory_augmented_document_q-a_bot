"""
chat.py
-------
Simple command-line chat with the memory-augmented Q&A bot.

    python chat.py

Type a user_id to identify yourself, then ask questions. Run this, exit,
and run it again with the same user_id to see long-term memory persist
across separate runs (short-term memory will be empty again, since that
only lives in RAM for one run).

Commands:
    /remember <fact>   save a fact to your long-term memory
    /quit              exit
"""

from qa_bot import QABot


def print_answer(result):
    print(f"\nReturning user? {result['returning_user']}")
    print(f"\n[Short-term memory - this session]\n{result['short_term_history']}")
    print(f"\n[Long-term memory - saved facts about you]\n{result['long_term_memory']}")
    print(f"\n[Document matches]")
    for i, m in enumerate(result["document_matches"], 1):
        print(f"  {i}. (source: {m['source']}, distance: {m['score']:.4f})")
        print(f"     {m['text'][:200]}{'...' if len(m['text']) > 200 else ''}")
    print()


def main():
    print("Loading embedding model (first time only, downloads ~80MB)...")
    bot = QABot()

    if bot.knowledge_base_size() == 0:
        print("\nKnowledge base is empty! Run 'python setup.py' first.\n")
        return

    user_id = input("Enter your user_id: ").strip() or "guest"
    print(f"\nChatting as '{user_id}'. Type /quit to exit.\n")

    while True:
        question = input(f"[{user_id}] > ").strip()
        if not question:
            continue
        if question == "/quit":
            break
        if question.startswith("/remember "):
            fact = question[len("/remember "):]
            bot.remember(user_id, fact)
            print(f"(saved to long-term memory: {fact})\n")
            continue

        result = bot.ask(user_id, question)
        print_answer(result)


if __name__ == "__main__":
    main()
