from rich.console import Console

from src.config import OPENAI_API_KEY
from src.graph import build_graph
from src.indexing import populate_vector_store

CHECKPOINTER_THREAD_ID = "abc123"

def chat():
    console = Console()
    graph = build_graph()
    populate_vector_store()

    config = {"configurable": {"thread_id": CHECKPOINTER_THREAD_ID}}

    while True:
        console.print("[bold]Użytkownik[/bold]: ", end="")
        user_input = input()

        if user_input in ["exit", "quit", "q"]:
            console.print(f"[blue][bold]Asystent[/bold]: Bywaj")
            break

        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            message = step["messages"][-1]
            if message.type == "ai" and not message.tool_calls:
                console.print(f"[blue][bold]Asystent[/bold]: {message.content}[/blue]")


if __name__ == "__main__":
    chat()