import asyncio
from rich.console import Console

from src.config import OPENAI_API_KEY
from src.graph import get_graph
from src.indexing import populate_vector_store

CHECKPOINTER_THREAD_ID = "abc123"


async def chat(input: str) -> str:
    graph = get_graph()
    config = {"configurable": {"thread_id": CHECKPOINTER_THREAD_ID}}

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": input}]},
        config=config,
    )

    return result["messages"][-1].content


async def main():
    console = Console()

    while True:
        console.print("[bold]Użytkownik[/bold]: ", end="")
        user_input = await asyncio.to_thread(input)

        if user_input in ["exit", "quit", "q"]:
            console.print(f"[blue][bold]Asystent[/bold]: Bywaj")
            break

        response = await chat(user_input)
        console.print(f"[blue][bold]Asystent[/bold]: {response}[/blue]")


if __name__ == "__main__":
    asyncio.run(main())
