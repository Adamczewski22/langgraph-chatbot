from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

from src.llm import get_llm
from src.tools import retrieve
from src.utils import get_message_window


CONVO_MEMORY_WINDOW = 6

SYSTEM_TEXT = """Jesteś asystentem obsługi klienta firmy biletowej.

Zasady odpowiedzi (RAG + narzędzie):
1) ZAWSZE odpowiadaj wyłącznie na podstawie dostarczonego kontekstu verbatim chyba że się witasz, lub tłumaczysz kim jesteś.
2) Jeśli kontekst jest pusty lub niepewny - NAJPIERW wywołaj narzędzie `retrieve`
   z pytaniem użytkownika (możesz lekko je przeformułować). Nie udzielaj odpowiedzi
   zanim nie pobierzesz kontekstu.
3) Jeśli po pobraniu kontekstu nadal brak jednoznacznej odpowiedzi, odpowiedz dokładnie:
   "Nie posiadam informacji na ten temat. Napisz do nas: support@mail.pl".
4) Gdy odpowiedź jest w kontekście, podaj ją (maks. 3 zdania), bez domysłów i
   bez faktów spoza kontekstu.
5) Odpowiadaj w języku użytkownika (domyślnie po polsku), bądź uprzejmy, ciepły, i rzeczowy.
"""


async def query_or_respond(state: MessagesState):
    llm_with_tools = get_llm().bind_tools([retrieve])
    system_message = SystemMessage(SYSTEM_TEXT)
    history = get_message_window(state["messages"], CONVO_MEMORY_WINDOW)

    response = await llm_with_tools.ainvoke([system_message] + history)
    return {"messages": [response]} 


async def generate(state: MessagesState):
    tool_messages = []
    for message in reversed(state["messages"]):
        if (message.type == "tool"):
            tool_messages.append(message)
        else:
            break
    
    context = "\n\n".join([mes.content for mes in tool_messages])
    system_message = SystemMessage(SYSTEM_TEXT + f"\nKONTEKST VERBATIM:\n```{context}```")
    history = get_message_window(state["messages"], CONVO_MEMORY_WINDOW)

    response = await get_llm().ainvoke([system_message] + history)
    return {"messages": [response]}