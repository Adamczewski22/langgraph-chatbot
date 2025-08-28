from langchain.chat_models import init_chat_model
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter

from langgraph.graph import StateGraph, MessagesState ,END
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from src.config import OPENAI_API_KEY

from rich.console import Console
from pathlib import Path

console = Console()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "faq.docx"

NUM_OF_RAG_RESULTS = 4
CONVO_MEMORY_WINDOW = 5

SYSTEM_TEXT = "Jesteś pomocnym asystentem, który odpowiada na pytania użytkownika. " \
              "W swojej odpowiedzi bazuj wyłącznie na zawartym kontekście. " \
              "Jeśli informacji potrzebnej do odpowiedzi nie ma w kontekście napisz: 'Nie posiadam informacji na ten temat'. " \
              "Odpowiadaj zwięźle (maksymalnie 3 zdania)."

CHECKPOINTER_THREAD_ID = "abc123"


llm = init_chat_model(model="gpt-4.1", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embedding=embeddings)
loader = Docx2txtLoader(FAQ_PATH)

docs = loader.load()
splitter = CharacterTextSplitter("#", chunk_size=240, chunk_overlap=0)
all_splits = splitter.split_documents(docs)

vector_store.add_documents(all_splits)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Zdobądź wszelkie informacje dotyczące firmy biletowej i wszystkich innych zagadnień"""
    context = vector_store.similarity_search(query, k=NUM_OF_RAG_RESULTS)
    console.print(context)
    serialized_context = "\n\n".join([doc.page_content for doc in context])
    return serialized_context, context


def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    history = [
        mes for mes in state["messages"]
    ][-CONVO_MEMORY_WINDOW:]

    response = llm_with_tools.invoke(history)
    return {"messages": [response]}


tool_node = ToolNode([retrieve])


def generate(state: MessagesState):
    tool_messages = []
    for message in reversed(state["messages"]):
        if (message.type == "tool"):
            tool_messages.append(message)
        else:
            break
    
    context = "\n\n".join([mes.content for mes in tool_messages])
    system_message = SystemMessage(SYSTEM_TEXT + f"\nKontekts (verbatim):\n```{context}```")

    history = state["messages"][-CONVO_MEMORY_WINDOW:]
    response = llm.invoke([system_message] + history)
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")

graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
)
graph_builder.add_edge("tools", "generate")


memory = MemorySaver()
config = {"configurable": {"thread_id": CHECKPOINTER_THREAD_ID}}

graph = graph_builder.compile(checkpointer=memory)

console.print(graph.invoke(
        {"messages": [{"role": "user", "content": "Helo"}]},
        config=config,
    )
)

