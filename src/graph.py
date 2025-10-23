from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from functools import lru_cache

from src.nodes import query_or_respond, generate
from src.tools import retrieve

@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """Builds and returns a runnable graph"""

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node("tools", ToolNode([retrieve]))
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")

    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
    )
    graph_builder.add_edge("tools", "generate")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph
