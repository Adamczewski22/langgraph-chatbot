from langchain_core.messages import BaseMessage

def get_message_window(messages: list[BaseMessage], window_size: int) -> list[BaseMessage]:
    """Get window_size messages from a list of messages with ToolMessages and tool calls filtered out"""
    return [
        m for m in messages
        if m.type in ["system", "human"]
        or (m.type == "ai" and not m.tool_calls)
    ][-window_size:]
