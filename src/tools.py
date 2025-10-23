from langchain_core.tools import tool
from src.llm import get_vector_store

NUM_OF_RAG_RESULTS = 4

@tool(response_format="content_and_artifact")
async def retrieve(query: str):
    """Zdobądź wszelkie informacje dotyczące firmy biletowej i wszystkich innych zagadnień"""
    context = await get_vector_store().asimilarity_search(query, k=NUM_OF_RAG_RESULTS)
    serialized_context = "\n\n".join([doc.page_content for doc in context])
    return serialized_context, context