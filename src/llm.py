from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

LLM_MODEL = "gpt-4.1"
LLM_PROVIDER = "openai"
EMBEDDINGS_MODEL = "text-embedding-3-large"


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    return init_chat_model(model=LLM_MODEL, model_provider=LLM_PROVIDER)

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

@lru_cache(maxsize=1)
def get_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(get_embeddings())
