from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

from src.llm import get_vector_store

SEPARATOR = "#"
CHUNK_SIZE = 240
CHUNK_OVERLAP = 0
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "faq.docx"


async def populate_vector_store() -> None:
    """Populates the in-memory vector store with indexed data"""
    vs = get_vector_store()
    docs = await load_docs(FAQ_PATH)
    splits = split_docs(docs)
    vs.add_documents(splits)


def split_docs(docs: list[Document]) -> list[Document]:
    """Chunks the documents"""
    splitter = CharacterTextSplitter(
        separator=SEPARATOR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)
    return splits


async def load_docs(path: Path) -> list[Document]:
    """Loads the document based on path"""
    loader = Docx2txtLoader(path)
    docs = await loader.aload()
    return docs
