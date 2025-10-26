from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.llm import get_vector_store
from src.config import OPENAI_API_KEY

SEPARATOR = "#"
CHUNK_SIZE = 240
CHUNK_OVERLAP = 0
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "faq.docx"


def populate_vector_store() -> None:
    """Populates the persistent vector store with indexed data"""
    vs = get_vector_store()
    docs = load_docs(FAQ_PATH)
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


def load_docs(path: Path) -> list[Document]:
    """Loads the document based on path"""
    loader = Docx2txtLoader(path)
    docs = loader.load()
    return docs


if __name__ == "__main__":
    populate_vector_store()
