from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

def env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val

OPENAI_API_KEY = env("OPENAI_API_KEY")
LANGSMITH_API_KEY = env("LANGSMITH_API_KEY")