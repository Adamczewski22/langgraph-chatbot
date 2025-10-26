import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints
from typing import Annotated
from contextlib import asynccontextmanager

from src.run import chat

MAX_USER_MESSAGE_LEN = 200
CHAT_TIMEOUT = 30


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log.info("Server starting")
        # TODO
        yield
    
    except Exception as e:
        log.exception("Error while populating vector store: %s", e)


app = FastAPI(title="Chatbot API", lifespan=lifespan)


class UserMessage(BaseModel):
    text: Annotated[str, StringConstraints(max_length=MAX_USER_MESSAGE_LEN)]

class ChatResponse(BaseModel):
    text: str


@app.post("/chat", response_model=ChatResponse)
async def chat_route(user_message: UserMessage) -> ChatResponse:
    try:
        chat_response = await asyncio.wait_for(chat(user_message.text), timeout=CHAT_TIMEOUT)
        return ChatResponse(text=chat_response)
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout while waiting for chatbot response")

    except Exception as e:
        log.exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
