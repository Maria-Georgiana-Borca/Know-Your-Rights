import os
import time
import uuid
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from rag_and_prompting.rag import generate_with_retries, get_local_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Know Your Rights RAG API", version="1.0")

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            
    allow_methods=["*"],
    allow_headers=["*"],
)

#Global LLM client
llm_client = None

@app.on_event("startup")
async def on_startup():
    global llm_client
    llm_client = get_local_llm("rollama3-8b-instruct")
    logger.info("LLM client initialized successfully.")

def get_llm_client():
    if not llm_client:
        raise HTTPException(status_code=500, detail="LLM client not initialized")
    return llm_client

#Pydantic model for chat request and response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ChatChoice(BaseModel):
    index: int
    finish_reason: str
    message: ChatMessage

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[ChatChoice]

class ModelInfo(BaseModel):
    id: str

class ModelList(BaseModel):
    object: str
    data: List[ModelInfo]

@app.get("/v1/models")
async def list_models():
    """List available RAG models."""
    return ModelList(object="list", data=[ModelInfo(id="know-your-rights-rag")])

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(req: ChatRequest, llm=Depends(get_llm_client)):
    user_msg = req.messages[-1].content
    logger.info("Received user message: %s", user_msg)

    try:
        answer = generate_with_retries(user_msg, llm)
    except Exception as err:
        logger.error("LLM generation failed: %s", err, exc_info=True)
        raise HTTPException(500, "Internal generation error")

    response = ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        choices=[
            ChatChoice(
                index=0,
                finish_reason="stop",
                message=ChatMessage(role="assistant", content=answer)
            )
        ]
    )

    logger.info("Generated answer: %s", answer)
    return response