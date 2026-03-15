"""
api/main.py
------------
FIXES:
  [06] asyncio.get_event_loop() → asyncio.get_running_loop()
  [07] Empty query now rejected with 422 (Field min_length=1)
  [08] Full structured logging added
"""
from __future__ import annotations
import asyncio
import time
from functools import partial

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.supervisor import graph
from app.cache.query_cache import cache
from app.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="HR Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)   # FIX [07]
    history: list[dict] = []


class ChatResponse(BaseModel):
    intent: str
    answer: str
    sources: list[str]
    cached: bool = False


def _build_messages(history: list[dict], query: str) -> list:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=query))
    return msgs


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    logger.info("POST /chat | query=%r history_turns=%d", req.query[:80], len(req.history))

    if cached_answer := cache.get(req.query):
        logger.info("POST /chat | cache HIT — returning cached answer")
        return ChatResponse(intent="cached", answer=cached_answer, sources=[], cached=True)

    try:
        messages = _build_messages(req.history, req.query)
        loop = asyncio.get_running_loop()         # FIX [06]: was get_event_loop()
        invoke = partial(
            graph.invoke,
            {"messages": messages, "intent": "", "answer": "", "sources": []},
        )
        result = await loop.run_in_executor(None, invoke)

        answer  = result["answer"]
        sources = result["sources"]
        intent  = result["intent"]

        cache.set(req.query, answer)

        elapsed = time.perf_counter() - t0
        logger.info(
            "POST /chat | intent=%s sources=%s answer_len=%d elapsed=%.2fs",
            intent, sources, len(answer), elapsed,
        )
        return ChatResponse(intent=intent, answer=answer, sources=sources)

    except Exception as e:
        logger.exception("POST /chat | unhandled error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict:
    logger.debug("GET /health")
    return {"status": "ok", "cache_size": len(cache)}
