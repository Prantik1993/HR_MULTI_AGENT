from __future__ import annotations
import asyncio
from functools import partial
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.supervisor import graph
from app.cache.query_cache import cache

app = FastAPI(title="HR Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
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
    if cached_answer := cache.get(req.query):
        return ChatResponse(intent="cached", answer=cached_answer, sources=[], cached=True)
    try:
        messages = _build_messages(req.history, req.query)
        loop = asyncio.get_event_loop()
        invoke = partial(
            graph.invoke,
            {"messages": messages, "intent": "", "answer": "", "sources": []},
        )
        result = await loop.run_in_executor(None, invoke)
        answer  = result["answer"]
        sources = result["sources"]
        intent  = result["intent"]
        cache.set(req.query, answer)
        return ChatResponse(intent=intent, answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "cache_size": len(cache)}
