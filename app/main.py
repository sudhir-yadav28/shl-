from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title="SHL Conversational Recommender", version="0.1.0")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    return ChatResponse(
        reply=f"[stub] received: {last_user[:80]}",
        recommendations=[],
        end_of_conversation=False,
    )
