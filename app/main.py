from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.agent import AgentDeps, respond
from app.catalog import load_catalog
from app.config import CATALOG_PATH, EMBEDDINGS_PATH, EMBEDDING_IDS_PATH
from app.retrieval import HybridIndex
from app.schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading catalog and indexes...")
    catalog = load_catalog(CATALOG_PATH)
    index = HybridIndex.build(catalog, EMBEDDINGS_PATH, EMBEDDING_IDS_PATH)
    app.state.deps = AgentDeps(catalog=catalog, index=index)
    logger.info(f"ready: {len(catalog.products)} products, emb shape {index.embeddings.shape}")
    yield


app = FastAPI(title="SHL Conversational Recommender", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return respond(req.messages, app.state.deps)
