from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

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


LANDING_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>SHL Conversational Recommender</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px; margin: 60px auto; padding: 0 24px; color: #1a1a1a; line-height: 1.55; }
  h1 { font-size: 1.5rem; margin-bottom: 0.2em; }
  code, pre { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  pre { background: #f5f5f7; padding: 14px 16px; border-radius: 6px; overflow-x: auto; font-size: 0.9rem; }
  a { color: #0366d6; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .endpoint { margin: 12px 0; }
  .method { display: inline-block; background: #0366d6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
  .method.post { background: #28a745; }
  footer { margin-top: 40px; color: #6a737d; font-size: 0.85rem; }
</style></head>
<body>
<h1>SHL Conversational Assessment Recommender</h1>
<p>A stateless API that takes a hiring manager from a vague intent to a grounded shortlist of SHL Individual Test Solutions through dialogue.</p>

<div class="endpoint"><span class="method">GET</span> <a href="/health"><code>/health</code></a> — readiness check</div>
<div class="endpoint"><span class="method post">POST</span> <code>/chat</code> — conversation turn (stateless; send full history)</div>
<div class="endpoint"><span class="method">GET</span> <a href="/docs"><code>/docs</code></a> — interactive API explorer (try the agent from your browser)</div>

<h2 style="font-size: 1.1rem; margin-top: 32px;">Try it from the terminal</h2>
<pre>curl -X POST https://shl-recommender-ab5b.onrender.com/chat \\
  -H 'Content-Type: application/json' \\
  -d '{"messages":[{"role":"user","content":"Hiring a senior Java backend engineer with Spring SQL AWS Docker"}]}'</pre>

<footer>Built for the SHL Labs AI Intern take-home. <a href="https://github.com/sudhir-yadav28/shl-">Source on GitHub</a>.</footer>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return respond(req.messages, app.state.deps)
