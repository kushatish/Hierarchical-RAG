"""Run from rag_cag_project: uvicorn app.main:app --host 127.0.0.1"""
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
import hmac
import json
import os
import logging
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, Request
from pydantic import BaseModel, Field
from core.engine import Assistant
from core.operations import RateLimiter, RequestMetrics

MAX_UPLOAD = 10 * 1024 * 1024
limiter = RateLimiter(limit=int(os.getenv("RAG_RATE_LIMIT", "120")))
metrics = RequestMetrics()
logger = logging.getLogger("evidence_qa.requests")

@lru_cache(maxsize=1)
def get_assistant():
    return Assistant()

@asynccontextmanager
async def lifespan(app):
    if os.getenv("RAG_REQUIRE_API_KEY", "false").lower() == "true" and not os.getenv("RAG_API_KEY"):
        raise RuntimeError("RAG_API_KEY is required for this deployment")
    get_assistant()
    yield

app = FastAPI(title="Evidence QA", version="1.0.0", lifespan=lifespan)

def authorize(request: Request, x_api_key: str = Header(default="")):
    identity = request.client.host if request.client else "unknown"
    if not limiter.allow(identity):
        raise HTTPException(429, "Request rate limit exceeded", headers={"Retry-After": str(limiter.window)})
    key = os.getenv("RAG_API_KEY", "")
    if key and not hmac.compare_digest(x_api_key, key):
        raise HTTPException(401, "Invalid API key")

@app.middleware("http")
async def observe_request(request: Request, call_next):
    started = time.perf_counter()
    request_id = uuid4().hex
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", "unmatched")
        elapsed = (time.perf_counter() - started) * 1000
        metrics.record(request.method, route, status, elapsed)
        logger.info(json.dumps({"request_id": request_id, "method": request.method, "route": route, "status": status, "latency_ms": round(elapsed, 3)}))

@app.get("/ready")
def readiness():
    try:
        get_assistant().revision()
    except Exception as exc:
        raise HTTPException(503, "Document store unavailable") from exc
    return {"status": "ready"}

@app.get("/metrics", dependencies=[Depends(authorize)])
def request_metrics():
    return metrics.snapshot()

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    use_cache: bool = True
    semantic_cache: bool = False
    hierarchy: bool = True

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", dependencies=[Depends(authorize)])
def query_api(req: QueryRequest):
    try:
        return get_assistant().ask(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        # Do not expose model-provider responses, paths or credentials to clients.
        raise HTTPException(503, "Answer generation failed; check server configuration") from exc

@app.get("/documents", dependencies=[Depends(authorize)])
def documents():
    return get_assistant().list_documents()

@app.post("/documents", dependencies=[Depends(authorize)])
def upload_document(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, "Supported formats: PDF, TXT, MD")
    try:
        with TemporaryDirectory() as directory:
            path = Path(directory) / name
            size = 0
            with path.open("wb") as target:
                while block := file.file.read(64 * 1024):
                    size += len(block)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "Maximum upload size is 10 MiB")
                    target.write(block)
            return get_assistant().ingest_file(path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Could not extract document text; use a readable, unencrypted PDF or UTF-8 text") from exc
    finally:
        file.file.close()

@app.get("/documents/{doc_id}/pages/{page}", dependencies=[Depends(authorize)])
def source_page(doc_id: str, page: int, revision: int | None = None):
    current_revision, docs = get_assistant().snapshot()
    if revision is not None and revision != current_revision:
        raise HTTPException(409, "Document library changed; ask again for current citations")
    for doc in docs:
        if doc["id"] == doc_id:
            pages = json.loads(doc["pages"])
            if 1 <= page <= len(pages):
                return {"source": doc["name"], "page": page, "text": pages[page - 1]}
    raise HTTPException(404, "Document page not found")

@app.delete("/documents/{doc_id}", dependencies=[Depends(authorize)])
def delete_document(doc_id: str):
    if not get_assistant().delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": True}
