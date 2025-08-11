# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import uvicorn
import os
import re
import json
import shutil

from dotenv import load_dotenv

load_dotenv()

from rag_core import rag_pipeline
from vector_store import build_vector_store
from retrieval import RetrievalError
import uuid
import logging
from logging_utils import log_event
from tests.evaluation import run_retrieval_accuracy_fixture

app = FastAPI(title="Minimal RAG API")

# Request logging middleware with request_id and duration
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger = logging.getLogger(__name__)
    log_event(logger, "api.request.start", request_id=request_id, method=request.method, path=request.url.path)
    import time
    start_ns = time.perf_counter_ns()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = getattr(response, "status_code", 200)
        return response
    finally:
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        if response is not None:
            try:
                if 'X-Request-ID' not in response.headers:
                    response.headers['X-Request-ID'] = request_id
                # Include processing time for easy client-side consumption
                response.headers['X-Response-Time-ms'] = f"{duration_ms:.3f}"
            except Exception:
                pass
        log_event(
            logger,
            "api.request.end",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=round(duration_ms, 3),
            status_code=status_code,
        )

# CORS for local development UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Schemas ---
class QueryRequest(BaseModel):
    store_id: str = Field(description="Identifier of the vector store to query")
    question: str
    k: int = 3
    history: Optional[List[Tuple[str, str]]] = None
    include_raw_chunks: bool = False


class RetrievedChunk(BaseModel):
    summary: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: dict
    raw_chunks: List[dict] | None = None


def _validate_store_id(store_id: str) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", store_id or ""):
        raise HTTPException(status_code=400, detail="Invalid store_id format.")


def _store_paths(store_id: str) -> tuple[str, str]:
    base_store_dir = "chroma_store_versions"
    base_data_dir = os.path.join("data", "stores")
    persist_dir = os.path.join(base_store_dir, store_id)
    data_dir = os.path.join(base_data_dir, store_id)
    return persist_dir, data_dir


# --- Query endpoint ---
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, raw_request: Request):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    _validate_store_id(request.store_id)
    persist_dir, _ = _store_paths(request.store_id)
    if not os.path.isdir(persist_dir):
        raise HTTPException(status_code=404, detail="Vector store not found. Build it first.")

    try:
        answer, retrieved_chunks, metadata, raw_chunks = rag_pipeline(
            question=request.question,
            persist_dir=persist_dir,
            k=request.k,
            history=request.history,
        )
    except RetrievalError as e:
        log_event(
            logging.getLogger(__name__),
            "api.error",
            request_id=getattr(raw_request.state, "request_id", None),
            error_type=type(e).__name__,
            message=str(e),
            where="query_rag.retrieval",
        )
        raise HTTPException(status_code=502, detail="Upstream retrieval failed. Please try again later.")
    except Exception as e:
        log_event(
            logging.getLogger(__name__),
            "api.error",
            request_id=getattr(raw_request.state, "request_id", None),
            error_type=type(e).__name__,
            message=str(e),
            where="query_rag.pipeline",
        )
        raise HTTPException(status_code=502, detail="Upstream LLM or pipeline failed. Please try again later.")

    retrieved_chunks_models = [RetrievedChunk(**chunk) for chunk in retrieved_chunks]

    log_event(logging.getLogger(__name__), "api.query.success", store_id=request.store_id, k=request.k)
    metadata.pop("persist_dir", None)
    return QueryResponse(
        answer=answer,
        retrieved_chunks=retrieved_chunks_models,
        metadata=metadata,
        raw_chunks=[{"text": t, "metadata": m} for t, m in raw_chunks] if request.include_raw_chunks else None,
    )


# --- Store management ---
class CreateStoreRequest(BaseModel):
    store_id: str = Field(description="Identifier to create or rebuild")
    rebuild: bool = True


class StoreInfo(BaseModel):
    store_id: str
    path: str
    num_chunks: int
    embeddings_model: str


@app.post("/stores", response_model=StoreInfo, status_code=201)
def create_or_rebuild_store(req: CreateStoreRequest):
    _validate_store_id(req.store_id)
    persist_dir, data_dir = _store_paths(req.store_id)
    base_dir = os.path.dirname(persist_dir)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Enforce delete-before-build policy: if store exists, reject
    if os.path.isdir(persist_dir):
        log_event(logging.getLogger(__name__), "api.store.exists", store_id=req.store_id)
        raise HTTPException(status_code=409, detail="Store already exists. Delete it before building.")

    # Build in a temporary directory, then atomically replace target
    import time
    tmp_dir = os.path.join(base_dir, f".tmp_{req.store_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        path, num = build_vector_store(tmp_dir, data_dir=data_dir)
    except Exception as e:
        log_event(logging.getLogger(__name__), "api.error", error_type=type(e).__name__, message=str(e), where="create_or_rebuild_store.build")
        raise HTTPException(status_code=500, detail="Failed to build vector store")

    # Ensure target does not exist, then swap in
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.replace(tmp_dir, persist_dir)

    manifest = {
        "store_id": req.store_id,
        "path": persist_dir,
        "num_chunks": num,
        "embeddings_model": "text-embedding-3-small",
    }
    try:
        with open(os.path.join(persist_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
    except OSError:
        pass
    log_event(logging.getLogger(__name__), "api.store.created", store_id=req.store_id, num_chunks=num)
    return StoreInfo(**manifest)


@app.get("/stores", response_model=List[StoreInfo])
def list_stores():
    base_dir = "chroma_store_versions"
    stores: List[StoreInfo] = []
    if not os.path.isdir(base_dir):
        return stores
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        manifest_path = os.path.join(path, "manifest.json")
        num_chunks = 0
        model = "text-embedding-3-small"
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    num_chunks = int(m.get("num_chunks", 0))
                    model = m.get("embeddings_model", model)
            except Exception:
                pass
        stores.append(StoreInfo(store_id=name, path=path, num_chunks=num_chunks, embeddings_model=model))
    return stores


@app.get("/stores/{store_id}", response_model=StoreInfo)
def get_store(store_id: str):
    _validate_store_id(store_id)
    persist_dir, _ = _store_paths(store_id)
    if not os.path.isdir(persist_dir):
        raise HTTPException(status_code=404, detail="Store not found")
    manifest_path = os.path.join(persist_dir, "manifest.json")
    num_chunks = 0
    model = "text-embedding-3-small"
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
                num_chunks = int(m.get("num_chunks", 0))
                model = m.get("embeddings_model", model)
        except Exception:
            pass
    return StoreInfo(store_id=store_id, path=persist_dir, num_chunks=num_chunks, embeddings_model=model)


@app.delete("/stores/{store_id}")
def delete_store(store_id: str):
    _validate_store_id(store_id)
    persist_dir, data_dir = _store_paths(store_id)
    if not os.path.isdir(persist_dir) and not os.path.isdir(data_dir):
        raise HTTPException(status_code=404, detail="Store not found")
    shutil.rmtree(persist_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)
    log_event(logging.getLogger(__name__), "api.store.deleted", store_id=store_id)
    return {"deleted": store_id}


@app.post("/stores/{store_id}/documents")
async def upload_documents(store_id: str, files: List[UploadFile] = File(...)):
    _validate_store_id(store_id)
    _, data_dir = _store_paths(store_id)
    os.makedirs(data_dir, exist_ok=True)
    saved = []
    for f in files:
        if not f.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files supported")
        dest = os.path.join(data_dir, f.filename)
        with open(dest, "wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        saved.append(f.filename)
    return {"store_id": store_id, "saved": saved}


@app.get("/health")
def health():
    return {"status": "ok"}


class EvalRequest(BaseModel):
    k: int = 5
    max_questions: int | None = None


@app.post("/tests/retrieval-accuracy")
def api_retrieval_accuracy(req: EvalRequest):
    try:
        results = run_retrieval_accuracy_fixture(k=req.k, max_questions=req.max_questions)
        log_event(
            logging.getLogger(__name__),
            "api.eval.completed",
            k=req.k,
            total_questions=results.get("total_questions"),
        )
        return results
    except Exception as e:
        log_event(
            logging.getLogger(__name__),
            "api.error",
            error_type=type(e).__name__,
            message=str(e),
            where="api_retrieval_accuracy",
        )
        raise HTTPException(status_code=500, detail="Evaluation failed")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    log_event(
        logging.getLogger(__name__),
        "api.error",
        request_id=request_id,
        error_type="HTTPException",
        status_code=exc.status_code,
        message=str(exc.detail),
        path=request.url.path,
    )
    headers = {"X-Request-ID": request_id} if request_id else {}
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail, "request_id": request_id}, headers=headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    log_event(
        logging.getLogger(__name__),
        "api.error",
        request_id=request_id,
        error_type=type(exc).__name__,
        status_code=500,
        message=str(exc),
        path=request.url.path,
    )
    headers = {"X-Request-ID": request_id} if request_id else {}
    return JSONResponse(status_code=500, content={"message": "Internal server error", "request_id": request_id}, headers=headers)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
