# To be implemented
# src/inference/api.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
import time

from src.inference.generator import (
    load_model,
    load_tokenizer,
    generate,
    generate_python,
    generate_sql,
    generate_completion,
)


# ── Global model state ────────────────────────────────────────
# Loaded once at startup — reused for every request
state = {}


# ── Startup / shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load on startup
    print("\n  Loading MiniGPT model...")
    state["model"], state["device"] = load_model()
    state["tokenizer"] = load_tokenizer()
    print("  Model ready — server accepting requests\n")
    yield
    # Cleanup on shutdown
    state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model unloaded")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "MiniGPT API",
    description = "GPT model built from scratch — Python, SQL, code assistance",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response schemas ────────────────────────────────

class PythonRequest(BaseModel):
    description : str = Field(
        ...,
        example     = "calculate the fibonacci sequence up to n",
        description = "Plain English description of the Python function",
    )
    max_tokens  : int   = Field(default=200, ge=10,  le=500)
    temperature : float = Field(default=0.7, ge=0.1, le=1.5)

class SQLRequest(BaseModel):
    question    : str = Field(
        ...,
        example     = "find all orders placed after 2023",
        description = "Plain English question",
    )
    context     : str = Field(
        ...,
        example     = "CREATE TABLE orders (id INT, date TEXT, amount FLOAT)",
        description = "SQL table schema(s) as CREATE TABLE statements",
    )
    max_tokens  : int   = Field(default=150, ge=10,  le=400)
    temperature : float = Field(default=0.4, ge=0.1, le=1.0)

class CompletionRequest(BaseModel):
    prompt      : str = Field(
        ...,
        example     = "def merge_sort(arr):",
        description = "Partial code to complete",
    )
    max_tokens  : int   = Field(default=150, ge=10,  le=500)
    temperature : float = Field(default=0.7, ge=0.1, le=1.5)

class GenerateRequest(BaseModel):
    prompt      : str = Field(..., description="Raw prompt text")
    max_tokens  : int   = Field(default=200, ge=10,  le=500)
    temperature : float = Field(default=0.8, ge=0.1, le=2.0)
    top_k       : int   = Field(default=40,  ge=1,   le=100)

class GenerationResponse(BaseModel):
    result      : str
    tokens      : int
    time_ms     : float
    model       : str = "minigpt-35m"


# ── Helper ────────────────────────────────────────────────────
def timed_generate(fn, *args, **kwargs) -> GenerationResponse:
    t0     = time.time()
    result = fn(*args, **kwargs)
    dt_ms  = (time.time() - t0) * 1000
    tokens = len(state["tokenizer"].encode(result).ids)
    return GenerationResponse(
        result  = result,
        tokens  = tokens,
        time_ms = round(dt_ms, 1),
    )


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name"    : "MiniGPT API",
        "status"  : "running",
        "model"   : "minigpt-35m",
        "device"  : state.get("device", "unknown"),
        "routes"  : ["/python", "/sql", "/complete", "/generate", "/health"],
    }


@app.get("/health")
def health():
    gpu_mb = None
    if torch.cuda.is_available():
        gpu_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
    return {
        "status" : "ok",
        "device" : state.get("device", "unknown"),
        "gpu_mb" : gpu_mb,
    }


@app.post("/python", response_model=GenerationResponse)
def python_endpoint(req: PythonRequest):
    """Generate a Python function from a plain English description."""
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    try:
        return timed_generate(
            generate_python,
            state["model"],
            state["tokenizer"],
            req.description,
            state["device"],
            temperature = req.temperature,
            max_tokens  = req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/sql", response_model=GenerationResponse)
def sql_endpoint(req: SQLRequest):
    """Generate a SQL query from a question and table context."""
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    try:
        return timed_generate(
            generate_sql,
            state["model"],
            state["tokenizer"],
            req.question,
            req.context,
            state["device"],
            temperature = req.temperature,
            max_tokens  = req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/complete", response_model=GenerationResponse)
def complete_endpoint(req: CompletionRequest):
    """Complete a partial code snippet."""
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    try:
        return timed_generate(
            generate_completion,
            state["model"],
            state["tokenizer"],
            req.prompt,
            state["device"],
            temperature = req.temperature,
            max_tokens  = req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate", response_model=GenerationResponse)
def generate_endpoint(req: GenerateRequest):
    """Raw generation — full control over prompt and parameters."""
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    try:
        return timed_generate(
            generate,
            state["model"],
            state["tokenizer"],
            req.prompt,
            req.max_tokens,
            req.temperature,
            req.top_k,
            state["device"],
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.inference.api:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = False,
        app_dir  = ROOT,
    )