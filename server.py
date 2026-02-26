"""
El Barto Serve - OpenAI-compatible API for Stable-DiffCoder diffusion code models.

Wraps ByteDance's Stable-DiffCoder mask-diffusion inference in a standard
/v1/chat/completions endpoint so any OpenAI-compatible client (Continue.dev,
Open WebUI, curl, etc.) can use it.

Designed for NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified memory).
"""

import json
import os
import time
import uuid
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Optional

# Suppress harmless PyTorch warning about GB10 (SM 12.1) exceeding the officially
# supported CUDA capability range. Everything works correctly with CUDA 13.0 wheels.
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

# Patch flash-attn -> SDPA before any transformers imports (required for DGX Spark SM 12.1)
import patches.sdpa_patch  # noqa: F401

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger("el-barto")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Configuration (env vars with sane defaults)
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("ELBARTO_MODEL_PATH", "ByteDance-Seed/Stable-DiffCoder-8B-Instruct")
MODEL_REVISION = os.getenv("ELBARTO_MODEL_REVISION", None)
HOST = os.getenv("ELBARTO_HOST", "0.0.0.0")
PORT = int(os.getenv("ELBARTO_PORT", "8000"))
API_KEY = os.getenv("ELBARTO_API_KEY", "")
DEFAULT_STEPS = int(os.getenv("ELBARTO_STEPS", "256"))
DEFAULT_GEN_LENGTH = int(os.getenv("ELBARTO_GEN_LENGTH", "512"))
DEFAULT_BLOCK_LENGTH = int(os.getenv("ELBARTO_BLOCK_LENGTH", "4"))
DEFAULT_THRESHOLD = os.getenv("ELBARTO_THRESHOLD", None)
DEFAULT_REMASKING = os.getenv("ELBARTO_REMASKING", "low_confidence")

# Parse threshold (None means use all steps, float means early stopping)
if DEFAULT_THRESHOLD is not None:
    DEFAULT_THRESHOLD = float(DEFAULT_THRESHOLD)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
model = None
tokenizer = None


def load_model():
    """Load model and tokenizer onto the best available device."""
    global model, tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available — running on CPU (this will be very slow)")

    load_kwargs = {"trust_remote_code": True}
    if MODEL_REVISION:
        load_kwargs["revision"] = MODEL_REVISION
        logger.info("Pinned to revision: %s", MODEL_REVISION)

    logger.info("Loading tokenizer from %s ...", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **load_kwargs)

    logger.info("Loading model from %s (bf16) ...", MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        **load_kwargs,
    ).to(device).eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(
        "Model loaded: %.1fB params on %s | Memory: %.1f GB",
        param_count,
        device,
        torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down El Barto.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="El Barto Serve",
    description="OpenAI-compatible API for Stable-DiffCoder diffusion code models",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Authentication (optional — set ELBARTO_API_KEY to enable)
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if API_KEY and (not key or key.replace("Bearer ", "") != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI chat completions format)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class DiffusionParams(BaseModel):
    """Extra diffusion-specific params, passed via the 'extra_body' field."""
    steps: Optional[int] = None
    gen_length: Optional[int] = None
    block_length: Optional[int] = None
    threshold: Optional[float] = None
    remasking: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "stable-diffcoder"
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stream: bool = False
    # Diffusion-specific overrides (clients can pass via extra_body)
    steps: Optional[int] = None
    gen_length: Optional[int] = None
    block_length: Optional[int] = None
    threshold: Optional[float] = None
    remasking: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
_UNSET = object()


def generate_response(
    messages: list[ChatMessage],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    steps: Optional[int] = None,
    gen_length: Optional[int] = None,
    block_length: Optional[int] = None,
    threshold: object = _UNSET,
    remasking: Optional[str] = None,
) -> tuple[str, int, int]:
    """Run diffusion generation and return (text, prompt_tokens, completion_tokens)."""
    # Build chat prompt using the model's chat template
    chat_messages = [{"role": m.role, "content": m.content} for m in messages]
    prompt_text = tokenizer.apply_chat_template(
        chat_messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(prompt_text)["input_ids"]
    prompt_tokens = len(input_ids)
    input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)

    # Resolve generation parameters
    _steps = steps or DEFAULT_STEPS
    _gen_length = max_tokens or gen_length or DEFAULT_GEN_LENGTH
    _block_length = block_length or DEFAULT_BLOCK_LENGTH
    _remasking = remasking or DEFAULT_REMASKING
    _threshold = DEFAULT_THRESHOLD if threshold is _UNSET else threshold

    # Ensure gen_length is divisible by block_length
    _gen_length = (_gen_length // _block_length) * _block_length
    if _gen_length == 0:
        _gen_length = _block_length

    # Ensure steps is divisible by the number of blocks
    num_blocks = _gen_length // _block_length
    _steps = max(num_blocks, (_steps // num_blocks) * num_blocks)

    logger.info(
        "Generating: steps=%d gen_length=%d block_length=%d threshold=%s remasking=%s temp=%.2f",
        _steps, _gen_length, _block_length, _threshold, _remasking, temperature,
    )

    start = time.perf_counter()

    output_ids = model.generate(
        input_ids,
        steps=_steps,
        gen_length=_gen_length,
        block_length=_block_length,
        temperature=temperature,
        remasking=_remasking,
        tokenizer=tokenizer,
        shift=False,
        threshold=_threshold,
        eos_id=tokenizer.eos_token_id,
    )

    elapsed = time.perf_counter() - start

    # Decode only the generated portion
    generated_ids = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    logger.info(
        "Done: %d tokens in %.2fs (%.1f tok/s effective)",
        completion_tokens, elapsed,
        completion_tokens / elapsed if elapsed > 0 else 0,
    )

    return text, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
def stream_chunks(text: str, request_id: str, model_name: str):
    """Yield SSE chunks in OpenAI streaming format."""
    # Send the full text as a single chunk (diffusion generates all at once)
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": text},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Send the stop chunk
    stop_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        text, prompt_tokens, completion_tokens = generate_response(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            steps=request.steps,
            gen_length=request.gen_length,
            block_length=request.block_length,
            threshold=request.threshold,
            remasking=request.remasking,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    if request.stream:
        return StreamingResponse(
            stream_chunks(text, request_id, request.model),
            media_type="text/event-stream",
        )

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=text),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "stable-diffcoder",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "el-barto",
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "loading",
        "model": MODEL_PATH,
        "device": str(model.device) if model else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
