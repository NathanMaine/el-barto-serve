# El Barto Serve

> *"I didn't do it. Nobody saw me do it. You can't prove anything."*
>
> — Bart Simpson, on how diffusion models generate code

OpenAI-compatible API server for [Stable-DiffCoder](https://huggingface.co/ByteDance-Seed/Stable-DiffCoder-8B-Instruct) — ByteDance's mask-diffusion code LLM that spray-paints code through iterative denoising instead of boring left-to-right token generation.

Built for the **NVIDIA DGX Spark** (Grace Blackwell GB10, 128GB unified memory), but runs on any CUDA GPU.

## Why This Exists

Stable-DiffCoder-8B-Instruct tops the benchmarks for 8B code models — beating Qwen2.5-Coder, CodeLlama, and every other diffusion LLM. But it uses a **non-standard diffusion inference pipeline** that no existing serving framework supports (not vLLM, not Ollama, not TensorRT-LLM).

El Barto wraps the custom diffusion generation in a standard **`/v1/chat/completions`** endpoint so you can use it with:

- **[Continue.dev](https://continue.dev)** in VS Code
- **[Open WebUI](https://github.com/open-webui/open-webui)**
- **curl**, Python scripts, or any OpenAI-compatible client

```
[Your Mac / VS Code]              [DGX Spark]
       |                                |
  Continue.dev  ----HTTP:8000---->  El Barto Serve
  Open WebUI                        (Stable-DiffCoder)
  curl
```

## Quick Start

### Option A: Native Install (DGX Spark)

```bash
git clone https://github.com/NathanMaine/el-barto-serve.git
cd el-barto-serve

# Automated setup (creates venv, installs CUDA 13.0 PyTorch, deps)
./setup-spark.sh

# Activate and run
source .venv/bin/activate
python server.py
```

### Option B: Docker (NGC Container)

```bash
docker build -t el-barto-serve .
docker run -it --gpus all \
  -p 8000:8000 \
  -e ELBARTO_MODEL_PATH=/models/Stable-DiffCoder-8B-Instruct \
  -v /path/to/your/model:/models/Stable-DiffCoder-8B-Instruct \
  el-barto-serve
```

### Option C: Other CUDA GPUs

```bash
git clone https://github.com/NathanMaine/el-barto-serve.git
cd el-barto-serve
python -m venv .venv && source .venv/bin/activate
pip install torch  # Standard PyTorch for your GPU
pip install -r requirements.txt
python server.py
```

## Usage

### Test with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stable-diffcoder",
    "messages": [{"role": "user", "content": "Write a binary search in Python"}],
    "temperature": 0.0
  }'
```

### Connect from VS Code (Continue.dev)

1. Install the [Continue extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
2. Open Continue settings (`~/.continue/config.json`)
3. Add El Barto as a model:

```json
{
  "models": [
    {
      "title": "El Barto (Stable-DiffCoder)",
      "provider": "openai",
      "model": "stable-diffcoder",
      "apiBase": "http://YOUR_SPARK_IP:8000/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

See [examples/continue-config.json](examples/continue-config.json) for the full config.

## Configuration

All settings via environment variables (or `.env` file — copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ELBARTO_MODEL_PATH` | `ByteDance-Seed/Stable-DiffCoder-8B-Instruct` | Local path or HuggingFace model ID |
| `ELBARTO_HOST` | `0.0.0.0` | Bind address |
| `ELBARTO_PORT` | `8000` | Server port |
| `ELBARTO_STEPS` | `256` | Diffusion denoising steps (more = higher quality, slower) |
| `ELBARTO_GEN_LENGTH` | `512` | Max output tokens |
| `ELBARTO_BLOCK_LENGTH` | `4` | Block diffusion granularity |
| `ELBARTO_THRESHOLD` | `None` | Early stopping confidence (0.0-1.0); lower = faster |
| `ELBARTO_REMASKING` | `low_confidence` | Remasking strategy (`low_confidence` or `random`) |

### Tuning for Speed vs Quality

```bash
# Maximum quality (slow — 512 steps, no early stopping)
ELBARTO_STEPS=512 ELBARTO_THRESHOLD= python server.py

# Balanced (default)
ELBARTO_STEPS=256 python server.py

# Fast mode (fewer steps + early stopping)
ELBARTO_STEPS=128 ELBARTO_THRESHOLD=0.5 python server.py

# Fastest (aggressive early stopping — "eat my shorts" mode)
ELBARTO_STEPS=64 ELBARTO_THRESHOLD=0.3 python server.py
```

## DGX Spark Notes

Things we learned so the Spark doesn't have a cow:

- **Flash Attention is broken on SM 12.1** — El Barto auto-patches to use PyTorch's native SDPA, which is actually ~2% faster on Blackwell with cuDNN 9.13+. No action needed.
- **CUDA 13.0 required** — The setup script handles this. Don't use standard PyTorch pip wheels.
- **Python 3.12.x recommended** — 3.13.x has known issues on Spark.
- **Unified memory is your friend** — The 15GB model leaves ~113GB free. No CPU-to-GPU transfer overhead.
- **Static memory footprint** — Unlike autoregressive models with growing KV caches, diffusion operates on fixed-size tensors. No OOM surprises mid-generation.
- **Performance bug workaround** — If throughput suddenly drops 50% with GPU stuck at ~14W, do a full AC power cycle (unplug from wall for 60 seconds). This is a known firmware issue.

## How Diffusion Code Generation Works

Traditional LLMs generate code left-to-right, one token at a time. Stable-DiffCoder works differently:

1. **Mask** — Start with the full output length filled with `[MASK]` tokens
2. **Denoise** — Iteratively predict and unmask the most confident tokens
3. **Refine** — Each step reveals more of the code, like graffiti appearing on a wall

This "any-order" generation means the model can consider the full structure simultaneously, making it naturally better at maintaining syntax, matching brackets, and reasoning about code structure.

```
Step 0:   [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] ...
Step 32:  def    [MASK] sort   [MASK] [MASK] :      ...
Step 64:  def    quick  sort   (      arr    :      ...
Step 128: def    quick  sort   (      arr    :    list) -> list: ...
```

## API Reference

### POST `/v1/chat/completions`

Standard OpenAI chat completions format. Supports both streaming and non-streaming.

Extra fields for diffusion control (pass via request body):

```json
{
  "steps": 256,
  "gen_length": 512,
  "block_length": 4,
  "threshold": null,
  "remasking": "low_confidence"
}
```

### GET `/v1/models`

List available models.

### GET `/health`

Health check with model status and device info.

## Benchmarks

Stable-DiffCoder-8B-Instruct vs other ~8B models:

| Model | HumanEval | MBPP | MHPP | BigCodeBench |
|-------|-----------|------|------|--------------|
| Qwen2.5-Coder-7B-Instruct | **88.4** | 83.5 | 26.7 | 48.8 |
| Seed-Coder-8B-Instruct | 84.8 | 85.2 | 36.2 | 53.3 |
| **Stable-DiffCoder-8B-Instruct** | 86.6 | **85.7** | **42.4** | **54.8** |

## License

MIT

---

*El Barto was here.*
