# DGX Spark Setup Guide

Step-by-step guide for setting up El Barto Serve, LM Studio, and AI coding tools on NVIDIA DGX Spark.

## Hardware & Network

| Device | IP | Role |
|---|---|---|
| DGX Spark | `SPARK_IP` | AI inference server |
| NAS (optional) | `NAS_IP` | Model storage |
| Workstation | `WORKSTATION_IP` | Development machine |

**Spark Specs:** Grace Blackwell GB10, 128GB unified memory, 3.7TB NVMe, DGX OS (Ubuntu 24.04), CUDA 13.0, Python 3.12

## 1. SSH Access

Generate an SSH key on your workstation (if you don't have one):

```bash
ssh-keygen -t ed25519
```

Copy it to the Spark:

```bash
ssh-copy-id YOUR_USER@SPARK_IP
```

Verify:

```bash
ssh YOUR_USER@SPARK_IP "hostname"
```

## 2. NAS Mounts (Optional)

If you have a NAS with NFS shares, configure them in `/etc/fstab` on the Spark:

```
NAS_IP:/volume1/ai-models  /mnt/nas/ai-models  nfs  defaults  0  0
```

If a mount isn't active, mount it manually:

```bash
sudo mount /mnt/nas/ai-models
```

**Tip:** Always copy models from NAS to local NVMe (`~/models/`) before loading them. NFS is ~120 MB/s; local NVMe is orders of magnitude faster for random reads during inference.

## 3. Install LM Studio CLI

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
export PATH="$HOME/.lmstudio/bin:$PATH"
# Add the export line to ~/.bashrc for persistence
```

### Start the API Server

```bash
lms server start --bind 0.0.0.0 --port 1234
```

This exposes an OpenAI-compatible API at `http://SPARK_IP:1234`.

### Managing Models

LM Studio expects GGUF models in `~/.lmstudio/models/`. To use models stored elsewhere, symlink them:

```bash
mkdir -p ~/.lmstudio/models/publisher/model-name
ln -sf ~/models/actual-model-file.gguf ~/.lmstudio/models/publisher/model-name/
```

Common commands:

```bash
lms ls                          # List available models
lms load <model-name>           # Load a model into memory
echo "y" | lms unload <model>   # Unload a model (piped to avoid interactive prompt)
lms ps                          # Show loaded models and status
```

## 4. Install El Barto Serve (Stable-DiffCoder)

```bash
cd ~
git clone https://github.com/NathanMaine/el-barto-serve.git
cd el-barto-serve
./setup-spark.sh
```

### Get the Model

Download from HuggingFace or copy from local storage:

```bash
# Option A: Download from HuggingFace
pip install huggingface-hub
huggingface-cli download ByteDance-Seed/Stable-DiffCoder-8B-Instruct \
  --local-dir ~/models/Stable-DiffCoder-8B-Instruct

# Option B: Copy from NAS/network storage
mkdir -p ~/models/Stable-DiffCoder-8B-Instruct
rsync -ah --progress /path/to/Stable-DiffCoder-8B-Instruct/ \
  ~/models/Stable-DiffCoder-8B-Instruct/
```

### Configure and Run

```bash
cp .env.example .env
# Edit .env:
#   ELBARTO_MODEL_PATH=/home/YOUR_USER/models/Stable-DiffCoder-8B-Instruct

source .venv/bin/activate
python server.py
```

El Barto serves at `http://SPARK_IP:8000` with an OpenAI-compatible `/v1/chat/completions` endpoint.

### Run in Background

```bash
nohup python server.py > /tmp/elbarto.log 2>&1 &
```

## 5. Pair with a Reasoning Model (Optional)

El Barto works great alongside a reasoning model like QwQ-32B for code review. Load one in LM Studio:

```bash
# Copy your GGUF model to local storage
mkdir -p ~/models/your-reasoning-model
cp /path/to/model.gguf ~/models/your-reasoning-model/

# Symlink into LM Studio
mkdir -p ~/.lmstudio/models/publisher/model-name
ln -sf ~/models/your-reasoning-model/model.gguf \
  ~/.lmstudio/models/publisher/model-name/

# Load it
lms load model-name
```

**Tip:** Reasoning models think extensively before answering — use them for complex code review, not casual chat. If a model gets stuck in a repetition loop, unload and reload it:

```bash
echo "y" | lms unload model-name
lms load model-name
```

## 6. Connect VS Code (Continue.dev)

Install the [Continue extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue) in VS Code.

Edit `~/.continue/config.yaml`:

```yaml
name: Local Config
version: 1.0.0
schema: v1
models:
  - name: El Barto (Stable-DiffCoder)
    provider: openai
    model: stable-diffcoder
    apiBase: http://SPARK_IP:8000/v1
    apiKey: not-needed
    requestOptions:
      extraBodyProperties:
        max_tokens: 384
  - name: Reasoning Model (Code Review)
    provider: openai
    model: your-model-name
    apiBase: http://SPARK_IP:1234/v1
    apiKey: lm-studio
    systemMessage: >
      You are a senior software engineer performing code review. Analyze code
      for bugs, security issues, performance problems, and maintainability
      concerns. Be concise and actionable. For each issue found, explain the
      problem, why it matters, and suggest a specific fix. Prioritize critical
      issues over style nits.
    requestOptions:
      extraBodyProperties:
        repeat_penalty: 1.1
        temperature: 0.3
```

**Important:** Cap `max_tokens` at ~384 for El Barto. Diffusion models fill their entire token budget — without a cap, you'll get gibberish noise after the actual code ends.

### Usage

- **Open Continue:** `Ctrl+L` in VS Code
- **Switch models:** Use the dropdown at the bottom of the Continue panel
- **Code review:** Select code, press `Ctrl+L`, choose your reasoning model, ask "Review this code"
- **Generate code:** Select context, choose El Barto, describe what you want

## Troubleshooting

### Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `setup-spark.sh` fails with `AttributeError: total_mem` | Bug in older versions | Update repo — fixed in commit `6536b38` (`total_mem` → `total_memory`) |
| PyTorch warns about "cuda capability 12.1" | GB10 exceeds PyTorch's officially supported range | Harmless — suppressed automatically by server.py since commit `6536b38` |
| `sudo` not working via remote SSH | No interactive terminal for password | Run sudo commands directly on the Spark terminal |
| NAS mount not active | Not auto-mounted after reboot | Run `sudo mount /mnt/nas/<share-name>` |
| GPU stuck at ~14W, throughput drops 50% | Known firmware bug | Full AC power cycle (unplug wall for 60 seconds) |
| Diffusion model outputs gibberish after code | Model fills remaining token budget with noise | Cap `max_tokens` to 384 in your client config |

### Useful Commands

```bash
# Check GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader

# Check what's loaded in LM Studio
lms ps

# Check El Barto logs
tail -f /tmp/elbarto.log

# Test El Barto API
curl http://SPARK_IP:8000/health

# Test LM Studio API
curl http://SPARK_IP:1234/v1/models

# Scan network for devices
nmap -sn YOUR_SUBNET/24
```

## Memory Budget

The Spark has 128GB unified memory shared between CPU and GPU:

| Component | Memory |
|---|---|
| Stable-DiffCoder (El Barto) | ~16.5 GB |
| 32B reasoning model Q8 (LM Studio) | ~34.8 GB |
| System / OS | ~5 GB |
| **Available** | **~71.7 GB** |

You can load additional models in LM Studio alongside El Barto, as long as total usage stays under ~120 GB.
