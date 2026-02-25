# ============================================================================
# El Barto Serve - Docker image for NVIDIA DGX Spark
#
# Based on NVIDIA's NGC PyTorch container (validated for Blackwell/SM 12.1).
# ============================================================================
FROM nvcr.io/nvidia/pytorch:25.09-py3

LABEL maintainer="NathanMaine"
LABEL description="El Barto Serve - Diffusion code model server for DGX Spark"

WORKDIR /app

# NGC container ships a newer transformers — pin to 4.46.2 as required
# Also pin numpy < 2.0 for compatibility with NGC container PyTorch extensions
RUN pip install --no-cache-dir \
    "transformers==4.46.2" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0" \
    "numpy<2.0"

COPY server.py .
COPY patches/ ./patches/

# Run as non-root user
RUN useradd -m appuser
USER appuser

# Default: serve on port 8000
EXPOSE 8000

ENV ELBARTO_HOST=0.0.0.0
ENV ELBARTO_PORT=8000

# Pre-download model on build (optional — uncomment to bake into image)
# ENV ELBARTO_MODEL_PATH=ByteDance-Seed/Stable-DiffCoder-8B-Instruct
# RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('${ELBARTO_MODEL_PATH}', trust_remote_code=True); \
#     AutoModelForCausalLM.from_pretrained('${ELBARTO_MODEL_PATH}', trust_remote_code=True)"

CMD ["python", "server.py"]
