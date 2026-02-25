#!/usr/bin/env bash
# ============================================================================
# El Barto Serve - DGX Spark Setup Script
#
# Sets up the Python environment for running Stable-DiffCoder on DGX Spark
# (Grace Blackwell GB10, SM 12.1, CUDA 13.0, ARM64/aarch64).
# ============================================================================
set -euo pipefail

echo "======================================"
echo "  El Barto Serve - DGX Spark Setup"
echo "  \"Nobody saw me do it.\" - Bart"
echo "======================================"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.12.x first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: ${PYTHON_VERSION}"

if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
    echo "WARNING: Python 3.12.x is recommended for DGX Spark. You have ${PYTHON_VERSION}."
    echo "         Other versions may have compatibility issues with CUDA 13.0 wheels."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# Check for CUDA 13.0
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "CUDA version: ${CUDA_VERSION}"
else
    echo "WARNING: nvcc not found. Ensure CUDA 13.0 is installed and in PATH."
    echo "         Try: export PATH=/usr/local/cuda-13.0/bin:\$PATH"
fi

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
echo
echo "Creating virtual environment at ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip ..."
pip install --upgrade pip

# ---------------------------------------------------------------------------
# Install PyTorch for DGX Spark (CUDA 13.0, aarch64)
# ---------------------------------------------------------------------------
echo
echo "Installing PyTorch for DGX Spark (CUDA 13.0 / SM 12.1) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# ---------------------------------------------------------------------------
# Install project dependencies
# ---------------------------------------------------------------------------
echo
echo "Installing el-barto-serve dependencies ..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
echo
echo "Verifying PyTorch + CUDA ..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'Device:          {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'Memory:          {mem_gb:.1f} GB')
"

echo
echo "Verifying transformers ..."
python3 -c "
import transformers
print(f'transformers version: {transformers.__version__}')
assert transformers.__version__ == '4.46.2', f'Expected 4.46.2, got {transformers.__version__}'
print('OK - version matches requirement')
"

echo
echo "======================================"
echo "  Setup complete!"
echo "  Activate:  source ${VENV_DIR}/bin/activate"
echo "  Run:       python server.py"
echo "  \"Eat my shorts.\" - Bart"
echo "======================================"
