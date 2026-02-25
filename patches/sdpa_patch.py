"""
Flash Attention -> SDPA patch for NVIDIA DGX Spark (SM 12.1).

Flash Attention pre-compiled wheels don't exist for linux_aarch64 + sm121,
and building from source is fragile. PyTorch's native Scaled Dot-Product
Attention (SDPA) with cuDNN 9.13+ is actually ~2% faster on Blackwell anyway.

This module monkey-patches transformers to avoid flash-attn imports and
force the SDPA backend. Import it before loading the model:

    import patches.sdpa_patch  # noqa: F401
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(...)
"""

import logging
import sys

logger = logging.getLogger("el-barto.patch")


def _apply_sdpa_patch():
    """Force transformers to use SDPA instead of flash-attention."""
    try:
        import transformers.utils
        # Tell transformers that flash_attn is not available
        if hasattr(transformers.utils, "is_flash_attn_2_available"):
            transformers.utils.is_flash_attn_2_available = lambda: False
            logger.info("Patched: is_flash_attn_2_available -> False (using SDPA)")

        if hasattr(transformers.utils, "is_flash_attn_available"):
            transformers.utils.is_flash_attn_available = lambda: False

        # Also patch the import check module if it exists
        if hasattr(transformers.utils, "import_utils"):
            import_utils = transformers.utils.import_utils
            if hasattr(import_utils, "_is_flash_attn_2_available"):
                import_utils._is_flash_attn_2_available = False
            if hasattr(import_utils, "is_flash_attn_2_available"):
                import_utils.is_flash_attn_2_available = lambda: False
    except ImportError:
        logger.warning("transformers not installed, skipping flash-attn patch")
        return

    # Block flash_attn from being imported at all (prevents stray imports)
    sys.modules["flash_attn"] = None
    sys.modules["flash_attn.flash_attn_interface"] = None
    logger.info("Blocked flash_attn module imports")


# Auto-apply on import
_apply_sdpa_patch()
