from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def is_low_resource_mode() -> bool:
    return os.environ.get("TANTRA_LOW_RESOURCE", "0") in {"1", "true", "True"}


def apply_low_resource_settings() -> None:
    if not is_low_resource_mode():
        return
    try:
        # Limit BLAS/OMP threads
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Limit PyTorch threads
        import torch
        try:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        logger.info("Low-resource mode applied: limited CPU threads and disabled parallel tokenizers")
    except Exception as e:
        logger.warning(f"Failed to apply low-resource settings: {e}")


