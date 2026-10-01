"""
KV Cache Manager — Pre-allocated GPU memory pool for PagedAttention
====================================================================

This module provides a ``KVCacheManager`` that, once the model is loaded,
checks the remaining free VRAM and pre-allocates a large contiguous tensor
to serve as the physical page pool for the KV cache.

The pool is shaped as:
    [num_pages, 2, num_layers, page_size, num_kv_heads, head_dim]

where the ``2`` dimension holds K and V tensors respectively.

Architecture constants default to Qwen2.5-1.5B-Instruct (GQA):
    - num_hidden_layers  = 28
    - num_key_value_heads = 2
    - head_dim            = 128

Usage
-----
Call ``KVCacheManager.initialize()`` **after** the model has been loaded
onto the GPU so that the free VRAM measurement is accurate::

    from kv_cache_manager import KVCacheManager

    model = load_my_model()                   # eats ~0.9 GB
    kv_mgr = KVCacheManager.initialize()      # claims most of the rest
    print(kv_mgr)

The manager is deliberately minimal right now — it only handles pool
allocation.  Slot allocation, page tables, and cache read/write ops
will be added incrementally as PagedAttention understanding deepens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Model architecture defaults (Qwen2.5-1.5B-Instruct)
# ---------------------------------------------------------------------------
DEFAULT_NUM_LAYERS    = 28
DEFAULT_NUM_KV_HEADS  = 2
DEFAULT_HEAD_DIM      = 128
DEFAULT_DTYPE         = torch.float16


@dataclass
class KVPoolStats:
    """Human-readable snapshot of the allocated KV pool."""

    num_pages: int
    page_size: int
    total_tokens: int
    pool_memory_mb: float
    free_vram_before_mb: float
    free_vram_after_mb: float
    device: str

    def __str__(self) -> str:
        return (
            f"KVPoolStats(\n"
            f"  device             = {self.device}\n"
            f"  num_pages          = {self.num_pages:,}\n"
            f"  page_size          = {self.page_size} tokens/page\n"
            f"  total_tokens       = {self.total_tokens:,}\n"
            f"  pool_memory        = {self.pool_memory_mb:.1f} MB\n"
            f"  free_vram_before   = {self.free_vram_before_mb:.1f} MB\n"
            f"  free_vram_after    = {self.free_vram_after_mb:.1f} MB\n"
            f")"
        )


class KVCacheManager:
    """Pre-allocates a contiguous GPU tensor as the physical KV cache pool.

    Parameters
    ----------
    pool : torch.Tensor
        The raw page pool with shape
        ``[num_pages, 2, num_layers, page_size, num_kv_heads, head_dim]``.
    stats : KVPoolStats
        Allocation diagnostics.
    """

    def __init__(self, pool: torch.Tensor, stats: KVPoolStats) -> None:
        self.pool = pool
        self.stats = stats

    # ------------------------------------------------------------------
    # Factory — the main entry-point
    # ------------------------------------------------------------------

    @classmethod
    def initialize(
        cls,
        *,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
        head_dim: int = DEFAULT_HEAD_DIM,
        dtype: torch.dtype = DEFAULT_DTYPE,
        page_size: int = 16,
        vram_fraction: float = 0.90,
        reserve_mb: float = 128.0,
        device: str = "cuda",
    ) -> "KVCacheManager":
        """Allocate the KV cache pool from remaining free VRAM.

        Parameters
        ----------
        num_layers : int
            Number of transformer layers.
        num_kv_heads : int
            Number of key-value attention heads (GQA groups).
        head_dim : int
            Dimensionality of each attention head.
        dtype : torch.dtype
            Data type for the cache entries (default fp16 = 2 bytes).
        page_size : int
            Number of tokens per page (default 16, matches vLLM).
        vram_fraction : float
            Fraction of *free* VRAM to claim (0.0-1.0).
        reserve_mb : float
            Megabytes of free VRAM to keep aside as a safety margin on top
            of ``vram_fraction``.  This prevents OOM from PyTorch's internal
            fragmentation and CUDA context overhead.
        device : str
            CUDA device string.

        Returns
        -------
        KVCacheManager
            Ready-to-use manager with the pool tensor allocated.

        Raises
        ------
        RuntimeError
            If CUDA is unavailable or there isn't enough VRAM for even one
            page.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available — KVCacheManager requires a GPU.")

        # -- Measure free VRAM ------------------------------------------------
        torch.cuda.synchronize(device)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        free_vram_before_mb = free_bytes / (1024 ** 2)

        # -- Compute per-page memory cost -------------------------------------
        # Each page stores K and V for every layer:
        #   shape per page: [2, num_layers, page_size, num_kv_heads, head_dim]
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        elements_per_page = 2 * num_layers * page_size * num_kv_heads * head_dim
        bytes_per_page = elements_per_page * bytes_per_element

        # -- Decide how many pages we can fit ---------------------------------
        budget_bytes = free_bytes * vram_fraction - reserve_mb * (1024 ** 2)
        if budget_bytes <= 0:
            raise RuntimeError(
                f"Not enough free VRAM to allocate a KV cache pool. "
                f"Free: {free_vram_before_mb:.1f} MB, "
                f"reserve: {reserve_mb:.0f} MB."
            )

        num_pages = int(math.floor(budget_bytes / bytes_per_page))
        if num_pages < 1:
            raise RuntimeError(
                f"Cannot fit even a single page ({bytes_per_page / 1024:.1f} KB) "
                f"into available budget ({budget_bytes / (1024**2):.1f} MB)."
            )

        # -- Allocate ---------------------------------------------------------
        pool = torch.zeros(
            (num_pages, 2, num_layers, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        free_vram_after_mb = free_after / (1024 ** 2)

        pool_memory_mb = pool.nelement() * bytes_per_element / (1024 ** 2)

        stats = KVPoolStats(
            num_pages=num_pages,
            page_size=page_size,
            total_tokens=num_pages * page_size,
            pool_memory_mb=pool_memory_mb,
            free_vram_before_mb=free_vram_before_mb,
            free_vram_after_mb=free_vram_after_mb,
            device=device,
        )

        print(f"[KVCacheManager] Allocated {num_pages:,} pages "
              f"({pool_memory_mb:.1f} MB) — "
              f"{num_pages * page_size:,} tokens capacity")

        return cls(pool, stats)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_pages(self) -> int:
        return self.stats.num_pages

    @property
    def page_size(self) -> int:
        return self.stats.page_size

    @property
    def total_tokens(self) -> int:
        return self.stats.total_tokens

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(num_pages={self.num_pages:,}, "
            f"page_size={self.page_size}, "
            f"pool={self.pool.shape}, "
            f"dtype={self.pool.dtype})"
        )

    def __str__(self) -> str:
        return str(self.stats)
