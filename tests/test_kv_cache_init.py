"""
test_kv_cache_init.py
=====================
Smoke-test for the KVCacheManager that is wired into TargetEngine.

What this test does
-------------------
1. Initializes a TargetEngine via ``TargetEngine.from_pretrained()``.
   - Loads Qwen2.5-1.5B-Instruct in 4-bit NF4.
   - Internally calls ``KVCacheManager.initialize()`` to pre-allocate
     the KV cache pool from remaining free VRAM.
2. Asserts that the KVCacheManager was created and its pool is live on
   the GPU.
3. Prints a full VRAM breakdown so you can see exactly how much GPU
   memory the engine + pool are consuming.

Run with:
    python tests/test_kv_cache_init.py
or via pytest:
    pytest tests/test_kv_cache_init.py -s
"""

from __future__ import annotations

import sys
import os

import torch

# ---------------------------------------------------------------------------
# Path setup — make target-model importable regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_MODEL_DIR = os.path.join(PROJECT_ROOT, "target-model")

sys.path.insert(0, TARGET_MODEL_DIR)
sys.path.insert(0, PROJECT_ROOT)

from target_engine import TargetEngine       # noqa: E402  (after sys.path setup)
from kv_cache_manager import KVCacheManager  # noqa: E402


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_kv_cache_manager_initialize() -> None:
    """Initialize the TargetEngine and verify the KVCacheManager is live."""

    # ---- 1. Initialize the engine (loads model + allocates KV pool) --------
    print("\n[TEST] Initializing TargetEngine (this will load the model) ...")
    engine = TargetEngine.from_pretrained()

    # ---- 2. Basic assertions on the KVCacheManager -------------------------
    kv_mgr = engine._kv_cache_manager

    assert kv_mgr is not None, \
        "KVCacheManager was not attached to the engine — check from_pretrained()"

    assert isinstance(kv_mgr, KVCacheManager), \
        f"Expected KVCacheManager, got {type(kv_mgr)}"

    assert kv_mgr.num_pages >= 1, \
        f"Pool must have at least 1 page, got {kv_mgr.num_pages}"

    assert kv_mgr.pool.is_cuda, \
        "KV pool tensor is not on the GPU"

    assert not kv_mgr.pool.is_meta, \
        "KV pool tensor is a meta tensor — allocation did not happen"

    print("\n[TEST] KVCacheManager assertions passed ✓")
    print(kv_mgr)  # prints the full KVPoolStats

    # ---- 3. VRAM report ----------------------------------------------------
    torch.cuda.synchronize()
    allocated_bytes      = torch.cuda.memory_allocated()
    reserved_bytes       = torch.cuda.memory_reserved()
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    allocated_gb = allocated_bytes / 1e9
    reserved_gb  = reserved_bytes  / 1e9
    free_gb      = free_bytes      / 1e9
    total_gb     = total_bytes     / 1e9
    used_gb      = total_gb - free_gb

    print("\n" + "=" * 55)
    print("  VRAM Usage Report")
    print("=" * 55)
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM       : {total_gb:.2f} GB")
    print(f"  Currently in use : {used_gb:.2f} GB  ({used_gb / total_gb * 100:.1f} %)")
    print(f"  Free             : {free_gb:.2f} GB")
    print(f"  PyTorch allocated: {allocated_gb:.2f} GB  (tensors only)")
    print(f"  PyTorch reserved : {reserved_gb:.2f} GB  (cached allocator)")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point (also works standalone without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_kv_cache_manager_initialize()
    print("\n[TEST] Done.")
