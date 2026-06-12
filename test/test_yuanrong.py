"""Integration tests for UcmYuanrongStore.

Requires a running yuanrong worker at the configured host:port.

Example (start a local worker first, then run):
    python -m pytest test/test_yuanrong.py -v
"""
import hashlib
import uuid

import pytest
import torch

from ucm.logger import init_logger
from ucm.store.yuanrong.connector import UcmYuanrongStore

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — point these at your running yuanrong worker
# ---------------------------------------------------------------------------
YUANRONG_CONFIG = {
    "host": "127.0.0.1",
    "port": 31501,
    "tensor_size_list": [1024],  # 1 tensor per block, 1024 bytes each
}

# A float32 tensor of 256 elements occupies 1024 bytes.
_TENSOR_SHAPE = (1, 256)
_TENSOR_DTYPE = torch.float32


def debug_diagnose():
    """快速诊断：逐步测试，看到底哪一步失败。"""
    import sys
    print("=" * 50)
    print("[诊断] 步骤1: 连接 yuanrong worker...")
    try:
        store = UcmYuanrongStore(YUANRONG_CONFIG)
        print("[诊断] ✅ 连接成功")
    except Exception as e:
        print(f"[诊断] ❌ 连接失败: {e}")
        return

    print("[诊断] 步骤2: lookup 随机不存在的 key...")
    try:
        masks = store.lookup([uuid.uuid4().bytes])
        print(f"[诊断] ✅ lookup 返回 {masks}")
    except Exception as e:
        print(f"[诊断] ❌ lookup 失败: {e}")
        return

    print("[诊断] 步骤3: 创建 CPU tensor 并 dump...")
    try:
        t = torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE)
        bid = block_id_from_tensor(t)
        task = store.dump([bid], [0], [[t]])
        print("[诊断] ✅ dump 已发起，等待完成...")
        store.wait(task)
        print("[诊断] ✅ dump 完成")
    except Exception as e:
        print(f"[诊断] ❌ dump 失败: {e}")
        return

    print("[诊断] 步骤4: lookup 刚才写入的 key...")
    try:
        masks = store.lookup([bid])
        print(f"[诊断] ✅ lookup 返回 {masks}")
    except Exception as e:
        print(f"[诊断] ❌ lookup 失败: {e}")
        return

    print("[诊断] 步骤5: load 刚才写入的数据...")
    try:
        dst = torch.empty(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE)
        task = store.load([bid], [0], [[dst]])
        store.wait(task)
        print(f"[诊断] ✅ load 完成，内容一致: {torch.equal(t, dst)}")
    except Exception as e:
        print(f"[诊断] ❌ load 失败: {e}")
        return

    print("=" * 50)
    print("[诊断] 全部通过！yuanrong store 工作正常。")


def block_id_from_tensor(tensor: torch.Tensor) -> bytes:
    """Generate a deterministic block ID from tensor content."""
    tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
    hash_object = hashlib.blake2b(tensor_bytes)
    return bytes.fromhex(hash_object.hexdigest()[:32])


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

def test_lookup_found():
    """lookup returns True for block IDs that exist after dump."""
    print("\n[DEBUG] 创建 tensor 和 block_id...")
    block_data = [torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    block_ids = [block_id_from_tensor(t) for t in block_data]
    shard_index = [0] * len(block_ids)
    src_tensors = [[t] for t in block_data]

    print("[DEBUG] 连接 yuanrong worker...")
    store = UcmYuanrongStore(YUANRONG_CONFIG)
    print("[DEBUG] 连接成功，执行 dump...")
    task = store.dump(block_ids=block_ids, shard_index=shard_index, src_tensor=src_tensors)
    print("[DEBUG] dump 已发起，等待完成...")
    store.wait(task)
    print("[DEBUG] dump 完成，执行 lookup...")

    masks = store.lookup(block_ids)
    print(f"[DEBUG] lookup 结果: {masks}")
    assert all(mask is True for mask in masks)

def test_lookup_not_found():
    """lookup returns False for block IDs that have never been written."""
    store = UcmYuanrongStore(YUANRONG_CONFIG)
    block_ids = [uuid.uuid4().bytes for _ in range(10)]
    masks = store.lookup(block_ids)
    assert all(mask is False for mask in masks)



# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def test_dump_once():
    """Dump data once and confirm it is visible via lookup."""
    block_data = [torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    block_ids = [block_id_from_tensor(t) for t in block_data]
    shard_index = [0] * len(block_ids)
    src_tensors = [[t] for t in block_data]

    store = UcmYuanrongStore(YUANRONG_CONFIG)
    task = store.dump(block_ids=block_ids, shard_index=shard_index, src_tensor=src_tensors)
    store.wait(task)

    masks = store.lookup(block_ids)
    assert all(mask is True for mask in masks)


def test_dump_repeated():
    """Repeated dump of the same block IDs does not raise."""
    block_data = [torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    block_ids = [block_id_from_tensor(t) for t in block_data]
    shard_index = [0] * len(block_ids)
    src_tensors = [[t] for t in block_data]

    store = UcmYuanrongStore(YUANRONG_CONFIG)
    # First dump
    task = store.dump(block_ids=block_ids, shard_index=shard_index, src_tensor=src_tensors)
    store.wait(task)

    # Second dump with the same keys
    task = store.dump(block_ids=block_ids, shard_index=shard_index, src_tensor=src_tensors)
    store.wait(task)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def test_load_existing_data():
    """Load back the data that was previously dumped — content must match."""
    block_data = [torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    dst_data = [torch.empty(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    block_ids = [block_id_from_tensor(t) for t in block_data]
    shard_index = [0] * len(block_ids)
    src_tensors = [[t] for t in block_data]
    dst_tensors = [[t] for t in dst_data]

    store = UcmYuanrongStore(YUANRONG_CONFIG)
    # Write
    task = store.dump(block_ids=block_ids, shard_index=shard_index, src_tensor=src_tensors)
    store.wait(task)

    # Read back
    task = store.load(block_ids=block_ids, shard_index=shard_index, dst_tensor=dst_tensors)
    store.wait(task)

    # Verify content
    for src, dst in zip(block_data, dst_data):
        assert torch.equal(src, dst), "Loaded tensor does not match original"


def test_load_non_existent_data():
    """Loading data that was never written raises RuntimeError."""
    block_data = [torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    dst_data = [torch.empty(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    block_ids = [block_id_from_tensor(t) for t in block_data]
    shard_index = [0] * len(block_ids)
    dst_tensors = [[t] for t in dst_data]

    store = UcmYuanrongStore(YUANRONG_CONFIG)

    # Confirm they do not exist yet
    masks = store.lookup(block_ids)
    assert all(mask is False for mask in masks)

    # Attempting to load should fail
    with pytest.raises(RuntimeError, match="Transfer failed|async_mget_h2d failed"):
        task = store.load(block_ids=block_ids, shard_index=shard_index, dst_tensor=dst_tensors)
        store.wait(task)


# ---------------------------------------------------------------------------
# lookup_on_prefix
# ---------------------------------------------------------------------------


def test_lookup_on_prefix_full_hit():
    """lookup_on_prefix returns last index when every block is present."""
    block_data = [torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(3)]
    block_ids = [block_id_from_tensor(t) for t in block_data]
    shard_index = [0] * len(block_ids)
    src_tensors = [[t] for t in block_data]

    store = UcmYuanrongStore(YUANRONG_CONFIG)
    task = store.dump(block_ids=block_ids, shard_index=shard_index, src_tensor=src_tensors)
    store.wait(task)

    idx = store.lookup_on_prefix(block_ids)
    assert idx == len(block_ids) - 1


def test_lookup_on_prefix_first_miss():
    """lookup_on_prefix returns -1 when the first block is missing."""
    block_data = [torch.randn(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(3)]
    block_ids = [block_id_from_tensor(t) for t in block_data]

    store = UcmYuanrongStore(YUANRONG_CONFIG)
    # Only write the 2nd and 3rd blocks, leave the first missing
    existing = block_ids[1:]
    task = store.dump(
        block_ids=existing,
        shard_index=[0] * len(existing),
        src_tensor=[[t] for t in block_data[1:]],
    )
    store.wait(task)

    # Prefix scan should stop at the very first block (miss)
    idx = store.lookup_on_prefix(block_ids)
    assert idx == -1
if __name__ == "__main__":
    debug_diagnose()