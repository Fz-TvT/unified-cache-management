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


def block_id_from_tensor(tensor: torch.Tensor) -> bytes:
    """Generate a deterministic block ID from tensor content."""
    tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
    hash_object = hashlib.blake2b(tensor_bytes)
    return bytes.fromhex(hash_object.hexdigest()[:32])


def pinned_tensor(*args, **kwargs) -> torch.Tensor:
    """Create a pinned-memory (page-locked) CPU tensor filled with random values.

    Pinned memory can be safely accessed by device DMA operations, avoiding
    segfaults when yuanrong SDK reads from CPU pointers.
    """
    t = torch.empty(*args, **kwargs, pin_memory=True)
    t.normal_()
    return t


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


def test_lookup_not_found():
    """lookup returns False for block IDs that have never been written."""
    store = UcmYuanrongStore(YUANRONG_CONFIG)
    block_ids = [uuid.uuid4().bytes for _ in range(10)]
    masks = store.lookup(block_ids)
    assert all(mask is False for mask in masks)


def test_lookup_found():
    """lookup returns True for block IDs that exist after dump."""
    block_data = [pinned_tensor(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    block_ids = [block_id_from_tensor(t) for t in block_data]
    shard_index = [0] * len(block_ids)
    src_tensors = [[t] for t in block_data]

    store = UcmYuanrongStore(YUANRONG_CONFIG)
    task = store.dump(block_ids=block_ids, shard_index=shard_index, src_tensor=src_tensors)
    store.wait(task)

    masks = store.lookup(block_ids)
    assert all(mask is True for mask in masks)


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def test_dump_once():
    """Dump data once and confirm it is visible via lookup."""
    block_data = [pinned_tensor(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
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
    block_data = [pinned_tensor(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
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
    block_data = [pinned_tensor(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    dst_data = [torch.empty(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE, pin_memory=True) for _ in range(5)]
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
    block_data = [pinned_tensor(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(5)]
    dst_data = [torch.empty(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE, pin_memory=True) for _ in range(5)]
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
    block_data = [pinned_tensor(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(3)]
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
    block_data = [pinned_tensor(_TENSOR_SHAPE, dtype=_TENSOR_DTYPE) for _ in range(3)]
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
