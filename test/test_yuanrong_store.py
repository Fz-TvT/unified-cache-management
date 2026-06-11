"""Unit tests for UcmYuanrongStore.

Uses monkeypatching to replace the yuanrong SDK (yr.datasystem.hetero_client)
with fake objects, avoiding any dependency on a real yuanrong worker.
"""
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ucm.store.yuanrong.connector import UcmYuanrongStore, UcmYuanrongTask

# ---------------------------------------------------------------------------
# Fake SDK types – these stand in for yr.datasystem.hetero_client types
# ---------------------------------------------------------------------------


@dataclass
class FakeBlob:
    ptr: int
    size: int


@dataclass
class FakeDeviceBlobList:
    dev_idx: int
    blob_list: list


class FakeSetParam:
    """Stand-in for yr.datasystem.kv_client.SetParam."""
    pass


class FakeHeteroClient:
    """Stand-in for yr.datasystem.hetero_client.HeteroClient."""

    def __init__(self, host=None, port=None, connect_timeout_ms=None):
        self.init_called = True
        self.init_host = host
        self.init_port = port
        # MagicMock 作为方法，调用后自动记录调用信息并返回 MagicMock
        self.exist = MagicMock()
        self.async_mget_h2d = MagicMock()
        self.async_mset_d2h = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    """Patch the yr.datasystem modules with fake types before each test."""
    patches = [
        patch("yr.datasystem.hetero_client.Blob", FakeBlob),
        patch("yr.datasystem.hetero_client.DeviceBlobList", FakeDeviceBlobList),
        patch("yr.datasystem.hetero_client.HeteroClient", FakeHeteroClient),
        patch("yr.datasystem.kv_client.SetParam", FakeSetParam),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture
def valid_config():
    return {
        "host": "127.0.0.1",
        "port": 18482,
        "tensor_size_list": [1024, 512],
    }


@pytest.fixture
def store(mock_sdk, valid_config):
    """Return a fully-initialized UcmYuanrongStore (with mocked SDK)."""
    return UcmYuanrongStore(valid_config)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_success(self, mock_sdk):
        """Initialisation with valid config should create a client and call init."""
        config = {
            "host": "10.0.0.1",
            "port": 9999,
            "tensor_size_list": [2048],
        }
        store = UcmYuanrongStore(config)
        assert store.host == "10.0.0.1"
        assert store.port == 9999
        assert store.key_prefix == "ucm"
        assert store.device_id == 0
        assert store.tensor_size_list == [2048]
        assert store.timeout_ms == 60 * 1000
        assert isinstance(store.client, FakeHeteroClient)
        assert store.client.init_called is True
        assert store.client.init_host == "10.0.0.1"
        assert store.client.init_port == 9999

    def test_init_defaults(self, mock_sdk):
        """Config with only host+port should use sensible defaults."""
        config = {"host": "localhost", "port": 18482}
        store = UcmYuanrongStore(config)
        assert store.key_prefix == "ucm"
        assert store.device_id == 0
        assert store.timeout_ms == 60 * 1000

    def test_init_missing_host_raises_value_error(self, mock_sdk):
        """Missing or empty host should raise a ValueError."""
        with pytest.raises(ValueError, match="host"):
            UcmYuanrongStore({"port": 18482})
        with pytest.raises(ValueError, match="host"):
            UcmYuanrongStore({"host": "", "port": 18482})

    def test_init_missing_port_raises_value_error(self, mock_sdk):
        """Missing or empty port should raise a ValueError."""
        with pytest.raises(ValueError, match="port"):
            UcmYuanrongStore({"host": "127.0.0.1"})
        with pytest.raises(ValueError, match="port"):
            UcmYuanrongStore({"host": "127.0.0.1", "port": 0})

    def test_init_when_sdk_missing_raises_import_error(self):
        """When the yuanrong SDK is not installed, init should raise ImportError."""
        with pytest.raises(ImportError, match="yuanrong"):
            UcmYuanrongStore({"host": "127.0.0.1", "port": 18482})


# ---------------------------------------------------------------------------
# cc_store
# ---------------------------------------------------------------------------


class TestCcStore:
    def test_cc_store_returns_zero(self, store):
        """cc_store() should always return 0 for a pure Python store."""
        assert store.cc_store() == 0


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_calls_exist_with_correct_keys(self, store):
        """lookup() should call client.exist() with encoded keys (shard_index=0)."""
        block_ids = [b"aabb", b"ccdd"]
        store.client.exist.return_value = [True, False]

        result = store.lookup(block_ids)

        expected_keys = ["ucm:61616262:0", "ucm:63636464:0"]
        store.client.exist.assert_called_once_with(expected_keys)
        assert result == [True, False]

    def test_lookup_raises_runtime_error_on_failure(self, store):
        """lookup() should wrap client exceptions in RuntimeError."""
        store.client.exist = MagicMock(side_effect=Exception("connection lost"))
        with pytest.raises(RuntimeError, match="lookup failed"):
            store.lookup([b"test"])


# ---------------------------------------------------------------------------
# lookup_on_prefix
# ---------------------------------------------------------------------------


class TestLookupOnPrefix:
    def test_all_hit(self, store):
        """When all blocks exist, return the last index."""
        store.client.exist.return_value = [True, True, True]
        assert store.lookup_on_prefix([b"a", b"b", b"c"]) == 2

    def test_partial_hit(self, store):
        """When some blocks miss, return the index before the first miss."""
        store.client.exist.return_value = [True, True, False, True]
        assert store.lookup_on_prefix([b"a", b"b", b"c", b"d"]) == 1

    def test_first_miss(self, store):
        """When the first block is missing, return -1."""
        store.client.exist.return_value = [False, True, True]
        assert store.lookup_on_prefix([b"a", b"b", b"c"]) == -1

    def test_all_miss(self, store):
        """When all blocks are missing, return -1."""
        store.client.exist.return_value = [False, False]
        assert store.lookup_on_prefix([b"a", b"b"]) == -1


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_is_noop(self, store):
        """prefetch() should not raise and not call any client method."""
        store.prefetch([b"a", b"b"])
        # no exception means success


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_load_data_calls_async_mget_h2d(self, store):
        """load_data() should encode keys, build blob lists, and call async_mget_h2d."""
        store.tensor_size_list = [1024, 512]
        block_ids = [b"\x01\x02", b"\x03\x04"]
        shard_index = [0, 1]
        dst_addr = [[0x1000, 0x2000], [0x3000, 0x4000]]

        task = store.load_data(block_ids, shard_index, dst_addr)

        expected_keys = ["ucm:0102:0", "ucm:0304:1"]
        store.client.async_mget_h2d.assert_called_once()
        call_args = store.client.async_mget_h2d.call_args[0]
        assert call_args[0] == expected_keys                    # keys
        assert len(call_args[1]) == 2                           # dev_blob_lists
        assert call_args[2] == store.timeout_ms                 # timeout_ms

        # Verify Blob construction
        blob_lists = call_args[1]
        assert isinstance(blob_lists[0], FakeDeviceBlobList)
        assert len(blob_lists[0].blob_list) == 2
        assert blob_lists[0].blob_list[0].ptr == 0x1000
        assert blob_lists[0].blob_list[0].size == 1024
        assert blob_lists[0].blob_list[1].ptr == 0x2000
        assert blob_lists[0].blob_list[1].size == 512

        assert isinstance(task, UcmYuanrongTask)
        assert task.keys == expected_keys

    def test_load_data_with_numpy_input(self, store):
        """load_data() should accept 2D numpy array as dst_addr."""
        store.tensor_size_list = [256, 256]
        block_ids = [b"\xaa"]
        shard_index = [0]
        dst_addr = np.array([[0x1000, 0x2000]], dtype=np.int64)

        task = store.load_data(block_ids, shard_index, dst_addr)

        store.client.async_mget_h2d.assert_called_once()
        assert isinstance(task, UcmYuanrongTask)

    def test_load_data_raises_on_length_mismatch(self, store):
        """load_data() should raise ValueError when block_ids and shard_index differ in length."""
        with pytest.raises(ValueError, match="block_ids length"):
            store.load_data([b"\x01"], [0, 1], [[0x1000]])

    def test_load_data_raises_on_addr_mismatch(self, store):
        """load_data() should raise ValueError when addr rows != block_ids."""
        with pytest.raises(ValueError, match="Length of block_ids"):
            store.load_data([b"\x01", b"\x02"], [0, 1], [[0x1000]])

    def test_load_data_raises_runtime_error_on_failure(self, store):
        """load_data() should wrap client exception in RuntimeError."""
        store.client.async_mget_h2d = MagicMock(
            side_effect=Exception("timeout")
        )
        with pytest.raises(RuntimeError, match="async_mget_h2d failed"):
            store.load_data([b"\x01"], [0], [[0x1000]])


# ---------------------------------------------------------------------------
# dump_data
# ---------------------------------------------------------------------------


class TestDumpData:
    def test_dump_data_calls_async_mset_d2h(self, store):
        """dump_data() should encode keys, build blob lists, and call async_mset_d2h."""
        store.tensor_size_list = [2048]
        block_ids = [b"\xab\xcd"]
        shard_index = [2]
        src_addr = [[0x5000]]

        task = store.dump_data(block_ids, shard_index, src_addr)

        expected_keys = ["ucm:abcd:2"]
        store.client.async_mset_d2h.assert_called_once()
        call_args = store.client.async_mset_d2h.call_args[0]
        assert call_args[0] == expected_keys                    # keys
        assert len(call_args[1]) == 1                           # dev_blob_lists

        # Verify SetParam was constructed
        assert isinstance(call_args[2], FakeSetParam)

        # Verify Blob
        blob_list = call_args[1][0].blob_list
        assert len(blob_list) == 1
        assert blob_list[0].ptr == 0x5000
        assert blob_list[0].size == 2048

        assert isinstance(task, UcmYuanrongTask)
        assert task.keys == expected_keys

    def test_dump_data_raises_runtime_error_on_failure(self, store):
        """dump_data() should wrap client exception in RuntimeError."""
        store.client.async_mset_d2h = MagicMock(
            side_effect=Exception("write failed")
        )
        with pytest.raises(RuntimeError, match="async_mset_d2h failed"):
            store.dump_data([b"\x01"], [0], [[0x1000]])


# ---------------------------------------------------------------------------
# load (tensor interface)
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_delegates_to_load_data_with_data_ptr(self, store):
        """load() should extract data_ptr from tensors and delegate to load_data()."""
        store.tensor_size_list = [1024]
        block_ids = [b"\xaa"]
        shard_index = [0]
        tensors = [[torch.empty(256, dtype=torch.float32)]]

        task = store.load(block_ids, shard_index, tensors)

        store.client.async_mget_h2d.assert_called_once()
        call_args = store.client.async_mget_h2d.call_args[0]
        # The pointer in the Blob should match the tensor's data_ptr
        blob = call_args[1][0].blob_list[0]
        assert blob.ptr == tensors[0][0].data_ptr()
        assert isinstance(task, UcmYuanrongTask)


# ---------------------------------------------------------------------------
# dump (tensor interface)
# ---------------------------------------------------------------------------


class TestDump:
    def test_dump_delegates_to_dump_data_with_data_ptr(self, store):
        """dump() should extract data_ptr from tensors and delegate to dump_data()."""
        store.tensor_size_list = [1024]
        block_ids = [b"\xbb"]
        shard_index = [1]
        tensors = [[torch.empty(512, dtype=torch.float32)]]

        task = store.dump(block_ids, shard_index, tensors)

        store.client.async_mset_d2h.assert_called_once()
        call_args = store.client.async_mset_d2h.call_args[0]
        blob = call_args[1][0].blob_list[0]
        assert blob.ptr == tensors[0][0].data_ptr()
        assert isinstance(task, UcmYuanrongTask)


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------


class TestWait:
    def test_wait_on_successful_task(self, store):
        """wait() should return normally when future.get() returns empty list."""
        future = MagicMock()
        future.get.return_value = []
        task = UcmYuanrongTask(future=future, keys=["k1"])
        store.wait(task)  # should not raise

    def test_wait_on_failed_keys_raises_runtime_error(self, store):
        """wait() should raise RuntimeError when future returns failed keys."""
        future = MagicMock()
        future.get.return_value = ["k1", "k2"]
        task = UcmYuanrongTask(future=future, keys=["k1", "k2"])
        with pytest.raises(RuntimeError, match="Transfer failed for 2"):
            store.wait(task)

    def test_wait_on_future_exception_raises_runtime_error(self, store):
        """wait() should wrap future.get() exceptions in RuntimeError."""
        future = MagicMock()
        future.get.side_effect = Exception("internal error")
        task = UcmYuanrongTask(future=future, keys=["k1"])
        with pytest.raises(RuntimeError, match="Yuanrong transfer failed"):
            store.wait(task)

    def test_wait_with_wrong_task_type_raises_type_error(self, store):
        """wait() should raise TypeError when task is not UcmYuanrongTask."""

        class WrongTask:
            pass

        with pytest.raises(TypeError, match="UcmYuanrongTask"):
            store.wait(WrongTask())


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


class TestCheck:
    def test_check_always_returns_false(self, store):
        """check() should always return False since yuanrong has no non-blocking query."""
        task = UcmYuanrongTask(future=MagicMock(), keys=[])
        assert store.check(task) is False


# ---------------------------------------------------------------------------
# _encode_key
# ---------------------------------------------------------------------------


class TestEncodeKey:
    def test_encode_key_format(self, store):
        """_encode_key() should produce the expected colon-delimited format."""
        key = store._encode_key(b"\x0a\x1b\x2c", 3)
        assert key == "ucm:0a1b2c:3"

    def test_encode_key_different_prefix(self, mock_sdk):
        """_encode_key() should respect a custom key_prefix in config."""
        config = {"host": "x", "port": 1, "key_prefix": "myapp", "tensor_size_list": [1]}
        store = UcmYuanrongStore(config)
        key = store._encode_key(b"\xde\xad", 0)
        assert key == "myapp:dead:0"


# ---------------------------------------------------------------------------
# _build_keys
# ---------------------------------------------------------------------------


class TestBuildKeys:
    def test_build_keys_with_matching_lengths(self, store):
        keys = store._build_keys([b"\x01", b"\x02"], [5, 10])
        assert keys == ["ucm:01:5", "ucm:02:10"]

    def test_build_keys_with_none_shard_index(self, store):
        keys = store._build_keys([b"\xaa", b"\xbb"], None)
        assert keys == ["ucm:aa:0", "ucm:bb:0"]

    def test_build_keys_raises_on_length_mismatch(self, store):
        with pytest.raises(ValueError, match="block_ids length"):
            store._build_keys([b"\x01", b"\x02"], [0])


# ---------------------------------------------------------------------------
# blob_size
# ---------------------------------------------------------------------------


class TestBlobSize:
    def test_blob_size_returns_correct_value(self, store):
        """blob_size() should return the value at the given index."""
        store.tensor_size_list = [1024, 2048, 4096]
        assert store.blob_size(0) == 1024
        assert store.blob_size(1) == 2048
        assert store.blob_size(2) == 4096

    def test_blob_size_with_single_element_list(self, store):
        """When tensor_size_list has one element, blob_size should always return it."""
        store.tensor_size_list = [512]
        assert store.blob_size(0) == 512
        assert store.blob_size(99) == 512

    def test_blob_size_raises_on_out_of_range(self, store):
        """blob_size() should raise IndexError when index >= len(tensor_size_list)."""
        store.tensor_size_list = [128, 256]
        with pytest.raises(IndexError):
            store.blob_size(2)

    def test_blob_size_raises_when_not_configured(self, store):
        """blob_size() should raise ValueError when tensor_size_list is None."""
        store.tensor_size_list = None
        with pytest.raises(ValueError, match="tensor_size_list"):
            store.blob_size(0)
