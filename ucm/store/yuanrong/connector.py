from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ucm.logger import init_logger
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

logger = init_logger(__name__)

DEFAULT_TIMEOUT_MS: int = 60 * 1000  # 60 seconds

@dataclass
class UcmYuanrongTask(Task):
    """A task class for Yuanrong operations wrapping a yuanrong Future.

    Attributes:
        future: The yuanrong Future object returned by async_mset_d2h / async_mget_h2d.
        keys: The list of yuanrong keys associated with this transfer.
    """

    future: object
    keys: List[str]

class UcmYuanrongStore(UcmKVStoreBaseV1):
    """A KV cache store implementation backed by yuanrong data system.
    This store implements the UcmKVStoreBaseV1 interface by adapting UCM block
    operations to yuanrong HeteroClient calls. It supports both tensor-based
    (load/dump) and raw-address-based (load_data/dump_data) transfer methods.
    """
    def __init__(self, config: Dict):
        """Initialize the Yuanrong store with configuration.
        """
        super().__init__(config)

        # Lazy-import yuanrong SDK
        try:
            from yr.datasystem.hetero_client import Blob, DeviceBlobList, HeteroClient
            from yr.datasystem.kv_client import SetParam
            self._Blob = Blob
            self._DeviceBlobList = DeviceBlobList
            self._HeteroClient = HeteroClient
            self._SetParam = SetParam
        except ImportError as e:
            raise ImportError(
                "Please install yuanrong SDK and ensure 'yr.datasystem.hetero_client' "
                "is available in PYTHONPATH to use UcmYuanrongStore."
            ) from e

        # Read configuration
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 18482)
        self.key_prefix = config.get("key_prefix", "ucm")
        self.device_id = config.get("device_id", 0)
        self.tensor_size_list = config.get("tensor_size_list", None)
        self.timeout_ms = config.get("timeout_ms", DEFAULT_TIMEOUT_MS)

        if not self.host:
            raise ValueError("'host' must be specified in config for UcmYuanrongStore!")
        if not self.port:
            raise ValueError("'port' must be specified in config for UcmYuanrongStore")

        # Initialize HeteroClient and connect to yuanrong worker
        self.client = self._HeteroClient()
        ret = self.client.init(self.host, self.port)
        if ret != 0:
            raise RuntimeError(
                f"Failed to initialize HeteroClient with {self.host}:{self.port}, "
                f"return code: {ret}."
            )

        logger.info(
            "UcmYuanrongStore initialized: host=%s, port=%s, device_id=%s, timeout_ms=%s",
            self.host,
            self.port,
            self.device_id,
            self.timeout_ms,
        )

    def _build_keys(self, block_ids: List[bytes], shard_index: List[int]) -> List[str]:
        """Build yuanrong keys from block IDs and shard indices.

        Args:
            block_ids: List of vLLM block hashes as raw bytes.
            shard_index: List of shard indices corresponding to each block ID.
        """
        if shard_index is None:
            shard_index = [0] * len(block_ids)
        if len(block_ids) != len(shard_index):
            raise ValueError(
                f"block_ids length ({len(block_ids)}) != shard_index length "
                f"({len(shard_index)})."
            )

        return [self._encode_key(bid, sid) for bid, sid in zip(block_ids, shard_index)]

    def _encode_key(self, block_id: bytes, shard_index: int) -> str:
        """Generate a stable string key for yuanrong.

        Format: ``{key_prefix}:{block_id.hex()}:{shard_index}``

        Args:
            block_id: vLLM block hash as raw bytes.
            shard_index: Shard index for TP/layer distinction.
        """
        return f"{self.key_prefix}:{block_id.hex()}:{shard_index}"


    def cc_store(self) -> int:
        """Return low-level C/C++ pointer to the underlying store.

        Returns:
            0 (no native store pointer available).
        """
        return 0

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        """Check presence of blocks in yuanrong storage.

        Uses ``exist()`` on the HeteroClient to probe keys. Keys are generated
        with ``shard_index=0`` since lookup is a presence check.

        Args:
            block_ids: List of vLLM block hashes (raw bytes).
        """
        keys = [self._encode_key(bid, 0) for bid in block_ids]
        try:
            result = self.client.exist(keys)
        except Exception as e:
            raise RuntimeError(f"lookup failed for {len(block_ids)} block_ids: {e}") from e

        return result

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        """Check presence of blocks and return the last consecutive hit index.

        Scans from the front; returns the index of the last consecutive hit.
        If the first block is not found, returns -1.

        Args:
            block_ids: List of vLLM block hashes (raw bytes).
        """
        res = self.lookup(block_ids)
        for i, hit in enumerate(res):   
            if not hit:
                return i - 1
        return len(res) - 1

    def prefetch(self, block_ids: List[bytes]) -> None:
        """No-op: yuanrong store has no intermediate cache layer.
        Args:
            block_ids: List of vLLM block hashes to prefetch.
        """
        pass

    def blob_size(self, block_index: int) -> int:
        if self.tensor_size_list is None:
            raise ValueError("tensor_size_list must be provided in config to use blob_size()")
        if len(self.tensor_size_list) == 1:
            return self.tensor_size_list[0]
        if block_index >= len(self.tensor_size_list):
            raise IndexError(f"block_index {block_index} out of range for tensor_size_list of length {len(self.tensor_size_list)}")
        return self.tensor_size_list[block_index]

    def _build_blob_lists(self, block_ids: List[bytes], addr_rows: List[List[int]]):
        """Build a list of DeviceBlobList from address rows and corresponding size rows.

        Args:
            addr_rows: 2D list of device addresses, one inner list per block.
            size_rows: 2D list of sizes, matching addr_rows structure.
        """
        if len(block_ids) != len(addr_rows):
            raise ValueError(
                f"Length of block_ids ({len(block_ids)}) must match length of "
                f"addr_rows ({len(addr_rows)})."
            )
        
        blob_lists = []
        for row in addr_rows:
            blobs = [
                self._Blob(int(ptr), self.blob_size(blob_index)) for blob_index, ptr in enumerate(row)
            ]
            if not blobs:
                raise ValueError("addr_rows must not contain empty rows")
            blob_lists.append(
                self._DeviceBlobList(dev_idx=self.device_id, blob_list=blobs)
            )
        return blob_lists
        
    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]] | np.ndarray,
    ) -> Task:
        """Initiate H2D (host-to-device) transfer of KV cache data.

        Encodes keys and converts destination addresses to DeviceBlobList
        structures, then calls ``async_mget_h2d``.

        Args:
            block_ids: Block hashes to load.
            shard_index: Shard index for each block.
            dst_addr: 2D structure where ``dst_addr[i]`` is a list of device
                pointers for block ``i``.
        """
        keys = self._build_keys(block_ids, shard_index)
        # Build DeviceBlobList
        dev_blob_lists = self._build_blob_lists(block_ids, self.to_rows(dst_addr))

        try:
            future = self.client.async_mget_h2d(keys, dev_blob_lists, self.timeout_ms)
        except Exception as e:
            raise RuntimeError(
                f"async_mget_h2d failed for {len(keys)} keys: {e}"
            ) from e

        return UcmYuanrongTask(future=future, keys=keys)

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]] | np.ndarray,
        prerequisite_handle: int = 0,
    ) -> Task:
        """Initiate D2H (device-to-host) transfer of KV cache data.

        Encodes keys and converts source addresses to DeviceBlobList
        structures, then calls ``async_mset_d2h``.

        Args:
            block_ids: Block hashes to store.
            shard_index: Shard index for each block.
            src_addr: 2D structure where ``src_addr[i]`` is a list of device
                pointers for block ``i``.
            prerequisite_handle: Optional event handle for stream sync (unused
                in current yuanrong implementation).

        """
        keys = self._build_keys(block_ids, shard_index)
        dev_blob_lists = self._build_blob_lists(block_ids, self.to_rows(src_addr))

        # Create SetParam (use default for now)
        set_param = self._SetParam()

        try:
            future = self.client.async_mset_d2h(keys, dev_blob_lists, set_param)
        except Exception as e:
            raise RuntimeError(
                f"async_mset_d2h failed for {len(keys)} keys: {e}"
            ) from e

        return UcmYuanrongTask(future=future, keys=keys)

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        """Initiate transfer of KV cache from yuanrong to device tensors.

        Extracts data pointers and sizes from destination tensors, then
        delegates to the same yuanrong ``async_mget_h2d`` path as ``load_data``.

        Args:
            block_ids: Hashes of the blocks to load.
            shard_index: Shard index for each block.
            dst_tensor: Double-list where ``dst_tensor[i][j]`` is the
                destination tensor on device for block ``i``, tensor ``j``.

        """
        return self.load_data(
            block_ids=block_ids,
            shard_index=shard_index,
            dst_addr=[[t.data_ptr() for t in row] for row in dst_tensor],
        )

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        """Initiate transfer of KV cache from device tensors to yuanrong.

        Extracts data pointers and sizes from source tensors, then
        delegates to the same yuanrong ``async_mset_d2h`` path as ``dump_data``.

        Args:
            block_ids: Hashes of the blocks to store.
            shard_index: Shard index for each block.
            src_tensor: Double-list where ``src_tensor[i][j]`` is the
                source tensor on device for block ``i``, tensor ``j``.

        """
        return self.dump_data(
            block_ids=block_ids,
            shard_index=shard_index,
            src_addr=[[t.data_ptr() for t in row] for row in src_tensor],
        )

    def wait(self, task: Task) -> None:
        """Block until the given transfer task completes.

        Calls ``future.get(timeout_ms)`` on the yuanrong Future. If the
        returned ``failed_keys`` list is non-empty, raises RuntimeError.

        Args:
            task: Task handle returned by load/dump/load_data/dump_data.

        """
        if not isinstance(task, UcmYuanrongTask):
            raise TypeError(
                f"Expected UcmYuanrongTask, got {type(task).__name__}."
            )

        try:
            failed_keys = task.future.get(self.timeout_ms)
        except Exception as e:
            raise RuntimeError(
                f"Yuanrong transfer failed: {e}"
            ) from e

        if failed_keys:
            sample = failed_keys[:5] if len(failed_keys) > 5 else failed_keys
            raise RuntimeError(
                f"Transfer failed for {len(failed_keys)} / {len(task.keys)} keys. "
                f"Sample failed keys: {sample}"
            )

    def check(self, task: Task) -> bool:
        """Non-blocking poll for task completion.

        The yuanrong Future only exposes blocking ``get(timeout_ms)`` with no
        non-blocking completion query. Always returns ``False``; callers must
        use ``wait()`` to synchronize.

        Args:
            task: Task handle returned by any transfer method.

        """
        return False

    @staticmethod
    def to_rows(addr)->List[List[int]]:
        """Convert a 2D structure of addresses to a list of lists of ints.

        Handles both list-of-lists and 2D numpy array inputs.

        Args:
            addr: Either a list of lists of ints or a 2D numpy array.
        """
        if hasattr(addr, "tolist"):
            return addr.tolist()
        return [list(row) for row in addr]