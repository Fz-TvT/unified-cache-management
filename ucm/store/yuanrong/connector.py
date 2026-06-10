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
        self.tensor_size = config.get("tensor_size", 1048576)   # Default 1MB per tensor if not specified
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

    # ------------------------------------------------------------------
    # Key encoding
    # ------------------------------------------------------------------

    def _encode_key(self, block_id: bytes, shard_index: int) -> str:
        """Generate a stable string key for yuanrong.

        Format: ``{key_prefix}:{block_id.hex()}:{shard_index}``

        Args:
            block_id: vLLM block hash as raw bytes.
            shard_index: Shard index for TP/layer distinction.

        Returns:
            Stable string key used as yuanrong key.
        """
        return f"{self.key_prefix}:{block_id.hex()}:{shard_index}"

    # ------------------------------------------------------------------
    # UcmKVStoreBaseV1 interface
    # ------------------------------------------------------------------

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

        Returns:
            List of booleans, True if the corresponding block exists.

        Raises:
            RuntimeError: If the exist call fails.
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

        Returns:
            Index of the last consecutive hit, or -1 if the first block misses.
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

    # ------------------------------------------------------------------
    # Raw-address-based transfer (load_data / dump_data)
    # ------------------------------------------------------------------
    def _build_dev_blob_list(self, addrs: List[int], sizes: List[int]):
        """Build a DeviceBlobList from address and size lists.

        Args:
            addrs: List of device memory pointers.
            sizes: List of transfer sizes in bytes, corresponding to each address.

        Returns:
            A DeviceBlobList instance.
        """
        blob_list = [self._Blob(ptr, sz) for ptr, sz in zip(addrs, sizes)]
        return self._DeviceBlobList(dev_idx=self.device_id, blob_list=blob_list)

    def _resolve_sizes_for_addr_matrix(
        self, addr_matrix: List[List[int]]
    ) -> List[List[int]]:
        """Resolve transfer sizes for a 2D address matrix.

        Uses ``tensor_size_list`` or ``tensor_size`` from config. Each inner
        list may have a different number of elements (different layers may have
        different numbers of KV tensors).

        Args:
            addr_matrix: 2D list of device pointers, where ``addr_matrix[i]``
                contains pointers for block ``i``.

        Returns:
            2D list of sizes matching the structure of ``addr_matrix``.

        """
        if self.tensor_size_list is not None:
            # tensor_size_list must be a 2D list matching addr_matrix structure
            if len(self.tensor_size_list) != len(addr_matrix):
                raise ValueError(
                    f"tensor_size_list length ({len(self.tensor_size_list)}) "
                    f"does not match addr matrix length ({len(addr_matrix)})."
                )
            return self.tensor_size_list

        if self.tensor_size is not None:
            # Single size for all: broadcast to match addr_matrix structure
            return [[self.tensor_size] * len(addrs) for addrs in addr_matrix]

        raise ValueError(
            "tensor_size or tensor_size_list must be set in config when using "
            "load_data/dump_data. Yuanrong requires explicit transfer sizes."
        )

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

        Returns:
            A UcmYuanrongTask wrapping the yuanrong Future.

        Raises:
            ValueError: If input lengths don't match.
            RuntimeError: If the yuanrong call fails.
        """
        if len(block_ids) != len(shard_index):
            raise ValueError(
                f"block_ids length ({len(block_ids)}) != shard_index length "
                f"({len(shard_index)})."
            )

        # Encode keys
        keys = [self._encode_key(bid, sid) for bid, sid in zip(block_ids, shard_index)]

        # Normalize dst_addr to list of lists
        if isinstance(dst_addr, np.ndarray):
            addr_list = dst_addr.tolist()
        else:
            addr_list = dst_addr

        if len(addr_list) != len(block_ids):
            raise ValueError(
                f"dst_addr length ({len(addr_list)}) != block_ids length "
                f"({len(block_ids)})."
            )

        # Resolve sizes
        size_matrix = self._resolve_sizes_for_addr_matrix(addr_list)

        # Build one DeviceBlobList per key
        dev_blob_lists = [
            self._build_dev_blob_list(addrs, sizes)
            for addrs, sizes in zip(addr_list, size_matrix)
        ]

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
        Returns:
            A UcmYuanrongTask wrapping the yuanrong Future.
        """
        if len(block_ids) != len(shard_index):
            raise ValueError(
                f"block_ids length ({len(block_ids)}) != shard_index length "
                f"({len(shard_index)})."
            )

        # Encode keys
        keys = [self._encode_key(bid, sid) for bid, sid in zip(block_ids, shard_index)]

        # Normalize src_addr to list of lists
        if isinstance(src_addr, np.ndarray):
            addr_list = src_addr.tolist()
        else:
            addr_list = src_addr

        if len(addr_list) != len(block_ids):
            raise ValueError(
                f"src_addr length ({len(addr_list)}) != block_ids length "
                f"({len(block_ids)})."
            )

        # Resolve sizes
        size_matrix = self._resolve_sizes_for_addr_matrix(addr_list)

        # Build one DeviceBlobList per key
        dev_blob_lists = [
            self._build_dev_blob_list(addrs, sizes)
            for addrs, sizes in zip(addr_list, size_matrix)
        ]

        # Create SetParam (use default for now)
        set_param = self._SetParam()

        try:
            future = self.client.async_mset_d2h(keys, dev_blob_lists, set_param)
        except Exception as e:
            raise RuntimeError(
                f"async_mset_d2h failed for {len(keys)} keys: {e}"
            ) from e

        return UcmYuanrongTask(future=future, keys=keys)

    # ------------------------------------------------------------------
    # Tensor-based transfer (load / dump)
    # ------------------------------------------------------------------

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

        Returns:
            A UcmYuanrongTask wrapping the yuanrong Future.
        """
        if len(block_ids) != len(shard_index):
            raise ValueError(
                f"block_ids length ({len(block_ids)}) != shard_index length "
                f"({len(shard_index)})."
            )
        if len(block_ids) != len(dst_tensor):
            raise ValueError(
                f"block_ids length ({len(block_ids)}) != dst_tensor length "
                f"({len(dst_tensor)})."
            )

        keys = [self._encode_key(bid, sid) for bid, sid in zip(block_ids, shard_index)]

        # Build one DeviceBlobList per block from its tensor list
        dev_blob_lists = []
        for tensor_list in dst_tensor:
            blobs = [self._Blob(t.data_ptr(), t.nbytes) for t in tensor_list]
            dev_blob_lists.append(
                self._DeviceBlobList(dev_idx=self.device_id, blob_list=blobs)
            )

        try:
            future = self.client.async_mget_h2d(keys, dev_blob_lists, self.timeout_ms)
        except Exception as e:
            raise RuntimeError(
                f"async_mget_h2d failed for {len(keys)} keys: {e}"
            ) from e

        return UcmYuanrongTask(future=future, keys=keys)

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

        Returns:
            A UcmYuanrongTask wrapping the yuanrong Future.
        """
        if len(block_ids) != len(shard_index):
            raise ValueError(
                f"block_ids length ({len(block_ids)}) != shard_index length "
                f"({len(shard_index)})."
            )
        if len(block_ids) != len(src_tensor):
            raise ValueError(
                f"block_ids length ({len(block_ids)}) != src_tensor length "
                f"({len(src_tensor)})."
            )

        keys = [self._encode_key(bid, sid) for bid, sid in zip(block_ids, shard_index)]

        # Build one DeviceBlobList per block from its tensor list
        dev_blob_lists = []
        for tensor_list in src_tensor:
            blobs = [self._Blob(t.data_ptr(), t.nbytes) for t in tensor_list]
            dev_blob_lists.append(
                self._DeviceBlobList(dev_idx=self.device_id, blob_list=blobs)
            )

        # Create SetParam (use default for now)
        set_param = self._SetParam()

        try:
            future = self.client.async_mset_d2h(keys, dev_blob_lists, set_param)
        except Exception as e:
            raise RuntimeError(
                f"async_mset_d2h failed for {len(keys)} keys: {e}"
            ) from e

        return UcmYuanrongTask(future=future, keys=keys)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def wait(self, task: Task) -> None:
        """Block until the given transfer task completes.

        Calls ``future.get(timeout_ms)`` on the yuanrong Future. If the
        returned ``failed_keys`` list is non-empty, raises RuntimeError.

        Args:
            task: Task handle returned by load/dump/load_data/dump_data.

        Raises:
            TypeError: If task is not a UcmYuanrongTask.
            RuntimeError: If the transfer failed (non-empty failed_keys) or
                the future itself raised an error.
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

        Returns:
            Always ``False``.
        """
        return False
