"""Experimental immutable QwenImage 2.1 cache storage; not a Diffusers API.

Disk async uses a background I/O worker and a separate CUDA copy stream.
The drop_hint regime requests POSIX_FADV_DONTNEED before every file read;
this is advisory and must not be described as guaranteed cold storage.

Async store assumes immutable caller tensors until the D2H event completes.
Qwen's freshly cloned prefix tensors satisfy this contract. Prefetch depth
counts future layers: current and next GPU copies can coexist, as can
allocator-retained blocks. Buffer limits apply independently to CFG caches.
"""

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import os
import shutil
import tempfile
import threading
import time

import torch


MODES = ("gpu", "ram_sync", "ram_async", "disk_sync", "disk_async")


class _Layer:
    def __init__(self, owner, index):
        self.owner, self.index = owner, index

    def store(self, key, value):
        self.owner.store(self.index, key, value)

    def get(self):
        return self.owner.get(self.index)


class CacheFactory:
    """Callable replacement for the pipeline's cache constructor."""

    def __init__(self, mode, disk_dir=None, page_cache="warm"):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if page_cache not in ("warm", "drop_hint"):
            raise ValueError(f"Unknown page-cache policy: {page_cache}")
        if mode.startswith("disk") and disk_dir is None:
            raise ValueError("disk_dir is required for disk modes")
        self.mode, self.disk_dir, self.page_cache = mode, disk_dir, page_cache
        self.instances = []

    def __call__(self, num_layers):
        cache = StorageCache(num_layers, self.mode, self.disk_dir, self.page_cache)
        self.instances.append(cache)
        return cache

    def clear(self):
        first_error = None
        for cache in self.instances:
            try:
                cache.clear()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def stats(self):
        result = {"mode": self.mode, "page_cache": self.page_cache,
                  "cache_count": len(self.instances), "cleared": all(c.closed for c in self.instances),
                  "prefetch_depth": 1, "max_pending_writes": 2,
                  "disk_fsync": self.mode.startswith("disk"), "cold_guaranteed": False}
        for cache in self.instances:
            for name, value in cache.counters.items():
                if name.startswith("max_"):
                    result[name] = max(result.get(name, 0), value)
                else:
                    result[name] = result.get(name, 0) + value
        return result

    def io_trace(self):
        return [dict(event, cache_index=index) for index, cache in enumerate(self.instances)
                for event in cache.io_events]


class StorageCache:
    def __init__(self, num_layers, mode, disk_dir=None, page_cache="warm"):
        self.mode, self.page_cache = mode, page_cache
        self.layer_caches = [_Layer(self, i) for i in range(num_layers)]
        self.entries = {}
        self.pending = {}
        self.write_queue = deque()
        self.retired = deque()
        self.stream = None
        self.device = None
        self.closed = False
        self.next_get = 0
        self.io_events = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-disk") if mode == "disk_async" else None
        self.directory = None
        if mode.startswith("disk"):
            Path(disk_dir).mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="request-", dir=disk_dir))
        self.counters = {name: 0 for name in (
            "stored_bytes", "h2d_bytes", "d2h_bytes", "disk_write_bytes", "disk_read_bytes",
            "stores", "gets", "max_prefetch_entries", "max_write_queue_entries",
            "drop_hint_calls", "disk_write_seconds", "disk_read_seconds")}

    def get_layer(self, index):
        return self.layer_caches[index]

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Cache has already been cleared")

    def _init_device(self, device):
        if self.device is None:
            self.device = torch.device(device)
            if self.mode.endswith("async"):
                if self.device.type != "cuda":
                    raise ValueError("Async experiment requires CUDA")
                self.stream = torch.cuda.Stream(device=self.device)
        elif self.device != device:
            raise ValueError("A cache cannot span devices")

    @staticmethod
    def _size(pair):
        return sum(t.numel() * t.element_size() for t in pair)

    @staticmethod
    def _byte_view(tensor):
        return memoryview(tensor.view(torch.uint8).numpy()).cast("B")

    def _to_host(self, pair, asynchronous):
        if not asynchronous:
            return tuple(t.detach().to(device="cpu", copy=True).contiguous() for t in pair), None
        host = tuple(torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True) for t in pair)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for cpu, gpu in zip(host, pair):
                cpu.copy_(gpu.detach(), non_blocking=True)
                gpu.record_stream(self.stream)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return host, ready

    def _write(self, index, pair, ready):
        if ready is not None:
            ready.synchronize()
        started = time.perf_counter()
        with torch.profiler.record_function("kv_disk_write"):
            for suffix, tensor in zip(("k", "v"), pair):
                path = self.directory / f"{index:04d}-{suffix}.bin"
                raw = self._byte_view(tensor)
                with open(path, "wb", buffering=0) as handle:
                    remaining = raw
                    while remaining:
                        count = handle.write(remaining)
                        if not count:
                            raise OSError("Short write of KV cache")
                        remaining = remaining[count:]
                    os.fsync(handle.fileno())
                self.counters["disk_write_bytes"] += len(raw)
        self.counters["disk_write_seconds"] += time.perf_counter() - started
        self.io_events.append({"operation": "write", "layer": index, "start": started,
                               "end": time.perf_counter(), "thread_id": threading.get_native_id()})

    def _read(self, index, pinned):
        entry = self.entries[index]
        if entry.get("write_future") is not None:
            entry["write_future"].result()
        pair = tuple(torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pinned)
                     for shape, dtype in entry["metadata"])
        started = time.perf_counter()
        with torch.profiler.record_function("kv_disk_read"):
            for suffix, tensor in zip(("k", "v"), pair):
                path = self.directory / f"{index:04d}-{suffix}.bin"
                raw = self._byte_view(tensor)
                with open(path, "rb", buffering=0) as handle:
                    if self.page_cache == "drop_hint":
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        self.counters["drop_hint_calls"] += 1
                    remaining = raw
                    while remaining:
                        count = handle.readinto(remaining)
                        if not count:
                            raise OSError("Truncated KV cache file")
                        remaining = remaining[count:]
                    if handle.read(1):
                        raise OSError("Unexpected trailing bytes in KV cache file")
                self.counters["disk_read_bytes"] += len(raw)
        self.counters["disk_read_seconds"] += time.perf_counter() - started
        self.io_events.append({"operation": "read", "layer": index, "start": started,
                               "end": time.perf_counter(), "thread_id": threading.get_native_id()})
        return pair

    def store(self, index, key, value):
        self._check_open()
        if index in self.entries:
            raise RuntimeError("Immutable cache cannot be overwritten")
        if key.device != value.device:
            raise ValueError("K and V must be on the same device")
        self._init_device(key.device)
        pair = (key, value)
        size = self._size(pair)
        self.counters["stored_bytes"] += size
        self.counters["stores"] += 1
        entry = {"metadata": [(tuple(t.shape), t.dtype) for t in pair], "size": size}
        self.entries[index] = entry
        if self.mode == "gpu":
            entry["pair"] = pair
            return
        if self.mode == "disk_async":
            while self.write_queue and self.write_queue[0].done():
                self.write_queue.popleft().result()
            while len(self.write_queue) >= 2:
                self.write_queue.popleft().result()
        host, ready = self._to_host(pair, self.mode.endswith("async"))
        self.counters["d2h_bytes"] += size
        if self.mode.startswith("ram"):
            entry.update(pair=host, ready=ready)
        elif self.mode == "disk_sync":
            self._write(index, host, ready)
        else:
            future = self.executor.submit(self._write, index, host, ready)
            entry["write_future"] = future
            self.write_queue.append(future)
            self.counters["max_write_queue_entries"] = max(
                self.counters["max_write_queue_entries"], len(self.write_queue))

    def _upload(self, index):
        entry = self.entries[index]
        if self.mode == "disk_async":
            host = self._read(index, pinned=True)
        else:
            host = entry["pair"]
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if entry.get("ready") is not None:
                self.stream.wait_event(entry["ready"])
            pair = tuple(t.to(device=self.device, non_blocking=True) for t in host)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        self.counters["h2d_bytes"] += entry["size"]
        return pair, ready, host

    def _prefetch(self, index):
        if index >= len(self.layer_caches) or index in self.pending:
            return
        if index not in self.entries:
            raise RuntimeError("All layers must be stored before decoding")
        if self.mode == "disk_async":
            self.pending[index] = self.executor.submit(self._upload, index)
        else:
            self.pending[index] = self._upload(index)
        self.counters["max_prefetch_entries"] = max(
            self.counters["max_prefetch_entries"], len(self.pending))

    def _reap_hosts(self):
        while self.retired and self.retired[0][0].query():
            self.retired.popleft()
        while len(self.retired) >= 2:
            self.retired.popleft()[0].synchronize()

    def get(self, index):
        self._check_open()
        if index not in self.entries:
            raise RuntimeError("KV cache has not been populated")
        self.counters["gets"] += 1
        entry = self.entries[index]
        if self.mode == "gpu":
            return entry["pair"]
        if not self.mode.endswith("async"):
            host = entry["pair"] if self.mode == "ram_sync" else self._read(index, pinned=False)
            pair = tuple(t.to(device=self.device, copy=True) for t in host)
            self.counters["h2d_bytes"] += entry["size"]
            return pair
        if index != self.next_get:
            raise RuntimeError("Async prototype requires ordered layer traversal")
        self.next_get = (index + 1) % len(self.layer_caches)
        self._reap_hosts()
        self._prefetch(index)
        pending = self.pending.pop(index)
        pair, ready, host = pending.result() if self.mode == "disk_async" else pending
        current = torch.cuda.current_stream(self.device)
        current.wait_event(ready)
        for tensor in pair:
            tensor.record_stream(current)
        # Keep pinned CPU buffers alive until the corresponding DMA finishes.
        self.retired.append((ready, host))
        self._prefetch(index + 1)
        return pair

    def clear(self):
        if self.closed:
            return
        self.closed = True
        error = None
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=False)
            for future in self.write_queue:
                try:
                    future.result()
                except Exception as caught:
                    error = error or caught
            for pending in self.pending.values():
                if self.mode == "disk_async":
                    try:
                        pending.result()
                    except Exception as caught:
                        error = error or caught
            if self.stream is not None:
                self.stream.synchronize()
        finally:
            self.pending.clear()
            self.write_queue.clear()
            self.retired.clear()
            self.entries.clear()
            if self.directory is not None:
                shutil.rmtree(self.directory)
        if error is not None:
            raise error
