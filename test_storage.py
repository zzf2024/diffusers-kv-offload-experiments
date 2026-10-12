"""Small CUDA checks for the experimental cache, independent of model weights."""

from pathlib import Path
import tempfile
import threading
import unittest

import torch

from storage import CacheFactory, MODES


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class StorageTests(unittest.TestCase):
    def make_pair(self, layer, shape=(1, 37, 2, 16)):
        count = 1
        for extent in shape:
            count *= extent
        key = torch.arange(count, dtype=torch.float32).reshape(shape).remainder(97).to(torch.bfloat16)
        return (key + layer).cuda(), (key - layer).cuda()

    def test_roundtrip_all_modes_and_cache_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in MODES:
                for policy in ("warm", "drop_hint") if mode.startswith("disk") else ("warm",):
                    with self.subTest(mode=mode, page_cache=policy):
                        factory = CacheFactory(mode, directory, page_cache=policy)
                        cache = factory(4)
                        expected = []
                        for index in range(4):
                            pair = self.make_pair(index)
                            expected.append(tuple(t.cpu().clone() for t in pair))
                            cache.get_layer(index).store(*pair)
                        for iteration in range(3):
                            for index in range(4):
                                pair = cache.get_layer(index).get()
                                for actual, wanted in zip(pair, expected[index]):
                                    self.assertTrue(torch.equal(actual.cpu(), wanted))
                                    self.assertEqual(actual.dtype, torch.bfloat16)
                                    self.assertEqual(actual.device.type, "cuda")
                        torch.cuda.synchronize()
                        factory.clear()
                        factory.clear()
                        stats = factory.stats()
                        self.assertTrue(stats["cleared"])
                        self.assertEqual(stats["gets"], 12)
                        self.assertEqual(stats["stores"], 4)
                        if mode != "gpu":
                            self.assertEqual(stats["h2d_bytes"], 3 * stats["stored_bytes"])
                            self.assertEqual(stats["d2h_bytes"], stats["stored_bytes"])
                        if mode.startswith("disk"):
                            self.assertEqual(stats["disk_read_bytes"], 3 * stats["stored_bytes"])
                            self.assertEqual(stats["disk_write_bytes"], stats["stored_bytes"])
                            self.assertEqual(len(factory.io_trace()), 16)
                            for event in factory.io_trace():
                                self.assertEqual(event["thread_id"] == threading.get_native_id(), mode == "disk_sync")
                            if policy == "drop_hint":
                                self.assertEqual(stats["drop_hint_calls"], 24)
                        if mode.endswith("async"):
                            self.assertLessEqual(stats["max_prefetch_entries"], 1)
                        self.assertEqual(list(Path(directory).iterdir()), [])

    def test_async_lifetimes_without_per_layer_synchronization(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("ram_async", "disk_async"):
                factory = CacheFactory(mode, directory)
                cache = factory(5)
                expected = []
                for index in range(5):
                    pair = self.make_pair(index, (1, 1024, 8, 32))
                    expected.append(tuple(t[:, :4].cpu().clone() for t in pair))
                    cache.get_layer(index).store(*pair)
                snapshots = []
                for iteration in range(3):
                    for index in range(5):
                        pair = cache.get_layer(index).get()
                        snapshots.append((index, tuple(t[:, :4].clone() for t in pair)))
                        del pair
                        # Encourage allocator reuse while copies and consumers remain queued.
                        junk = torch.empty((2, 1024, 8, 32), device="cuda", dtype=torch.bfloat16)
                        junk.fill_(11)
                        del junk
                factory.clear()
                for index, actual in snapshots:
                    for tensor, wanted in zip(actual, expected[index]):
                        self.assertTrue(torch.equal(tensor.cpu(), wanted))

    def test_offloaded_storage_owns_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in MODES[1:]:
                factory = CacheFactory(mode, directory)
                cache = factory(1)
                pair = self.make_pair(0, (2, 23, 2, 16))
                expected = tuple(t.cpu().clone() for t in pair)
                cache.get_layer(0).store(*pair)
                torch.cuda.synchronize()
                for tensor in pair:
                    tensor.zero_()
                got = cache.get_layer(0).get()
                for tensor, wanted in zip(got, expected):
                    self.assertTrue(torch.equal(tensor.cpu(), wanted))
                factory.clear()

    def test_cleanup_during_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("ram_async", "disk_async"):
                factory = CacheFactory(mode, directory)
                cache = factory(4)
                for index in range(4):
                    cache.get_layer(index).store(*self.make_pair(index))
                pair = cache.get_layer(0).get()
                result = pair[0].float().sum()
                factory.clear()
                self.assertTrue(torch.isfinite(result).item())
                self.assertEqual(cache.pending, {})
                self.assertEqual(cache.entries, {})
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_truncated_file_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = CacheFactory("disk_sync", directory)
            cache = factory(1)
            cache.get_layer(0).store(*self.make_pair(0))
            (cache.directory / "0000-k.bin").write_bytes(b"invalid")
            with self.assertRaisesRegex(OSError, "Truncated"):
                cache.get_layer(0).get()
            factory.clear()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_factory_branch_isolation_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = CacheFactory("disk_async", directory)
            positive, negative = factory(1), factory(1)
            for cache, layer in ((positive, 1), (negative, 8)):
                cache.get_layer(0).store(*self.make_pair(layer))
            self.assertFalse(torch.equal(positive.get_layer(0).get()[0], negative.get_layer(0).get()[0]))
            with self.assertRaisesRegex(RuntimeError, "overwritten"):
                positive.get_layer(0).store(*self.make_pair(2))
            factory.clear()
            with self.assertRaisesRegex(RuntimeError, "cleared"):
                positive.get_layer(0).get()
        with self.assertRaises(ValueError):
            CacheFactory("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
