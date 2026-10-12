"""CPU-only interval and schema checks using synthetic trace events."""

from argparse import Namespace
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from analyze_traces import Coverage, analyze, validate_output_path


def args(**overrides):
    values = dict(start_ms=0, end_ms=None, copy_name_regex=None, kernel_name_regex=None,
                  copy_stream=[], copy_bytes=[], results_layout=None, bin_ms=0.01, max_bins=2000, examples=20)
    values.update(overrides)
    return Namespace(**values)


def event(name, category, start, duration, device=0, stream=7, **arguments):
    return {"name": name, "cat": category, "ph": "X", "ts": start, "dur": duration,
            "pid": device, "tid": stream, "args": {"device": device, "stream": stream, **arguments}}


def events():
    return [
        event("compute_a", "kernel", 100, 10),
        event("compute_b", "kernel", 115, 15),
        event("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy", 105, 20, stream=8, bytes=256),
        event("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy", 108, 4, stream=9, bytes=256),
        event("Memcpy DtoH (Device -> Pageable)", "gpu_memcpy", 125, 15, stream=8, bytes=256),
        event("other_device_compute", "kernel", 100, 100, device=1),
        event("cudaMemcpyAsync", "cuda_runtime", 105, 90, stream=8),
        event("kv_disk_read", "user_annotation", 101, 3),
    ]


class AnalyzerTests(unittest.TestCase):
    def test_union_overlap_does_not_double_count_or_mix_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events(), "displayTimeUnit": "ns"}))
            (Path(directory) / "configuration.json").write_text(json.dumps({"image_count": 3}))
            report = analyze(path, args())
            self.assertEqual(report["configuration"]["image_count"], 3)
            device = next(row for row in report["devices"] if row["device"] == "0")
            self.assertAlmostEqual(device["compute_active_union_ms"], 0.025)
            h2d = device["directions"]["H2D"]
            self.assertAlmostEqual(h2d["copy_active_union_ms"], 0.020)
            self.assertAlmostEqual(h2d["sum_event_duration_ms"], 0.024)
            self.assertAlmostEqual(h2d["overlap_with_compute_ms"], 0.015)
            self.assertAlmostEqual(h2d["fraction_copy_active_overlapping_compute"], 0.75)
            self.assertAlmostEqual(device["directions"]["D2H"]["overlap_with_compute_ms"], 0.005)
            self.assertAlmostEqual(device["directions"]["H2D_or_D2H"]["overlap_with_compute_ms"], 0.020)
            self.assertEqual(report["cpu_disk_annotation_counts"], {"kv_disk_read": 1})
            self.assertEqual(report["trace_timestamp_unit"], "microseconds")
            self.assertEqual(h2d["host_memory_label_counts"], {"pinned": 2})
            self.assertEqual(len(device["overlap_examples"]), 3)

    def test_clipped_window_and_stream_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(events()))
            report = analyze(path, args(start_ms=0.007, end_ms=0.022, copy_stream=["8"]))
            device = next(row for row in report["devices"] if row["device"] == "0")
            h2d = device["directions"]["H2D"]
            self.assertAlmostEqual(h2d["copy_active_union_ms"], 0.015)
            self.assertAlmostEqual(h2d["overlap_with_compute_ms"], 0.010)
            self.assertEqual(h2d["events_without_byte_count"], 1)
            self.assertEqual(report["observed_selected_copy_counts_by_bytes"]["H2D"], {"missing_or_window_clipped": 1})

    def test_gzip_schema_aliases_and_invalid_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json.gz"
            sample = events()[:2]
            sample.append({"name": "memory transfer", "cat": "cuda_memcpy", "ph": "X", "ts": 105, "dur": 20,
                           "args": {"Device Id": 0, "Stream Id": 8, "Copy Kind": "HtoD", "Bytes": 256}})
            sample.append({"name": "bad", "cat": "kernel", "ph": "X", "ts": 100, "dur": -1})
            sample.append({"name": "begin_only", "cat": "kernel", "ph": "B", "ts": 100})
            with gzip.open(path, "wt") as handle:
                json.dump(sample, handle)
            report = analyze(path, args())
            self.assertEqual(report["devices"][0]["directions"]["H2D"]["event_count"], 1)
            invalid = report["source_schema"]["ignored_or_invalid"]
            self.assertEqual(invalid["invalid_or_nonpositive_timestamp_duration"], 1)
            self.assertEqual(invalid["recognized_gpu_or_io_event_without_complete_X_phase"], 1)

    def test_size_matched_inference_from_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(events()))
            results = Path(directory) / "results.json"
            layout = [{"layers": [{"tensors": [{"shape": [1, 16, 2, 4], "dtype": "torch.bfloat16"}] * 2}]}]
            results.write_text(json.dumps({"mode": "ram_async", "records": [{"run": 0, "profiled": True,
                              "actual_steps": 3, "phase": {"cache_layout": layout}}]}))
            report = analyze(path, args(results_layout=results))
            self.assertEqual(report["filters"]["copy_bytes"], [256])
            self.assertEqual(report["size_attribution"]["expected_full_trace_cache_copy_counts_by_bytes"],
                             {"D2H": {"256": 2}, "H2D": {"256": 4}})
            self.assertEqual(report["observed_selected_copy_counts_by_bytes"], {"H2D": {"256": 2}, "D2H": {"256": 1}})

    def test_missing_device_argument_uses_unambiguous_pid_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            sample = events()[:3]
            del sample[2]["args"]["device"]
            path.write_text(json.dumps(sample))
            report = analyze(path, args())
            self.assertEqual(len(report["devices"]), 1)
            self.assertAlmostEqual(report["devices"][0]["directions"]["H2D"]["overlap_with_compute_ms"], 0.015)

    def test_coverage_query_boundaries(self):
        coverage = Coverage([(1, 3), (2, 5), (6, 8)])
        self.assertEqual(coverage.covered(0, 10), 6)
        self.assertEqual(coverage.covered(3, 7), 3)
        self.assertEqual(coverage.covered(5, 6), 0)
        self.assertEqual(coverage.covered(10, 11), 0)

    def test_output_cannot_overwrite_raw_evidence_or_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, configuration, results = (root / name for name in ("trace.json", "configuration.json", "results.json"))
            for path in (trace, configuration, results):
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "raw input evidence"):
                    validate_output_path(path, [trace], results)
            alias = root / "alias.json"
            alias.symlink_to(trace)
            with self.assertRaises(ValueError):
                validate_output_path(alias, [trace], results)
            hardlink = root / "hardlink.json"
            hardlink.hardlink_to(trace)
            with self.assertRaises(ValueError):
                validate_output_path(hardlink, [trace], results)
            validate_output_path(root / "overlap.json", [trace], results)


if __name__ == "__main__":
    unittest.main(verbosity=2)
