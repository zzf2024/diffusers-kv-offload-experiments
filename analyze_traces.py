#!/usr/bin/env python3
"""Analyze CUDA copy/compute overlap in PyTorch Chrome traces on the CPU.

Example:
  python analyze_traces.py run/trace.json --output run/overlap.json
  python analyze_traces.py run/trace.json.gz --start-ms 1000 --end-ms 4000 \
      --copy-stream 7 --output run/window-overlap.json

Windows are relative to the earliest recognized GPU event, in milliseconds.
Chrome ts/dur values are microseconds; displayTimeUnit does not change them.
The analyzer measures temporal overlap, not the counterfactual time saved.
GPU KV-copy attribution is unavailable in the frozen experiment: the named
kv_disk_read/write scopes cover CPU I/O, not the surrounding CUDA copies.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def arg_value(args, *names):
    normalized = {key_name(key): value for key, value in args.items()}
    for name in names:
        if key_name(name) in normalized:
            return normalized[key_name(name)]
    return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


class Coverage:
    """Union of intervals with logarithmic interval-coverage queries."""

    def __init__(self, intervals):
        self.intervals = merged(intervals)
        self.starts = [item[0] for item in self.intervals]
        self.ends = [item[1] for item in self.intervals]
        self.prefix = [0.0]
        for start, end in self.intervals:
            self.prefix.append(self.prefix[-1] + end - start)

    def covered(self, start, end):
        first = bisect_right(self.ends, start)
        last = bisect_left(self.starts, end)
        if first >= last:
            return 0.0
        total = self.prefix[last] - self.prefix[first]
        return total - max(0.0, start - self.starts[first]) - max(0.0, self.ends[last - 1] - end)

    def intersections(self, start, end, limit=None):
        first = bisect_right(self.ends, start)
        last = bisect_left(self.starts, end)
        if limit is not None:
            last = min(last, first + limit)
        return [(max(start, self.starts[i]), min(end, self.ends[i])) for i in range(first, last)]


def intersection(left, right):
    output, first, second = [], 0, 0
    while first < len(left) and second < len(right):
        start = max(left[first][0], right[second][0])
        end = min(left[first][1], right[second][1])
        if end > start:
            output.append((start, end))
        if left[first][1] <= right[second][1]:
            first += 1
        else:
            second += 1
    return output


def classify(event):
    category = str(event.get("cat", "")).lower()
    category_parts = set(re.split(r"[,;\s]+", category))
    name = str(event.get("name", ""))
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    if category_parts.intersection(("cuda_runtime", "cuda_driver", "cpu_op", "user_annotation")):
        return None
    is_kernel = bool(category_parts.intersection(("kernel", "gpu_kernel", "cuda_kernel")))
    is_copy = bool(category_parts.intersection(("gpu_memcpy", "cuda_memcpy", "memcpy")))
    if not is_copy and "memcpy" in name.lower() and arg_value(args, "stream", "stream id") is not None:
        is_copy = True
    if is_kernel:
        return "kernel"
    if not is_copy:
        return None
    direction = key_name(name + " " + str(arg_value(args, "copy kind", "copy_kind", "direction") or ""))
    if any(value in direction for value in ("htod", "h2d", "hosttodevice")):
        return "H2D"
    if any(value in direction for value in ("dtoh", "d2h", "devicetohost")):
        return "D2H"
    if any(value in direction for value in ("dtod", "d2d", "devicetodevice")):
        return "D2D"
    source = key_name(arg_value(args, "src kind", "source memory kind", "src memory type") or "")
    target = key_name(arg_value(args, "dst kind", "destination memory kind", "dst memory type") or "")
    host_words = ("host", "pinned", "pageable")
    if any(word in source for word in host_words) and "device" in target:
        return "H2D"
    if "device" in source and any(word in target for word in host_words):
        return "D2H"
    return "unclassified_copy"


def load_events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    if isinstance(document, list):
        raw, metadata = document, {"layout": "top-level event array"}
    elif isinstance(document, dict) and isinstance(document.get("traceEvents"), list):
        raw = document["traceEvents"]
        metadata = {key: document.get(key) for key in ("schemaVersion", "displayTimeUnit", "baseTimeNanoseconds")}
        metadata["layout"] = "traceEvents object"
    else:
        raise ValueError("Expected a Chrome event array or an object containing traceEvents")
    counts = Counter()
    categories = Counter()
    invalid = Counter()
    events, scopes, metadata_events = [], [], []
    for index, event in enumerate(raw):
        if not isinstance(event, dict):
            invalid["non_object_event"] += 1
            continue
        phase = event.get("ph")
        counts[str(phase)] += 1
        categories[str(event.get("cat", ""))] += 1
        if phase == "M" and event.get("name") in ("process_name", "thread_name", "process_labels"):
            metadata_events.append(event)
        kind = classify(event)
        is_scope = event.get("name") in ("kv_disk_read", "kv_disk_write")
        if kind is None and not is_scope:
            continue
        if phase != "X":
            invalid["recognized_gpu_or_io_event_without_complete_X_phase"] += 1
            continue
        start, duration = number(event.get("ts")), number(event.get("dur"))
        if start is None or duration is None or duration <= 0:
            invalid["invalid_or_nonpositive_timestamp_duration"] += 1
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        device = arg_value(args, "device", "device id", "device index")
        device_source = "event args" if device is not None else "GPU event pid fallback"
        stream = arg_value(args, "stream", "stream id")
        memory_labels = " ".join([str(event.get("name", "")), str(arg_value(args, "src kind") or ""), str(arg_value(args, "dst kind") or "")]).lower()
        memory_type = "pinned" if "pinned" in memory_labels else ("pageable" if "pageable" in memory_labels else "unlabeled")
        item = {
            "event_index": index, "name": str(event.get("name", "")), "kind": kind,
            "start": start, "end": start + duration,
            "device": str(device) if device is not None else f"pid:{event.get('pid', 'unknown')}",
            "device_source": device_source,
            "stream": str(stream) if stream is not None else f"tid:{event.get('tid', 'unknown')}",
            "stream_source": "event args" if stream is not None else "GPU event tid fallback",
            "pid": event.get("pid"), "tid": event.get("tid"), "host_memory_label": memory_type,
            "bytes": number(arg_value(args, "bytes", "size bytes", "size")),
            "correlation": arg_value(args, "correlation", "correlation id"),
            "external_id": arg_value(args, "external id", "external_id"),
        }
        if is_scope:
            scopes.append(item)
        elif kind is not None:
            events.append(item)
    metadata.update(raw_event_count=len(raw), phase_counts=dict(counts), category_counts=dict(categories),
                    ignored_or_invalid=dict(invalid), process_thread_metadata=metadata_events)
    known_devices = defaultdict(set)
    for event in events:
        if event["device_source"] == "event args" and event["pid"] is not None:
            known_devices[str(event["pid"])].add(event["device"])
    for event in events:
        candidates = known_devices[str(event["pid"])]
        if event["device_source"] != "event args" and len(candidates) == 1:
            event["device"] = next(iter(candidates))
            event["device_source"] = "pid mapped from same-trace GPU events"
    return events, scopes, metadata


def intervals(events):
    return [(event["start"], event["end"]) for event in events]


def summary(events, compute, start, end):
    copy_coverage = Coverage(intervals(events))
    overlap = Coverage(intersection(copy_coverage.intervals, compute.intervals))
    copy_us, overlap_us = copy_coverage.covered(start, end), overlap.covered(start, end)
    return {
        "event_count": len(events),
        "sum_event_duration_ms": sum(event["end"] - event["start"] for event in events) / 1000,
        "copy_active_union_ms": copy_us / 1000,
        "overlap_with_compute_ms": overlap_us / 1000,
        "copy_active_without_compute_ms": (copy_us - overlap_us) / 1000,
        "fraction_copy_active_overlapping_compute": overlap_us / copy_us if copy_us else None,
        "copy_bytes_from_trace": sum(event["bytes"] for event in events if event["bytes"] is not None),
        "events_without_byte_count": sum(event["bytes"] is None for event in events),
        "host_memory_label_counts": dict(Counter(event["host_memory_label"] for event in events)),
    }, copy_coverage, overlap


def event_evidence(event, origin):
    return {
        key: event[key] for key in ("event_index", "name", "kind", "stream", "pid", "tid", "correlation", "external_id")
    } | {"start_ms": (event["start"] - origin) / 1000, "end_ms": (event["end"] - origin) / 1000}


def layout_sizes(path):
    document = json.loads(path.read_text())
    records = document.get("records", [])
    record = next((row for row in records if row.get("profiled")), None)
    if record is None:
        record = next((row for row in records if not row.get("warmup")), None)
    if record is None:
        raise ValueError("results-layout must contain a measured or profiled record")
    layout = record.get("phase", {}).get("cache_layout")
    if not layout:
        raise ValueError("Selected results record has no phase.cache_layout")
    element_sizes = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8,
                     "int8": 1, "uint8": 1, "int16": 2, "int32": 4, "int64": 8, "bool": 1}
    sizes = Counter()
    for cache in layout:
        for layer in cache["layers"]:
            for tensor in layer["tensors"]:
                dtype = tensor["dtype"].removeprefix("torch.")
                if dtype not in element_sizes:
                    raise ValueError(f"Unsupported cache dtype for byte matching: {dtype}")
                byte_count = math.prod(tensor["shape"]) * element_sizes[dtype]
                sizes[int(byte_count)] += 1
    steps = record.get("actual_steps")
    offload = document.get("mode") in ("ram_sync", "ram_async", "disk_sync", "disk_async")
    expected = None
    if offload and isinstance(steps, int) and steps > 0:
        expected = {"D2H": {str(size): count for size, count in sizes.items()},
                    "H2D": {str(size): count * (steps - 1) for size, count in sizes.items()}}
    return set(sizes), {"results_path": str(path.resolve()), "results_sha256": sha256(path),
                        "record_run": record.get("run"), "record_profiled": record.get("profiled"),
                        "mode": document.get("mode"), "tensor_counts_by_bytes": dict(sizes),
                        "expected_full_trace_cache_copy_counts_by_bytes": expected,
                        "attribution": "Size-matched inference only; unrelated transfers can have identical byte counts."}


def analyze(path, args):
    events, scopes, metadata = load_events(path)
    gpu_events = [event for event in events if event["kind"] != "unclassified_copy"]
    if not gpu_events:
        raise ValueError(f"No recognized complete CUDA kernel/copy events in {path}")
    origin = min(event["start"] for event in gpu_events)
    last = max(event["end"] for event in gpu_events)
    start = origin + args.start_ms * 1000
    end = last if args.end_ms is None else min(last, origin + args.end_ms * 1000)
    if end <= start:
        raise ValueError("Selected time window has no duration")
    copy_regex = re.compile(args.copy_name_regex) if args.copy_name_regex else None
    kernel_regex = re.compile(args.kernel_name_regex) if args.kernel_name_regex else None
    allowed_bytes = set(args.copy_bytes)
    size_attribution = None
    if args.results_layout is not None:
        derived_bytes, size_attribution = layout_sizes(args.results_layout)
        allowed_bytes.update(derived_bytes)
    selected = []
    excluded = Counter()
    for original in events:
        kind = original["kind"]
        if original["end"] <= start or original["start"] >= end:
            continue
        if kind != "kernel":
            if allowed_bytes and original["bytes"] not in allowed_bytes:
                excluded["copy_bytes_unmatched_or_missing"] += 1
                continue
            if args.copy_stream and original["stream"] not in args.copy_stream:
                excluded["copy_stream_filter"] += 1
                continue
            if copy_regex and not copy_regex.search(original["name"]):
                excluded["copy_name_filter"] += 1
                continue
        elif kernel_regex and not kernel_regex.search(original["name"]):
            excluded["kernel_name_filter"] += 1
            continue
        event = dict(original)
        event["start"], event["end"] = max(start, original["start"]), min(end, original["end"])
        if event["start"] != original["start"] or event["end"] != original["end"]:
            # A clipped time window cannot attribute a fraction of transfer bytes.
            event["bytes"] = None
        selected.append(event)
    devices = []
    for device in sorted({event["device"] for event in selected}):
        own = [event for event in selected if event["device"] == device]
        kernels = sorted((event for event in own if event["kind"] == "kernel"), key=lambda event: event["start"])
        copies = [event for event in own if event["kind"] in ("H2D", "D2H")]
        compute = Coverage(intervals(kernels))
        device_report = {
            "device": device, "kernel_count": len(kernels),
            "compute_active_union_ms": compute.covered(start, end) / 1000,
            "device_identity_sources": dict(Counter(event["device_source"] for event in own)),
            "other_copy_counts": dict(Counter(event["kind"] for event in own if event["kind"] not in ("kernel", "H2D", "D2H"))),
            "directions": {}, "streams": [], "windows": [], "overlap_examples": [],
        }
        coverages = {}
        for direction in ("H2D", "D2H", "H2D_or_D2H"):
            chosen = copies if direction == "H2D_or_D2H" else [event for event in copies if event["kind"] == direction]
            metrics, coverage, overlap = summary(chosen, compute, start, end)
            device_report["directions"][direction] = metrics
            coverages[direction] = (coverage, overlap)
        for stream in sorted({event["stream"] for event in own}):
            stream_events = [event for event in own if event["stream"] == stream]
            stream_kernels = [event for event in stream_events if event["kind"] == "kernel"]
            stream_report = {"stream": stream, "kernel_count": len(stream_kernels),
                             "stream_identity_sources": dict(Counter(event["stream_source"] for event in stream_events)),
                             "compute_active_union_ms": Coverage(intervals(stream_kernels)).covered(start, end) / 1000,
                             "copy_only_observed_stream": not stream_kernels and any(event["kind"] in ("H2D", "D2H") for event in stream_events),
                             "directions": {}}
            for direction in ("H2D", "D2H"):
                chosen = [event for event in stream_events if event["kind"] == direction]
                stream_report["directions"][direction] = summary(chosen, compute, start, end)[0]
            device_report["streams"].append(stream_report)
        effective_bin = max(args.bin_ms * 1000, (end - start) / args.max_bins)
        cursor = start
        while cursor < end:
            stop = min(end, cursor + effective_bin)
            window = {"start_ms": (cursor - origin) / 1000, "end_ms": (stop - origin) / 1000,
                      "compute_active_union_ms": compute.covered(cursor, stop) / 1000}
            for direction in ("H2D", "D2H"):
                coverage, overlap = coverages[direction]
                window[direction] = {"copy_active_union_ms": coverage.covered(cursor, stop) / 1000,
                                     "overlap_with_compute_ms": overlap.covered(cursor, stop) / 1000}
            device_report["windows"].append(window)
            cursor = stop
        device_report["effective_bin_ms"] = effective_bin / 1000
        ranked = sorted(((compute.covered(event["start"], event["end"]), event) for event in copies),
                        key=lambda item: item[0], reverse=True)
        for overlap_us, event in ranked[:args.examples]:
            if overlap_us <= 0:
                continue
            compute_event = next((kernel for kernel in kernels if kernel["start"] < event["end"] and kernel["end"] > event["start"]), None)
            sample = {"copy": event_evidence(event, origin), "overlap_ms": overlap_us / 1000,
                      "compute_union_intersection_examples_ms": [[(a - origin) / 1000, (b - origin) / 1000] for a, b in compute.intersections(event["start"], event["end"], limit=3)]}
            if compute_event is not None:
                sample["overlapping_kernel_example"] = event_evidence(compute_event, origin)
            device_report["overlap_examples"].append(sample)
        devices.append(device_report)
    configuration = None
    config_path = path.parent / "configuration.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        configuration = {key: config.get(key) for key in ("mode", "page_cache", "image_count", "resolution", "cfg_scale", "steps", "output_type", "profile")}
    scopes = [dict(scope, start=max(start, scope["start"]), end=min(end, scope["end"]))
              for scope in scopes if scope["start"] < end and scope["end"] > start]
    scope_counts = Counter(scope["name"] for scope in scopes)
    return {
        "trace_path": str(path.resolve()), "trace_file_bytes": path.stat().st_size,
        "trace_sha256": sha256(path), "source_schema": metadata, "configuration": configuration,
        "trace_timestamp_unit": "microseconds", "origin_trace_ts_us": origin,
        "time_window": {"start_ms": (start - origin) / 1000, "end_ms": (end - origin) / 1000},
        "filters": {"copy_streams": args.copy_stream, "copy_name_regex": args.copy_name_regex,
                    "copy_bytes": sorted(allowed_bytes), "kernel_name_regex": args.kernel_name_regex,
                    "excluded_events": dict(excluded)},
        "size_attribution": size_attribution,
        "observed_selected_copy_counts_by_bytes": {
            direction: dict(Counter(str(int(event["bytes"])) if event["bytes"] is not None else "missing_or_window_clipped"
                                    for event in selected if event["kind"] == direction))
            for direction in ("H2D", "D2H")
        },
        "gpu_copy_attribution": "Unattributed pipeline GPU copies; stream/name filters are explicit selection assumptions, not proof of KV origin.",
        "cpu_disk_annotation_counts": dict(scope_counts),
        "cpu_disk_annotation_scope": "Selected time window; unaffected by GPU copy stream/name/size filters",
        "cpu_disk_annotation_total_ms": {name: sum(scope["end"] - scope["start"] for scope in scopes if scope["name"] == name) / 1000 for name in scope_counts},
        "devices": devices,
    }


CAVEATS = [
    "Temporal overlap is measured on each GPU separately. It does not establish speedup or how much latency is hidden.",
    "Union durations avoid double-counting concurrent events. Sum event durations can exceed elapsed time and are separately reported.",
    "The frozen profiler window includes encoding, denoising, cleanup, and a uniform final latent CPU copy; with image output it also includes VAE decode and postprocessing.",
    "The frozen kv_disk_read/write annotations cover CPU file I/O only. GPU copies lack an explicit KV marker; whole-trace copy totals include non-KV transfers.",
    "Async variants use pinned host buffers while sync variants use ordinary host tensors. Pinning and stream overlap are confounded in the experimental comparison.",
    "Host-memory labels are copied from trace names/arguments when available; unlabeled transfers are not assumed pageable or pinned.",
    "A stream with copies and no observed kernels is a candidate transfer stream, not proof that its copies are exclusively KV traffic.",
    "Optional byte-size matching uses measured cache tensor shapes and dtype sizes. Matching expected counts strengthens attribution but cannot prove that no unrelated copies share those sizes; expected counts apply to the full trace only.",
    "CPU disk worker annotations may be absent because profiler thread capture differs by PyTorch version. Absence does not prove no disk I/O.",
    "Separately recorded disk_io_trace uses perf_counter timestamps; this analyzer does not align that clock to Chrome timestamps without a measured anchor.",
    "Only complete Chrome X events with positive finite durations are used. Non-X recognized events and invalid events are reported, not silently reconstructed.",
    "Device/stream IDs prefer event arguments; pid/tid fallback is flagged and may not identify the same device across different event schemas.",
    "This standard-library analyzer loads one complete trace into CPU memory at a time; large or compressed traces can require substantially more RAM than their file size.",
]


def validate_output_path(output, traces, results_layout=None):
    protected = list(traces) + [path.parent / "configuration.json" for path in traces]
    if results_layout is not None:
        protected.append(results_layout)
    destination = output.resolve()
    for source in protected:
        same_path = destination == source.resolve()
        same_file = output.exists() and source.exists() and os.path.samefile(output, source)
        if same_path or same_file:
            raise ValueError(f"Output must not overwrite raw input evidence: {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-ms", type=float, default=0.0)
    parser.add_argument("--end-ms", type=float)
    parser.add_argument("--copy-stream", action="append", default=[])
    parser.add_argument("--copy-bytes", action="append", type=int, default=[], help="Keep copies matching any supplied byte size")
    parser.add_argument("--results-layout", type=Path, help="Derive copy sizes and expected counts from benchmark results.json")
    parser.add_argument("--copy-name-regex")
    parser.add_argument("--kernel-name-regex")
    parser.add_argument("--bin-ms", type=float, default=100.0)
    parser.add_argument("--max-bins", type=int, default=2000)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()
    if args.start_ms < 0 or args.bin_ms <= 0 or args.max_bins < 1 or args.examples < 0:
        parser.error("start-ms/examples must be nonnegative; bin-ms/max-bins must be positive")
    if not math.isfinite(args.start_ms) or not math.isfinite(args.bin_ms) or (args.end_ms is not None and not math.isfinite(args.end_ms)):
        parser.error("Time-window and bin values must be finite")
    if any(size <= 0 for size in args.copy_bytes):
        parser.error("copy-bytes must be positive")
    try:
        validate_output_path(args.output, args.traces, args.results_layout)
    except ValueError as error:
        parser.error(str(error))
    reports = []
    for path in args.traces:
        reports.append(analyze(path, args))
    result = {"schema_version": 1, "analyzer_sha256": sha256(Path(__file__)),
              "scope": "CPU-only temporal analysis of measured CUDA activity", "caveats": CAVEATS,
              "traces": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "traces": len(reports),
                      "trace_file_bytes": sum(report["trace_file_bytes"] for report in reports)}))


if __name__ == "__main__":
    main()
