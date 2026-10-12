#!/usr/bin/env python3
"""Summarize immutable KV-offload result files without running GPU work.

Inputs are completed or partial suite summary.json files. Raw measurements are
never changed. Output must be outside the input evidence directories.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


LIMITATIONS = [
    "Observational measurements on one allocation and workload; no architecture-wide causal claim.",
    "Async modes combine pinned host memory, a CUDA copy stream and prefetch; sync modes use pageable host memory. Their difference does not isolate overlap.",
    "Pipeline timing excludes model loading, includes common cache cleanup and uniform final-latent capture, and follows each record's latency scope.",
    "Warmup and profiled runs are excluded from performance summaries; all retained correctness comparisons are reported.",
    "RSS excludes Linux page cache. Cgroup counters cover the whole pod and are sampled; file includes shmem, so these fields must not be summed.",
    "POSIX_FADV_DONTNEED is advisory. Process read_bytes measures accounted storage I/O where supported, not guaranteed device-cold reads or KV-only activity.",
    "No-cache numerical differences are descriptive, not an offload correctness failure when finite.",
    "A completed subset in a live summary does not establish completion of the requested full experiment matrix.",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def nested(row, *path):
    for key in path:
        if not isinstance(row, dict):
            return None
        row = row.get(key)
    return row


def per_run_validation(row, mode, cfg_scale, expected_steps):
    issues = []
    run = row.get("run")
    steps = row.get("actual_steps")
    cache = row.get("storage_stats", {})
    if not isinstance(steps, int) or steps < 2:
        return [{"run": run, "problem": "Missing or invalid actual step count"}]
    expected = {"cache_count": 0 if mode == "no_cache" else (2 if cfg_scale > 1 else 1)}
    stores, stored = cache.get("stores", 0), cache.get("stored_bytes", 0)
    layout = nested(row, "phase", "cache_layout")
    if layout is not None:
        expected["stores"] = sum(len(branch["layers"]) for branch in layout)
        dtype_bytes = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.float64": 8}
        try:
            expected["stored_bytes"] = sum(math.prod(tensor["shape"]) * dtype_bytes[tensor["dtype"]]
                for branch in layout for layer in branch["layers"] for tensor in layer["tensors"])
        except KeyError:
            issues.append({"run": run, "problem": "Cannot derive stored bytes from cache layout"})
    expected["gets"] = stores * (steps - 1)
    expected["d2h_bytes"] = stored if mode not in ("gpu", "no_cache") else 0
    expected["h2d_bytes"] = stored * (steps - 1) if mode not in ("gpu", "no_cache") else 0
    expected["disk_write_bytes"] = stored if mode.startswith("disk") else 0
    expected["disk_read_bytes"] = stored * (steps - 1) if mode.startswith("disk") else 0
    if expected_steps is not None and steps != expected_steps:
        issues.append({"run": run, "field": "actual_steps", "actual": steps, "expected": expected_steps})
    for name, wanted in expected.items():
        actual = cache.get(name, 0)
        if actual != wanted:
            issues.append({"run": run, "field": name, "actual": actual, "expected": wanted})
    if cache.get("cleared") is not True:
        issues.append({"run": run, "problem": "Cache cleanup not confirmed"})
    if mode != "no_cache" and (stores <= 0 or stored <= 0):
        issues.append({"run": run, "problem": "Cache enabled without positive stores and stored bytes"})
    for field in ("output_finite", "latent_finite"):
        if row.get(field) is False:
            issues.append({"run": run, "problem": f"Nonfinite {field}"})
    return issues


def summarize_experiment(source, suite, experiment, expected_steps):
    result = experiment.get("results", {})
    records = result.get("records", [])
    measured = [row for row in records if not row.get("warmup") and not row.get("profiled")]
    mode = experiment["mode"]
    cfg = experiment["cfg_scale"]
    row = {"suite": source.parent.name, "model": suite.get("model"), "model_revision": suite.get("model_revision"),
        **{key: experiment.get(key) for key in ("mode", "page_cache", "images", "resolution", "cfg_scale", "returncode", "directory")},
        "status": result.get("status"), "measured_runs": len(measured), "warmup_runs": sum(bool(r.get("warmup")) for r in records),
        "profiled_runs": sum(bool(r.get("profiled")) for r in records),
        "latency_scopes": sorted({r.get("latency_scope", "unspecified") for r in records}), "metrics": {}}
    metrics = row["metrics"]
    for key in ("pipeline_seconds", "denoising_seconds", "encoding_seconds", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "peak_host_rss_bytes"):
        metrics[key] = stats(r.get(key) for r in measured)
    for key in ("encoding_end_allocated_bytes", "denoise_end_allocated_bytes", "post_clear_allocated_bytes", "through_denoising_peak_allocated_bytes", "through_denoising_peak_reserved_bytes", "cleanup_seconds", "latent_capture_seconds"):
        metrics[key] = stats(nested(r, "phase", key) for r in measured)
    for key in ("memory_current_bytes", "anon_bytes", "file_bytes", "shmem_bytes"):
        for phase in ("initial", "peak", "end"):
            metrics[f"cgroup_{key}_{phase}"] = stats(nested(r, "host_memory", "cgroup", key, phase) for r in measured)
        increases = []
        for r in measured:
            initial, peak = (nested(r, "host_memory", "cgroup", key, phase) for phase in ("initial", "peak"))
            if initial is not None and peak is not None:
                increases.append(peak - initial)
        metrics[f"cgroup_{key}_peak_increase"] = stats(increases)
    for key in ("stored_bytes", "stores", "gets", "d2h_bytes", "h2d_bytes", "disk_write_bytes", "disk_read_bytes", "disk_write_seconds", "disk_read_seconds", "max_prefetch_entries", "max_write_queue_entries", "drop_hint_calls"):
        metrics[key] = stats(nested(r, "storage_stats", key) or 0 for r in measured)
    for key in ("read_bytes", "write_bytes", "rchar", "wchar"):
        metrics[f"process_io_{key}"] = stats(nested(r, "process_io_delta", key) for r in measured)
    ratios = []
    for r in measured:
        logical = nested(r, "storage_stats", "disk_read_bytes")
        physical = nested(r, "process_io_delta", "read_bytes")
        if logical and physical is not None:
            ratios.append(physical / logical)
    metrics["accounted_storage_read_to_logical_kv_read_ratio"] = stats(ratios)
    comparisons = experiment.get("correctness_vs_gpu")
    checks = comparisons if isinstance(comparisons, list) else []
    correctness = {"strict_offload_check": mode != "no_cache", "comparison_count": len(checks), "kinds": {}}
    for kind in ("output", "latents"):
        subset = [check for check in checks if check.get("kind") == kind]
        correctness["kinds"][kind] = {"comparisons": len(subset), "all_finite": all(c.get("finite") is True for c in subset) if subset else None,
            "all_exact": all(c.get("exact_equal") is True for c in subset) if subset else None,
            "max_abs": max((c["max_abs"] for c in subset if "max_abs" in c), default=None),
            "max_rmse": max((c["rmse"] for c in subset if "rmse" in c), default=None)}
    if not isinstance(comparisons, list):
        correctness["missing_comparison_reason"] = comparisons
    row["correctness"] = correctness
    issues = [issue for record in records for issue in per_run_validation(record, mode, cfg, expected_steps)]
    expected_pairs = {(record["run"], kind) for record in records if not record.get("warmup") for kind in ("output", "latents")}
    missing = expected_pairs - {(check.get("run"), check.get("kind")) for check in checks}
    if missing:
        issues.append({"problem": "Missing output or latent comparisons", "missing": sorted(missing)})
    for check in checks:
        if check.get("pass") is False:
            issues.append({"problem": "Failed comparison", "run": check.get("run"), "kind": check.get("kind")})
    row["validation_issues"] = issues
    row["evidence_pass"] = experiment.get("returncode") == 0 and result.get("status") == "completed" and bool(measured) and not issues
    remote_dir = Path(experiment.get("directory", ""))
    environment_path = source.parent / remote_dir.parent.name / remote_dir.name / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        row["provenance"] = {"environment_file": str(environment_path), "environment_sha256": digest(environment_path),
            "source_git": environment.get("source_git"), "source_hashes": environment.get("source_hashes"), "gpu": environment.get("gpu")}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    rows, inputs = [], []
    for path in args.summaries:
        path = path.resolve()
        if output == path.parent or path.parent in output.parents:
            parser.error("Output directory must be outside every raw input evidence directory")
        suite = json.loads(path.read_text())
        inputs.append({"path": str(path), "sha256": digest(path), "suite_manifest_exists": (path.parent / "checksums.json").exists()})
        rows.extend(summarize_experiment(path, suite, experiment, args.expected_steps) for experiment in suite["experiments"])
    def group(row):
        return tuple(row[key] for key in ("suite", "model_revision", "images", "resolution", "cfg_scale"))
    baselines = {group(row): row for row in rows if row["mode"] == "gpu" and row["evidence_pass"]}
    for row in rows:
        baseline = baselines.get(group(row))
        if not baseline:
            row["comparison_to_gpu"] = None
            continue
        comparison = {}
        for name, reducer in (("pipeline_seconds", "median"), ("peak_gpu_allocated_bytes", "max"), ("peak_gpu_reserved_bytes", "max"), ("peak_host_rss_bytes", "max"), ("denoise_end_allocated_bytes", "median")):
            actual, reference = row["metrics"].get(name), baseline["metrics"].get(name)
            if actual and reference:
                delta = actual[reducer] - reference[reducer]
                comparison[f"{name}_{reducer}_delta"] = delta
                comparison[f"{name}_{reducer}_change_percent"] = 100 * delta / reference[reducer] if reference[reducer] else None
        row["comparison_to_gpu"] = comparison
    output.mkdir(parents=True, exist_ok=True)
    report = {"analysis_script_sha256": digest(Path(__file__)), "inputs": inputs, "expected_steps": args.expected_steps,
        "comparison_convention": "Positive latency or memory change means an increase versus GPU caching in the same suite and workload.",
        "limitations": LIMITATIONS, "rows": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    flat = []
    for row in rows:
        item = {key: row[key] for key in ("suite", "mode", "page_cache", "images", "resolution", "cfg_scale", "status", "evidence_pass", "measured_runs", "warmup_runs", "profiled_runs")}
        csv_metrics = {
            "pipeline_seconds": ("min", "median", "max"),
            "denoising_seconds": ("median",),
            "peak_gpu_allocated_bytes": ("max",),
            "peak_gpu_reserved_bytes": ("max",),
            "peak_host_rss_bytes": ("max",),
            "cgroup_memory_current_bytes_peak": ("max",),
            "cgroup_anon_bytes_peak": ("max",),
            "cgroup_file_bytes_peak": ("max",),
            "cgroup_file_bytes_peak_increase": ("max",),
            "denoise_end_allocated_bytes": ("median",),
            "stored_bytes": ("median",),
            "stores": ("median",),
            "gets": ("median",),
            "d2h_bytes": ("median",),
            "h2d_bytes": ("median",),
            "disk_write_bytes": ("median",),
            "disk_read_bytes": ("median",),
            "process_io_read_bytes": ("median",),
            "process_io_write_bytes": ("median",),
            "accounted_storage_read_to_logical_kv_read_ratio": ("median",),
        }
        for metric, selected_statistics in csv_metrics.items():
            values = row["metrics"].get(metric)
            for statistic in selected_statistics:
                item[f"{metric}_{statistic}"] = values[statistic] if values else None
        item.update(row["comparison_to_gpu"] or {})
        for kind, values in row["correctness"]["kinds"].items():
            for name, value in values.items():
                item[f"{kind}_{name}"] = value
        item["validation_issue_count"] = len(row["validation_issues"])
        flat.append(item)
    fieldnames = list(dict.fromkeys(key for row in flat for key in row))
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"rows": len(rows), "evidence_pass": sum(row["evidence_pass"] for row in rows), "output_directory": str(output)}))


if __name__ == "__main__":
    main()
