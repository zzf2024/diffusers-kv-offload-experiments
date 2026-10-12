#!/usr/bin/env python3
"""Check the selected evidence and preserved sources without running GPU work."""

import hashlib
import json
from pathlib import Path
import re


EXPECTED_CASES = {"main": 16, "multi-image-cfg": 12, "profile": 5, "lifetime": 2, "pilot": 6}
EXPECTED_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
SOURCE_CASES = {
    "benchmark.py": "main/images1_res1024_cfg1/gpu_warm",
    "benchmark_lifetime.py": "lifetime/images1_res1024_cfg1/gpu_warm",
    "storage.py": "main/images1_res1024_cfg1/gpu_warm",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    data = (root / "evidence/a100.json").read_bytes()
    bundle = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)
    expected_sum = f"{digest(data)}  a100.json\n"
    require((root / "evidence/SHA256SUMS").read_text() == expected_sum, "Bundle checksum mismatch")
    require(bundle["schema_version"] == 1, "Unexpected schema")
    require(bundle["gpu"] == "NVIDIA A100-SXM4-80GB", "Unexpected recorded device")
    require(bundle["suite_case_counts"] == EXPECTED_CASES, "Case inventory changed")
    artifacts = bundle["artifacts"]
    require(len(artifacts) == 229, "Missing or unexpected provenance artifacts")
    for name, artifact in artifacts.items():
        require(re.fullmatch(r"[0-9a-f]{64}", artifact["original_sha256"]), f"Missing original hash: {name}")
        content = artifact["content"]
        require(artifact["format"] in ("json", "text"), f"Unexpected artifact format: {name}")
        encoded = (content.encode("utf-8") if artifact["format"] == "text" else
                   json.dumps(content, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        require(digest(encoded) == artifact["content_sha256"], f"Derivative content changed: {name}")

    for source, case in SOURCE_CASES.items():
        env = artifacts[f"evidence/results/{case}/environment.json"]["content"]
        recorded = env["source_hashes"][f"/workspace/kv-lab/{source}"]
        require(digest((root / source).read_bytes()) == recorded, f"Measured source changed: {source}")

    counts = {"cases": 0, "records": 0, "warmups": 0, "profiled": 0,
              "measured": 0, "comparisons": 0, "strict": 0, "descriptive_nonexact": 0}
    for suite, expected_cases in EXPECTED_CASES.items():
        summary = artifacts[f"evidence/results/{suite}/summary.json"]["content"]
        require(summary["model_revision"] == EXPECTED_REVISION, f"Checkpoint changed: {suite}")
        require(len(summary["experiments"]) == expected_cases, f"Incomplete suite: {suite}")
        seen = set()
        for case in summary["experiments"]:
            require(case["returncode"] == 0, f"Recorded experiment failed: {case['directory']}")
            relative = Path(case["directory"]).relative_to("/workspace/kv-lab")
            key = f"evidence/{relative.as_posix()}"
            require(key not in seen, f"Duplicate case: {key}")
            seen.add(key)
            results = artifacts[f"{key}/results.json"]["content"]
            config = artifacts[f"{key}/configuration.json"]["content"]
            checks = artifacts[f"{key}/comparison.json"]["content"]
            records = results["records"]
            require(results["status"] == "completed", f"Incomplete record: {key}")
            require(len(records) == config["warmup"] + config["repeats"], f"Missing repeat: {key}")
            require({r["run"] for r in records} == set(range(-config["warmup"], config["repeats"])), f"Repeat IDs changed: {key}")
            require({(c["run"], c["kind"]) for c in checks} ==
                    {(run, kind) for run in range(config["repeats"]) for kind in ("output", "latents")}, f"Incomplete comparisons: {key}")
            require(len(checks) == 2 * config["repeats"], f"Duplicate comparisons: {key}")
            for record in records:
                require(record["actual_steps"] == config["steps"], f"Incorrect step count: {key}")
                require(record["storage_stats"]["cleared"], f"Cache cleanup missing: {key}")
                counts["records"] += 1
                if record["warmup"]:
                    counts["warmups"] += 1
                elif record["profiled"]:
                    counts["profiled"] += 1
                else:
                    counts["measured"] += 1
            for check in checks:
                require(check["finite"], f"Non-finite output: {key}")
                counts["comparisons"] += 1
                if check["strict_offload_check"]:
                    require(check["pass"] is True and check["exact_equal"], f"Offload mismatch: {key}")
                    require(check["atol"] == 0 and check["rtol"] == 0, f"Relaxed tolerance: {key}")
                    counts["strict"] += 1
                elif not check["exact_equal"]:
                    counts["descriptive_nonexact"] += 1
            counts["cases"] += 1

    require(counts == {"cases": 41, "records": 100, "warmups": 23, "profiled": 5,
                       "measured": 72, "comparisons": 154, "strict": 136,
                       "descriptive_nonexact": 12}, f"Unexpected evidence coverage: {counts}")
    cross = artifacts["evidence/results/lifetime-vs-main.json"]["content"]
    require(len(cross) == 12 and all(c["pass"] and c["exact_equal"] for c in cross), "Lifetime cross-checks changed")
    traces = artifacts["analysis-traces/summary.json"]["content"]["results"]
    require(len(traces) == 5 and all(t["counts_match_expected"] for t in traces), "Trace copy counts changed")
    for trace in traces:
        expected = (0, 0) if trace["mode"] == "gpu_warm" else (448, 64)
        require(tuple(trace["directions"][d]["count"] for d in ("H2D", "D2H")) == expected, "Unexpected trace coverage")
    return counts


if __name__ == "__main__":
    try:
        counts = verify(Path(__file__).resolve().parent)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}")
    print(json.dumps({"pass": True, **counts}))
    print("Checks establish internal consistency and recorded source identity, not a new GPU run or independent authenticity.")
