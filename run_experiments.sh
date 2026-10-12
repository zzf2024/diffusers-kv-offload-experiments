#!/usr/bin/env bash
# Portable orchestration of the preserved measurement scripts.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash run_experiments.sh FRESH_RESULTS_DIR FRESH_LOCAL_SCRATCH_DIR" >&2
    exit 2
fi

runner_python="${KV_PYTHON:-python3}"
runner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
results_dir=$("$runner_python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")
scratch_dir=$("$runner_python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$2")

"$runner_python" - "$results_dir" "$scratch_dir" <<'PY'
from pathlib import Path
import sys

results, scratch = map(Path, sys.argv[1:])
if results == scratch or results in scratch.parents or scratch in results.parents:
    raise SystemExit("Results and scratch must be separate, non-nested directories.")
for path in (results, scratch):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"Refusing to overwrite a nonempty path: {path}")
    path.mkdir(parents=True, exist_ok=True)
PY

cd -- "$runner_root"
export TMPDIR="$scratch_dir"
revision=d26bb61231c349cf6b7896fa83353113880e1ba3
common=(--model-revision "$revision" --cache-dir "$scratch_dir" --output-type np)

run() {
    printf '%q ' "$@" | tee -a "$results_dir/commands.txt"
    printf '\n' | tee -a "$results_dir/commands.txt"
    if [[ ${KV_DRY_RUN:-0} != 1 ]]; then
        "$@"
    fi
}

if [[ ${KV_DRY_RUN:-0} != 1 ]]; then
    "$runner_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required; no GPU measurements were executed.")
if torch.cuda.get_device_capability()[0] < 8:
    raise SystemExit("This BF16 protocol requires an Ampere-or-newer CUDA device.")
PY
fi

run "$runner_python" -m unittest -v test_storage
run "$runner_python" smoke_transformer_metadata.py --output "$results_dir/smoke.json" --cache-dir "$scratch_dir" --dtype bf16
run "$runner_python" benchmark.py "${common[@]}" --output "$results_dir/main" --image-counts 0,1 --resolutions 1024 --cfg-scales 1 --steps 40 --warmup 1 --repeats 3 --disk-regimes warm,drop_hint
run "$runner_python" benchmark.py "${common[@]}" --output "$results_dir/multi-image-cfg" --image-counts 3 --resolutions 1024 --cfg-scales 1,4 --steps 8 --warmup 0 --repeats 1 --disk-regimes warm
run "$runner_python" benchmark.py "${common[@]}" --output "$results_dir/profile" --image-counts 1 --resolutions 1024 --cfg-scales 1 --steps 8 --warmup 1 --repeats 1 --modes gpu,ram_sync,ram_async,disk_sync,disk_async --disk-regimes drop_hint --profile
run "$runner_python" benchmark_lifetime.py "${common[@]}" --output "$results_dir/lifetime" --image-counts 1 --resolutions 1024 --cfg-scales 1 --steps 40 --warmup 1 --repeats 3 --modes gpu,ram_async --cache-lifetime after_pipeline

if [[ ${KV_DRY_RUN:-0} != 1 ]]; then
"$runner_python" - "$results_dir" <<'PY'
from pathlib import Path
import sys
import torch
from benchmark import compare, sha256, write_json

root = Path(sys.argv[1])
checks = []
for mode in ("gpu", "ram_async"):
    for run in range(3):
        for kind in ("output", "latents"):
            actual = root / "lifetime/images1_res1024_cfg1" / f"{mode}_warm" / f"{kind}_{run}.pt"
            reference = root / "main/images1_res1024_cfg1/gpu_warm" / f"{kind}_0.pt"
            checks.append({"mode": mode, "run": run, "kind": kind,
                           "actual": str(actual), "reference": str(reference),
                           "actual_sha256": sha256(actual), "reference_sha256": sha256(reference),
                           **compare(actual, reference, torch, strict=True)})
write_json(root / "lifetime-vs-main.json", checks)
if not all(check["pass"] for check in checks):
    raise SystemExit("Lifetime cross-check failed; measured diagnostics were retained.")
PY
fi

run "$runner_python" analyze_results.py "$results_dir/main/summary.json" "$results_dir/multi-image-cfg/summary.json" "$results_dir/lifetime/summary.json" --output-dir "$results_dir/analysis"

# Parse traces sequentially to bound host-memory use after GPU measurements.
for mode in gpu_warm ram_sync_warm ram_async_warm disk_sync_drop_hint disk_async_drop_hint; do
    trace_dir="$results_dir/profile/images1_res1024_cfg1/$mode"
    run "$runner_python" analyze_traces.py "$trace_dir/trace.json" --results-layout "$trace_dir/results.json" --output "$results_dir/analysis-traces/$mode.all_kernels.json"
    run "$runner_python" analyze_traces.py "$trace_dir/trace.json" --results-layout "$trace_dir/results.json" --kernel-name-regex '(?i)(gemm|flash_fwd_kernel)' --output "$results_dir/analysis-traces/$mode.gemm_flash_attention.json"
done

if [[ ${KV_DRY_RUN:-0} == 1 ]]; then
    echo "Dry run only: commands printed; no tests, model downloads, analyses, or GPU work executed."
fi
