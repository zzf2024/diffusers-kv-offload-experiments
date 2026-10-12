#!/usr/bin/env python3
"""Exercise actual QwenImage21 model cache paths without pretrained weights.

Random weights and small tensors test integration correctness only. They cannot
establish pretrained image quality or representative full-model performance.
"""

import argparse
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path

import torch
import diffusers
from diffusers import QwenImage21Transformer2DModel
from storage import CacheFactory


def source_metadata():
    installed_init = Path(diffusers.__file__).resolve()
    paths = {
        "diffusers_init": installed_init,
        "transformer_qwenimage21": Path(inspect.getfile(QwenImage21Transformer2DModel)).resolve(),
        "storage": Path(inspect.getfile(CacheFactory)).resolve(),
        "smoke_transformer": Path(__file__).resolve(),
    }
    metadata = {
        "diffusers": diffusers.__version__,
        "source_root": None,
        "source_revision": None,
        "source_tracked_dirty": None,
        "source_provenance_error": None,
        "source_files": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()
        },
    }
    try:
        root = subprocess.check_output(
            ["git", "-C", str(installed_init.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.PIPE,
        ).strip()
        metadata["source_root"] = root
        metadata["source_revision"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE,
        ).strip()
        metadata["source_tracked_dirty"] = bool(subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.PIPE,
        ).strip())
    except (OSError, subprocess.CalledProcessError) as error:
        metadata["source_provenance_error"] = str(error)
    return metadata


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.manual_seed(20261012)
    model = QwenImage21Transformer2DModel(in_channels=16, out_channels=16, num_layers=3, attention_head_dim=32, num_attention_heads=4, context_in_dim=128, mlp_ratio=2, axes_dims_rope=(8, 12, 12)).eval().to(device="cuda", dtype=dtype)
    model.set_attention_backend("native")
    results = []
    for condition_count in (0, 1, 3):
        for batch in (1, 2):
            text_length = 8
            condition_tokens = condition_count * 4
            target_tokens = 16
            encoder = torch.randn(batch, text_length + condition_count, 128, device="cuda", dtype=dtype)
            hidden = [torch.randn(batch, condition_tokens + target_tokens, 16, device="cuda", dtype=dtype) for _ in range(3)]
            for item in hidden[1:]:
                item[:, :condition_tokens] = hidden[0][:, :condition_tokens]
            img_mask = torch.cat([torch.zeros(batch, text_length, dtype=torch.bool), torch.ones(batch, condition_count + target_tokens // 4, dtype=torch.bool)], dim=1).cuda()
            shapes = [[(1, 2, 2)] * condition_count + [(1, 4, 4)]] * batch
            reference = None
            for mode in ("gpu", "no_cache", "ram_sync", "ram_async", "disk_sync", "disk_async"):
                factory = CacheFactory(mode if mode != "no_cache" else "gpu", args.cache_dir)
                cache = factory(3) if mode != "no_cache" else None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                outputs = []
                try:
                    for index in range(3):
                        value = model(hidden_states=hidden[index], encoder_hidden_states=encoder, timestep=torch.full((batch,), 1.0 - index / 3, device="cuda", dtype=dtype), img_shapes=shapes, img_mask=img_mask, kv_cache=cache, kv_cache_mode=None if cache is None else ("extract" if index == 0 else "cached"), return_dict=False)[0]
                        outputs.append(value[:, -target_tokens:].clone())
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    peak = torch.cuda.max_memory_allocated()
                    tensor = torch.stack(outputs).float().cpu()
                    if mode == "gpu":
                        reference = tensor
                    delta = tensor - reference
                    exact = torch.equal(tensor, reference)
                    finite = bool(torch.isfinite(tensor).all())
                    record = {"condition_images": condition_count, "batch": batch, "mode": mode, "seconds": elapsed, "peak_gpu_allocated_bytes": peak, "finite": finite, "exact_equal_vs_gpu": exact, "max_abs_vs_gpu": float(delta.abs().max()), "rmse_vs_gpu": float(delta.square().mean().sqrt()), "strict_check": mode != "no_cache", "stats": factory.stats()}
                    results.append(record)
                    print(json.dumps(record), flush=True)
                    if not finite or (mode != "no_cache" and not exact):
                        raise AssertionError(f"Storage integration correctness failed: {record}")
                finally:
                    factory.clear()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "passed", "scope": "Random-weight small-model integration correctness only; timings are not full-model performance evidence.", "seed": 20261012, "dtype": args.dtype, "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "architecture": torch.cuda.get_device_capability(), **source_metadata(), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": results}
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
