#!/usr/bin/env python3
"""Isolated Qwen-Image 2.1 KV storage experiment, with raw per-run evidence."""

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


MODES = ("no_cache", "gpu", "ram_sync", "ram_async", "disk_sync", "disk_async")


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {"command": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"command": args, "error": repr(exc)}


def proc_io():
    try:
        return {key: int(value) for key, value in (line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())}
    except OSError:
        return {}


class RSSSampler:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.samples = []
        self.cgroup_samples = []

    def sample_cgroup(self):
        root = Path("/sys/fs/cgroup")
        try:
            current = int((root / "memory.current").read_text())
            stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
            self.cgroup_samples.append({"time": time.perf_counter(), "memory_current_bytes": current, **{f"{key}_bytes": int(stats[key]) for key in ("anon", "file", "shmem") if key in stats}})
        except (OSError, ValueError):
            pass

    def sample(self):
        last_cgroup = 0.0
        while not self.stop_event.is_set():
            now = time.perf_counter()
            self.samples.append((now, self.process.memory_info().rss))
            if now - last_cgroup >= 0.1:
                self.sample_cgroup()
                last_cgroup = now
            self.stop_event.wait(0.01)

    def __enter__(self):
        self.samples.append((time.perf_counter(), self.process.memory_info().rss))
        self.sample_cgroup()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.samples.append((time.perf_counter(), self.process.memory_info().rss))
        self.sample_cgroup()

    @property
    def peak(self):
        return max((rss for _, rss in self.samples), default=self.process.memory_info().rss)

    def summary(self):
        result = {"rss_scope": "Current Python worker process, excluding Linux file page cache.", "rss_interval_seconds": 0.01, "rss_initial_bytes": self.samples[0][1], "rss_end_bytes": self.samples[-1][1], "rss_peak_bytes": self.peak, "cgroup_scope": "Whole pod cgroup: all processes plus file page cache; not isolated to this Python worker. file includes shmem, so these counters must not be summed.", "cgroup_interval_seconds": 0.1, "cgroup_available": bool(self.cgroup_samples)}
        if self.cgroup_samples:
            keys = [key for key in self.cgroup_samples[0] if key != "time"]
            result["cgroup"] = {key: {"initial": self.cgroup_samples[0][key], "end": self.cgroup_samples[-1][key], "peak": max(row[key] for row in self.cgroup_samples if key in row)} for key in keys}
            result["cgroup_samples"] = [{**row, "time": row["time"] - self.cgroup_samples[0]["time"]} for row in self.cgroup_samples]
        return result


def environment(args, torch, diffusers):
    source = Path(diffusers.__file__).resolve()
    props = torch.cuda.get_device_properties(0)
    packages = {}
    for package in ("torch", "diffusers", "transformers", "accelerate", "huggingface_hub", "safetensors", "psutil"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[package] = importlib.metadata.version(package)
    tracked = [Path(__file__), Path(__file__).with_name("storage.py"), source.parent / "models/transformers/transformer_qwenimage21.py", source.parent / "pipelines/qwenimage21/pipeline_qwenimage21.py"]
    git_root = source.parent.parent.parent
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": {"name": props.name, "architecture": list(torch.cuda.get_device_capability(0)), "total_memory_bytes": props.total_memory, "multiprocessors": props.multi_processor_count},
        "source_hashes": {str(p): sha256(p) for p in tracked if p.exists()},
        "source_git": command(["git", "-C", str(git_root), "rev-parse", "HEAD"]),
        "source_status": command(["git", "-C", str(git_root), "status", "--porcelain"]),
        "nvidia_smi": command(["nvidia-smi", "-q"]),
        "cpu": command(["lscpu"]),
        "block_devices": command(["lsblk", "-o", "NAME,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS"]),
        "cache_filesystem": command(["findmnt", "-T", str(args.cache_dir)]),
        "cgroup_limits": {name: Path("/sys/fs/cgroup", name).read_text().strip() for name in ("memory.max", "cpu.max", "memory.swap.max") if Path("/sys/fs/cgroup", name).exists()},
        "pip_freeze": command([sys.executable, "-m", "pip", "freeze"]),
        "model_id": args.model,
        "model_revision": args.model_revision,
        "methodology": {
            "timing": "CUDA synchronized before and after each measured pipeline call; warmup excluded; loading excluded; transformer CUDA events collected without per-step synchronization.",
            "host_memory": "Process RSS sampled every 10 ms; whole-pod cgroup memory.current and memory.stat anon/file/shmem every 100 ms. Short transients can be missed. Linux file page cache is outside RSS and included in the cgroup file counter. No concurrent model download permitted during measured runs.",
            "disk": "Process /proc/self/io read_bytes distinguishes physical block I/O from cached reads where supported; POSIX_FADV_DONTNEED is only a hint, never proof of cold disk.",
            "cleanup": "All cache modes are cleared after last denoising step before VAE. The reported baseline includes this experimental common cleanup.",
            "correctness": "Offload results require exact element equality of pre-VAE latents and final outputs to GPU-cache reference. no_cache differences are descriptive: upstream documents reduced-precision layout-dependent divergence. Latent D2H capture is performed once in every mode, with its elapsed time recorded and included uniformly in pipeline latency.",
            "gradients": "Not tested: inference-only KV cache experiment under torch.inference_mode().",
            "weights": "Entire pipeline resides on CUDA; no weight quantization or model weight offload.",
        },
    }


def condition_images(count, size, directory):
    import numpy as np
    from PIL import Image
    result, evidence = [], []
    yy, xx = np.indices((size, size))
    for index in range(count):
        array = np.stack(((xx * 255 // size + 43 * index) % 256, (yy * 255 // size + 71 * index) % 256, ((xx // 32 + yy // 32 + index) % 2) * 255), axis=-1).astype("uint8")
        image = Image.fromarray(array, "RGB")
        path = directory / f"condition_{index}.png"
        image.save(path)
        result.append(image)
        evidence.append({"path": str(path), "sha256": sha256(path), "size": image.size})
    return result, evidence


def compare(path, reference_path, torch, strict=True):
    reference = torch.load(reference_path, map_location="cpu", weights_only=True).float()
    actual = torch.load(path, map_location="cpu", weights_only=True).float()
    if actual.shape != reference.shape:
        return {"shape_equal": False, "actual": list(actual.shape), "reference": list(reference.shape), "pass": False}
    delta = actual - reference
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    exact = bool(torch.equal(actual, reference))
    result = {"shape_equal": True, "shape": list(actual.shape), "finite": finite, "exact_equal": exact, "atol": 0.0 if strict else None, "rtol": 0.0 if strict else None, "strict_offload_check": strict, "pass": (finite and exact if strict else None) if finite else False}
    if finite:
        result.update(max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()), rmse=float(delta.square().mean().sqrt()), relative_l2=float(delta.norm() / reference.norm().clamp_min(1e-12)))
    return result


def worker(args):
    import numpy as np
    import torch
    import diffusers
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
    from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21
    from storage import CacheFactory

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(directory / "environment.json", environment(args, torch, diffusers))
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    load_started = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, revision=args.model_revision, torch_dtype=dtype)
    pipe.to("cuda")
    pipe.transformer.set_attn_processor(QwenImage21AttnProcessor())
    pipe.transformer.set_attention_backend("native")
    pipe.set_progress_bar_config(disable=True)
    if not pipe.transformer.config.causal_condition:
        raise RuntimeError("This experiment requires a checkpoint with causal_condition enabled.")
    load_seconds = time.perf_counter() - load_started
    images, image_evidence = condition_images(args.image_count, args.resolution, directory)
    original_cache = pipeline_qwenimage21.QwenImage21KVCache
    factory = CacheFactory(args.mode if args.mode != "no_cache" else "gpu", disk_dir=args.cache_dir, page_cache=args.page_cache)
    pipeline_qwenimage21.QwenImage21KVCache = factory
    settings = vars(args).copy()
    settings["cache_dir"] = str(args.cache_dir)
    write_json(directory / "configuration.json", {**settings, "input_images": image_evidence, "model_load_seconds": load_seconds, "attention_processor": "QwenImage21AttnProcessor", "attention_backend": "native", "batch_size": 1, "explicit_sigma_schedule": list(np.linspace(1.0, 1 / args.steps, args.steps))})

    records = []
    try:
        for run in range(-args.warmup, args.repeats):
            factory.clear()
            factory = CacheFactory(args.mode if args.mode != "no_cache" else "gpu", disk_dir=args.cache_dir, page_cache=args.page_cache)
            pipeline_qwenimage21.QwenImage21KVCache = factory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before_io = proc_io()
            before_stats = factory.stats()
            forward_events, current_events = [], []
            phase = {"first_forward_wall": None, "last_step_wall": None, "cleanup_seconds": None}
            captured_latents = []

            def forward_pre(module, positional, keyword):
                if phase["first_forward_wall"] is None:
                    phase["first_forward_wall"] = time.perf_counter()
                    phase["encoding_end_allocated_bytes"] = torch.cuda.memory_allocated()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                current_events.append((start, end, keyword.get("kv_cache_mode"), time.perf_counter()))

            def forward_post(module, positional, keyword, output):
                start, end, cache_mode, cpu_start = current_events.pop()
                end.record()
                forward_events.append((start, end, cache_mode, time.perf_counter() - cpu_start))

            handles = [pipe.transformer.register_forward_pre_hook(forward_pre, with_kwargs=True), pipe.transformer.register_forward_hook(forward_post, with_kwargs=True)]

            def callback(pipeline, index, timestep, callback_kwargs):
                if index + 1 == pipeline.num_timesteps:
                    torch.cuda.synchronize()
                    phase["last_step_wall"] = time.perf_counter()
                    phase["through_denoising_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    phase["through_denoising_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    phase["denoise_end_allocated_bytes"] = torch.cuda.memory_allocated()
                    phase["cache_stats_before_clear"] = factory.stats()
                    phase["cache_layout"] = [{"cache_index": cache_index, "layers": [{"layer": layer, "tensors": [{"shape": list(shape), "dtype": str(entry_dtype)} for shape, entry_dtype in entry["metadata"]]} for layer, entry in cache.entries.items()]} for cache_index, cache in enumerate(factory.instances)]
                    cleanup_start = time.perf_counter()
                    factory.clear()
                    torch.cuda.synchronize()
                    phase["cleanup_seconds"] = time.perf_counter() - cleanup_start
                    phase["post_clear_allocated_bytes"] = torch.cuda.memory_allocated()
                    capture_start = time.perf_counter()
                    captured_latents.append(callback_kwargs["latents"].detach().to(device="cpu", copy=True))
                    phase["latent_capture_seconds"] = time.perf_counter() - capture_start
                return callback_kwargs

            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) if args.profile and run == 0 else contextlib.nullcontext()
            start_wall = time.perf_counter()
            try:
                with RSSSampler() as rss, torch.inference_mode(), profiler:
                    result = pipe(
                        prompt=args.prompt,
                        image=images or None,
                        negative_prompt="blurry, low quality" if args.cfg_scale > 1 else None,
                        true_cfg_scale=args.cfg_scale,
                        height=args.resolution,
                        width=args.resolution,
                        output_resolution=args.resolution,
                        num_inference_steps=args.steps,
                        sigmas=list(np.linspace(1.0, 1 / args.steps, args.steps)),
                        num_images_per_prompt=1,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        output_type=args.output_type,
                        use_kv_cache=args.mode != "no_cache",
                        callback_on_step_end=callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_wall
            finally:
                for handle in handles:
                    handle.remove()
            after_io = proc_io()
            after_stats = factory.stats()
            cache_count = 0 if args.mode == "no_cache" else (2 if args.cfg_scale > 1 else 1)
            expected_stores = cache_count * len(pipe.transformer.transformer_blocks)
            expected_gets = expected_stores * (args.steps - 1)
            if pipe.num_timesteps != args.steps or after_stats.get("stores", 0) != expected_stores or after_stats.get("gets", 0) != expected_gets:
                raise RuntimeError(f"Cache lifecycle was not exercised as configured: steps={pipe.num_timesteps}, stores={after_stats.get('stores', 0)}/{expected_stores}, gets={after_stats.get('gets', 0)}/{expected_gets}")
            record = {
                "run": run,
                "warmup": run < 0,
                "profiled": bool(args.profile and run == 0),
                "pipeline_seconds": elapsed,
                "latency_scope": "Text/image encoding, denoising, cache cleanup and uniform latent capture; VAE decode excluded." if args.output_type == "latent" else "Text/image encoding, denoising, cache cleanup, uniform latent capture, VAE decode and image postprocessing.",
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_host_rss_bytes": rss.peak,
                "host_memory": rss.summary(),
                "denoising_seconds": phase["last_step_wall"] - phase["first_forward_wall"] if phase["last_step_wall"] is not None else None,
                "encoding_seconds": phase["first_forward_wall"] - start_wall if phase["first_forward_wall"] is not None else None,
                "phase": phase,
                "process_io_delta": {key: after_io[key] - before_io.get(key, 0) for key in after_io},
                "storage_stats_delta": {key: value - before_stats.get(key, 0) for key, value in after_stats.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and not key.startswith("max_") and key not in ("prefetch_depth",)},
                "storage_stats": after_stats,
                "disk_io_trace": factory.io_trace(),
                "forwards": [{"mode": mode, "cuda_ms": begin.elapsed_time(end), "cpu_submit_seconds": cpu_seconds} for begin, end, mode, cpu_seconds in forward_events],
                "actual_steps": pipe.num_timesteps,
            }
            if run >= 0:
                tensor = result.detach().cpu() if isinstance(result, torch.Tensor) else torch.from_numpy(np.asarray(result))
                output_path = directory / f"output_{run}.pt"
                torch.save(tensor, output_path)
                record.update(output_file=str(output_path), output_sha256=sha256(output_path), output_shape=list(tensor.shape), output_dtype=str(tensor.dtype), output_finite=bool(torch.isfinite(tensor).all()))
                latent_path = directory / f"latents_{run}.pt"
                if len(captured_latents) != 1:
                    raise RuntimeError("Expected exactly one final latent capture.")
                torch.save(captured_latents[0], latent_path)
                record.update(latent_file=str(latent_path), latent_sha256=sha256(latent_path), latent_shape=list(captured_latents[0].shape), latent_dtype=str(captured_latents[0].dtype), latent_finite=bool(torch.isfinite(captured_latents[0]).all()))
                if args.output_type == "np":
                    from PIL import Image
                    png_path = directory / f"image_{run}.png"
                    Image.fromarray((np.asarray(result[0]).clip(0, 1) * 255).round().astype("uint8")).save(png_path)
                    record.update(image_file=str(png_path), image_sha256=sha256(png_path))
            if args.profile and run == 0:
                profiler.export_chrome_trace(str(directory / "trace.json"))
            records.append(record)
            write_json(directory / "results.json", {"mode": args.mode, "status": "running", "records": records})
            print(json.dumps({"mode": args.mode, "run": run, "seconds": elapsed, "gpu_gib": record["peak_gpu_allocated_bytes"] / 2**30}), flush=True)
            del result
        measured = [row for row in records if not row["warmup"] and not row["profiled"]]
        summary = {"median_pipeline_seconds": statistics.median(row["pipeline_seconds"] for row in measured) if measured else None, "timing_repeats": len(measured), "max_gpu_allocated_bytes": max((row["peak_gpu_allocated_bytes"] for row in measured), default=None)}
        write_json(directory / "results.json", {"mode": args.mode, "status": "completed", "summary": summary, "records": records})
    finally:
        factory.clear()
        pipeline_qwenimage21.QwenImage21KVCache = original_cache


def suite(args):
    from huggingface_hub import model_info
    import torch
    args.model_revision = model_info(args.model, revision=args.model_revision or "main").sha
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    modes = args.modes.split(",")
    if any(mode not in MODES for mode in modes):
        raise ValueError(f"Unknown modes: {modes}")
    if "gpu" in modes:
        modes.remove("gpu")
        modes.insert(0, "gpu")
    for count in map(int, args.image_counts.split(",")):
        for resolution in map(int, args.resolutions.split(",")):
            for cfg in map(float, args.cfg_scales.split(",")):
                workload = root / f"images{count}_res{resolution}_cfg{cfg:g}"
                for mode in modes:
                    regimes = args.disk_regimes.split(",") if mode.startswith("disk_") else ["warm"]
                    for regime in regimes:
                        target = workload / f"{mode}_{regime}"
                        target.mkdir(parents=True, exist_ok=True)
                        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", args.model, "--model-revision", args.model_revision, "--mode", mode, "--page-cache", regime, "--image-count", str(count), "--resolution", str(resolution), "--cfg-scale", str(cfg), "--steps", str(args.steps), "--seed", str(args.seed), "--warmup", str(args.warmup), "--repeats", str(args.repeats), "--dtype", args.dtype, "--output-type", args.output_type, "--prompt", args.prompt, "--output", str(target), "--cache-dir", str(args.cache_dir / f"images{count}_res{resolution}_cfg{cfg:g}_{mode}_{regime}")]
                        if args.profile:
                            cmd.append("--profile")
                        write_json(target / "command.json", {"argv": cmd})
                        with (target / "stdout.log").open("w") as output:
                            completed = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
                        item = {"mode": mode, "page_cache": regime, "images": count, "resolution": resolution, "cfg_scale": cfg, "returncode": completed.returncode, "directory": str(target)}
                        result_file = target / "results.json"
                        if result_file.exists():
                            item["results"] = json.loads(result_file.read_text())
                        if completed.returncode == 0:
                            reference = workload / "gpu_warm" / "output_0.pt"
                            if reference.exists():
                                item["correctness_vs_gpu"] = []
                                for run in range(args.repeats):
                                    for kind, baseline in (("output", reference), ("latents", workload / "gpu_warm" / "latents_0.pt")):
                                        item["correctness_vs_gpu"].append({"run": run, "kind": kind, **compare(target / f"{kind}_{run}.pt", baseline, torch, strict=mode != "no_cache")})
                                write_json(target / "comparison.json", item["correctness_vs_gpu"])
                            else:
                                item["correctness_vs_gpu"] = "Reference not available; not validated."
                        results.append(item)
                        write_json(root / "summary.json", {"model": args.model, "model_revision": args.model_revision, "experiments": results})
                        print(json.dumps({key: value for key, value in item.items() if key not in ("results", "correctness_vs_gpu")}), flush=True)
                        checks = item.get("correctness_vs_gpu", [])
                        if completed.returncode or (isinstance(checks, list) and any(row.get("pass") is False for row in checks)):
                            if not args.continue_on_failure:
                                raise RuntimeError(f"Experiment failed; retained diagnostics in {target}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.json")
    write_json(root / "checksums.json", {str(path.relative_to(root)): sha256(path) for path in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--model-revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--image-counts", default="0,1,3")
    parser.add_argument("--resolutions", default="1024")
    parser.add_argument("--cfg-scales", default="1")
    parser.add_argument("--disk-regimes", default="warm,drop_hint")
    parser.add_argument("--mode", choices=MODES, default="gpu")
    parser.add_argument("--image-count", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--page-cache", choices=("warm", "drop_hint"), default="warm")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output-type", choices=("latent", "np"), default="np")
    parser.add_argument("--prompt", default="A red ceramic teapot on a wooden table, soft daylight, detailed studio photograph.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error("Require at least two denoising steps, one repeat, and nonnegative warmup.")
    try:
        worker(args) if args.worker else suite(args)
    except Exception:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
