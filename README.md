# Qwen-Image 2.1 KV-cache offload experiments

Research evidence for [Diffusers issue #14833](https://github.com/huggingface/diffusers/issues/14833), recorded on 12 October 2026. This personal-repository bundle compares cache storage choices and their speed/memory trade-offs. It does not propose an upstream API or imply maintainer approval.

## Recorded environment

- NVIDIA A100-SXM4-80GB, SM80; driver 580.126.16; CUDA runtime 12.8.
- Python 3.12.3, PyTorch 2.8.0+cu128, Diffusers 0.41.0.dev0, Transformers 5.19.0, Accelerate 1.15.0, huggingface_hub 2.2.0, safetensors 0.8.0.
- Diffusers revision: `1e004eb6ec9435349c4a6dd6d1c44c7e0fec6b40`.
- `Qwen/Qwen-Image-2.1` revision: `d26bb61231c349cf6b7896fa83353113880e1ba3`.

All runs use batch 1, BF16, seed `20261012`, `QwenImage21AttnProcessor` with native attention, disabled TF32, and the entire pipeline on CUDA. Inputs use a fixed teapot prompt and deterministic synthetic condition images. There is no compilation or model-weight offload.

## Protocol

The 35 formal cases comprise:

| Suite | Cases | Condition images / CFG | Steps | Warmups / measurements |
| --- | ---: | --- | ---: | --- |
| Main | 16 | 0 or 1 / 1 | 40 | 1 / 3 |
| Additional correctness | 12 | 3 / 1 or 4 | 8 | 0 / 1 |
| Profiles | 5 | 1 / 1 | 8 | 1 / 1 profiled |
| Cache-lifetime control | 2 | 1 / 1 | 40 | 1 / 3 |

Formal resolution is 1024×1024. Main compares no cache, GPU cache, RAM sync/async, and disk sync/async with warm or `drop_hint` reads. Other suites use the subsets recorded in [the evidence bundle](evidence/a100.json). Six smaller pilot cases, six storage tests, and a 36-case random-weight integration suite provide additional, limited correctness evidence.

Each mode runs in a separate process. CUDA is synchronized around full-pipeline timing; loading, warmup and profiler runs are excluded from performance summaries. Timing includes encoding, denoising, final-latent CPU capture, VAE decoding and postprocessing.

**The main GPU reference shares the experimental cleanup before VAE decoding.** Immutable prefixes are stored once and retrieved on subsequent steps. RAM sync uses pageable storage and blocking copies; RAM async uses pinned storage, a copy stream and one-layer prefetch. Disk async adds a background worker and bounded staging. Every disk K/V write is followed by `fsync`.

## Main measurements

One condition image, 40 steps; three measured runs. RSS is the maximum sampled process value.

| Storage | Median seconds | Min–max seconds | RSS GiB |
| --- | ---: | ---: | ---: |
| GPU cache | 17.9348 | 17.9013–17.9882 | 2.1845 |
| No cache | 31.9479 | 31.8621–31.9638 | 2.1835 |
| RAM sync | 24.5701 | 24.4971–24.7685 | 4.1973 |
| RAM async | 17.9449 | 17.9001–17.9906 | 6.1873 |
| Disk sync, warm | 47.5380 | 47.3914–51.5149 | 2.2428 |
| Disk async, warm | 18.3460 | 18.2961–18.4870 | 2.5754 |
| Disk sync, drop hint | 127.3533 | 126.6116–127.7792 | 2.2473 |
| Disk async, drop hint | 105.0577 | 103.9801–105.6168 | 2.5757 |

GPU and RAM-async ranges overlap; the tiny median difference does not establish a speed difference. The text-only cache is just 12.5 MiB, with GPU/RAM-async medians 14.7572/14.7521 seconds. Small-cache results do not predict larger-cache behavior.

The one-image payload is 2,163,736,576 bytes (2.01514 GiB). Each cached request has 32 stores and 1,248 gets; RAM/disk offload transfers 84,385,726,464 H2D bytes, or 39 complete payload reads. Warm disk cases report **zero accounted storage reads** despite that logical I/O volume. Their performance depends on RAM page-cache service. Drop-hint cases account for the full logical read volume; disk async remains about 5.86× GPU-cache latency. Hints do not guarantee device-cold reads, and the scratch filesystem's physical backing is unproven.

Denoising-end allocation falls from 32.2688 to 30.2521 GiB with offload, but main full-pipeline allocated peaks remain approximately 36.81 GiB. RAM-async reserved memory is higher: 40.8008 versus 40.1035 GiB. Allocated and reserved bytes are distinct measures.

RAM-async RSS increases by approximately 4 GiB despite a 2.015-GiB logical payload. Retained allocations are plausible, but pinning/allocator attribution was not directly measured. RSS excludes page cache; whole-container `file` includes `shmem`, so those counters must not be added together. Sampling at 10 ms for RSS and 100 ms for container counters can miss transients. Low disk-mode RSS does not imply equally low total host-memory usage.

## Separate lifetime and overlap controls

Keeping caches resident through VAE produces full/VAE-interval allocated peaks of **38.82943 GiB for GPU cache and 36.81275 GiB for RAM async**, a 2.01668-GiB difference. All three repetitions agree, and all 12 latent/output checks against the main GPU reference are exact. The interval includes unpacking and postprocessing. The factory retains references briefly past return, so this is a lifetime control, not untouched upstream behavior; compare memory, not timing across harnesses.

Residual KV residency can raise VAE-stage memory without VAE using KV caching. This does **not** resolve the issue comment's `enable_model_cpu_offload` interaction: CPU weight offload was not tested.

Eight-step profiles match 448 H2D and 64 D2H copies of 33,808,384 bytes per offload case; GPU caching has no copies of that size. Counts and sizes support KV attribution but cannot exclude coincidentally matching transfers. RAM-async H2D overlap is **97.54% with any kernel and 69.00% with GEMM/FlashAttention**. Disk-async H2D overlap is **0% in its drop-hint trace**, despite a separate copy stream. Both synchronous modes also show zero copy/kernel overlap. Denominators are unions of matched copy intervals, not pipeline time or speedup. Pinning, prefetch and scheduling change together; overlap does not isolate causal speed gains.

## Correctness and limits

All 72 main offload latent/output comparisons are finite and element-exact against GPU caching (`atol=rtol=0`). Additional offload cases are also exact, including separate CFG branches. With three images and CFG 4, one-run allocated peaks are 44.61839/36.85563 GiB for GPU/RAM async; this short suite supports correctness and memory observations, not stable latency claims.

BF16 no-cache differences remain descriptive: main one-image latent/output maxima are 0.09375/0.1171875; the additional CFG-4 latent maximum is 1.265625. There is no human quality-equivalence claim, FP32 quality validation, gradient testing, Flux coverage or weight-offload compatibility result. Three sequential repetitions on one allocation and one prompt family do not establish generality.

## Reproduce and verify

Use Linux, an A100 80GB-class CUDA device, and separate space for model weights, results and scratch. The recorded host limit was 249,999,998,976 bytes RAM and 13.6 CPU cores. Those are the tested limits, not measured minimum requirements. Leave at least 40 GB for the checkpoint cache and 16 GB for scratch; the largest tested CFG cache is approximately 12 GiB. The measured scratch was a container overlay with unverified physical backing; the separate model/results mount was network FUSE.

Install the pinned editable checkout so the harness can record its Git revision:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
git clone https://github.com/huggingface/diffusers.git deps/diffusers
git -C deps/diffusers checkout 1e004eb6ec9435349c4a6dd6d1c44c7e0fec6b40
python -m pip install -r requirements.txt -e deps/diffusers
```

Choose fresh results and local scratch directories on appropriate filesystems. Keep checkpoint downloads outside timed runs:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p results
RESULTS_DIR=$(mktemp -d "$PWD/results/run.XXXXXX")
SCRATCH_DIR=$(mktemp -d /tmp/kv-scratch.XXXXXX)
bash run_experiments.sh "$RESULTS_DIR" "$SCRATCH_DIR"
(cd evidence && sha256sum -c SHA256SUMS)
python verify_submission.py
python -m unittest -v test_analyze_traces
```

The runner records the exact suite commands, checkpoint revision and settings, then analyzes each trace serially with all kernels and a GEMM/FlashAttention filter. Set `KV_DRY_RUN=1` to print commands without execution. Measured `benchmark.py`, `benchmark_lifetime.py` and `storage.py` are preserved byte-for-byte. Recorded GPU results apply to their historical hashes in `evidence/a100.json`; the new reproduction wrapper has not been exercised on a GPU.

## Evidence provenance

`evidence/a100.json` is a sanitized derivative containing measured results, environment details, historical source/output hashes, validation records and trace findings. Measurement values are preserved. Provider, account, billing and private-location metadata are excluded. Original archive and source hashes are retained for provenance; the complete raw archive is retained privately and is not published. `evidence/SHA256SUMS` verifies the public bundle's bytes, not the authenticity of its origin. This repository records completed experiments for review; it makes no upstream approval claim.
