# CUDA / GPU configuration reference

Every GPU feature in colibrì is **opt-in at runtime**: a CUDA-linked binary runs
identically to the pure-C build until `COLI_CUDA=1` is set. This page lists all
build-time (Makefile) and runtime (environment) variables that affect the CUDA
backend, plus the expert-placement variables it interacts with.

## How the pieces fit

The model splits into a **dense resident part** (attention, embed/lm_head, dense
MLP, shared expert, norms — always in memory) and the **routed MoE experts**
(three-tier placement: VRAM / pinned RAM / streamed from NVMe).

| Tier | Lives in | Computes on (discrete GPU) | Computes on (unified memory) |
|------|----------|---------------------------|------------------------------|
| Hot experts (`CUDA_EXPERT_GB`, `PIN`) | VRAM copy / RAM | CUDA | CUDA (zero-copy, no VRAM copy) |
| Pinned + LRU-cached experts | RAM | CPU | CUDA (zero-copy) |
| Streamed experts (cache miss) | NVMe → RAM | CPU | CUDA (zero-copy, after the same NVMe read) |
| Dense part + shared expert | RAM | CPU, or CUDA with `CUDA_DENSE=1` | same, zero-copy |

On **discrete GPUs** streamed experts deliberately stay on the CPU: copying an
expert over PCIe on every use would replace the disk bottleneck with a PCIe
bottleneck. On **coherent unified memory** (NVIDIA GB10 / DGX Spark, Grace
systems — auto-detected via `cudaDevAttrPageableMemoryAccess`) there is no copy
to avoid: the GPU reads host memory in place, so *every* loaded expert is
GPU-served and no tensor is ever duplicated into VRAM. See
[Unified memory](#unified-memory-dgx-spark--gb10-grace) below.

## Build-time (Makefile) variables

Used as `make <target> VAR=value`. The default build stays pure C / zero-dependency.

| Variable | Default | Meaning |
|----------|---------|---------|
| `CUDA` | `0` | `CUDA=1` compiles and links the CUDA backend (`backend_cuda.cu`, `-DCOLI_CUDA`). Linux only. |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA Toolkit root; sets the default `nvcc` and `-L$(CUDA_HOME)/lib64` rpath. |
| `NVCC` | `$(CUDA_HOME)/bin/nvcc` | Explicit compiler override. |
| `CUDA_ARCH` | `native` | GPU architecture (`-arch=`). `native` targets the GPU in this machine; set e.g. `sm_121` (GB10) or `sm_120` (RTX 5090) when cross-compiling. |
| `NVCCFLAGS` | `-O3 -std=c++17 -arch=$(CUDA_ARCH) …` | Full nvcc flag override. |
| `ARCH` | `native` | Host CPU flags (`-march=`/`-mcpu=`); `x86-64-v3` for a portable AVX2 binary. |
| `METAL` | `0` | Apple-GPU backend (macOS only); mutually independent from CUDA. |
| `PYTHON` | `python3` | Interpreter for `make test-python`. |

Targets: `make CUDA=1` (engine), `make cuda-test CUDA=1` (kernel correctness —
on unified-memory hardware it runs twice, zero-copy and forced-discrete),
`make cuda-bench CUDA=1` (tensor-core microbenchmark).

## Runtime: enabling the backend

| Variable | Default | Meaning |
|----------|---------|---------|
| `COLI_CUDA` | `0` | `1` enables the CUDA backend. Requesting it on a CPU-only binary fails at startup (no silent fallback). |
| `COLI_GPU` | `0` | Single CUDA device ordinal to use. |
| `COLI_GPUS` | — | Comma list of device ordinals (`0,1,2,3`). Mutually exclusive with `COLI_GPU`. |
| `COLI_CUDA_UNIFIED` | auto | Unified-memory zero-copy mode. Auto-detected per device (GB10/Grace → on). `0` forces the discrete upload path even on unified hardware (needed for `COLI_CUDA_TC_INT4`); `1` cannot force it onto hardware without coherent pageable-memory access. |

## Runtime: what runs on the GPU

| Variable | Default | Meaning |
|----------|---------|---------|
| `CUDA_EXPERT_GB` | `0` | Total VRAM budget (GB, across all selected devices) for the persistent hot-expert tier, filled from the `PIN` ranking. Clamped against free VRAM after reserving the projected dense set and 2 GB headroom per device. **Ignored on unified memory** (the tier costs no VRAM; all experts are GPU-served anyway). |
| `CUDA_DENSE` | `0` | `1` also places resident dense/attention projection tensors on the GPU, round-robin across devices. On unified memory these are zero-copy wraps (no VRAM footprint). |
| `CUDA_RELEASE_HOST` | `1` if multi-GPU, else `0` | Free the host copy of experts once uploaded to VRAM. **Forced to 0 on unified memory** — the GPU reads the host copy in place, releasing it would be a use-after-free. |
| `COLI_CUDA_ATTN` | `0` | Experimental decode-time (S≤4) MLA weight-absorption attention on the GPU. Requires `CUDA_DENSE=1` (needs `kv_b` GPU-resident). |

## Runtime: expert placement (CPU tiers the GPU builds on)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SNAP` | required | Directory of the converted model. |
| `PIN` | — | Path to a routing-frequency stats file (from `STATS=…` run); its hottest experts are pinned in RAM (and become the VRAM ranking for `CUDA_EXPERT_GB`). |
| `PIN_GB` | `10` | RAM budget for `PIN`. |
| `PIN_FILL` | `0` (`1` if multi-GPU + `CUDA_RELEASE_HOST`) | After the measured hot set, also pin never-seen experts to fill the remaining budget. |
| `AUTOPIN` | `1` | Without `PIN`: auto-pin from the accumulated `<SNAP>/.coli_usage` history (needs ≥5000 selections). |
| `RAM_GB` | auto (MemAvailable) | Overall RAM budget that caps the LRU expert cache and the autopin quota. |
| `REPIN` | `0` | Live re-pin pass every n emitted tokens (swaps cold pins for hot unpinned experts; refreshes the same VRAM/zero-copy slot). |
| `STATS` | — | Write the expert-usage histogram to this file at exit (input for `PIN`). |
| `COLI_MMAP` | `0` | Experts as zero-copy mmap views of the model files instead of pread slabs (page cache = expert cache). |
| `DIRECT` | `0` | O_DIRECT expert reads. |
| `DROP` | unset | If set: fadvise-DONTNEED streamed expert pages right after the read (protects throughput under page-cache pressure). |

## Runtime: kernel selection / debugging

| Variable | Default | Meaning |
|----------|---------|---------|
| `COLI_CUDA_W4_PACKED` | `1` | Packed int4 W4A32 expert kernels (exact). `0` falls back to the generic per-nibble path. |
| `COLI_CUDA_DUAL_PROJ` | `1` | Fused gate+up projection kernel. |
| `COLI_CUDA_TC_INT4` | `0` | int4 tensor-core (WMMA) expert path with quantized activations (approximate; see README quality note). Requires signed-nibble device copies, so on unified hardware it needs `COLI_CUDA_UNIFIED=0`. |
| `COLI_CUDA_TC_MIN_ROWS` | `8` | Minimum rows per expert before the tensor-core path engages. |
| `COLI_CUDA_ASYNC` | `1` | Pinned-staging async H2D/D2H for expert groups (discrete path only; unified memory has no copies at all). |
| `COLI_CUDA_PROFILE` | `0` | Accumulate H2D / kernel / D2H timings for expert groups, printed with the end-of-run stats. |

## Unified memory (DGX Spark / GB10, Grace)

Detected automatically; the startup banner shows
`[CUDA] device 0: NVIDIA GB10, … unified memory (zero-copy)` and
`[CUDA] mode: ALL experts on GPU (unified-memory zero-copy)`.

What changes versus a discrete GPU:

- **No VRAM duplication.** `coli_cuda_tensor_upload` wraps the host pointer
  (mmap view, pin slab, or streamed slab) instead of `cudaMalloc`+`cudaMemcpy`.
  Host int4 stays in its offset-binary layout; the kernels decode it in place.
- **No staging copies.** Expert-group inputs/outputs are read/written directly
  in the caller's buffers; the pinned-staging pipeline and H2D/D2H disappear.
- **Every expert is GPU-served** — pinned, LRU-cached, and streamed — because
  the wrap costs one small struct. The discrete-GPU rule "streaming experts
  stay on the CPU to avoid the PCIe copy" does not apply.
- `CUDA_EXPERT_GB` and `CUDA_RELEASE_HOST` are ignored (warned at startup).

Quickstart on a DGX Spark:

```bash
cd c && make CUDA=1
make cuda-test CUDA=1                        # validates zero-copy AND discrete paths
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=/path/glm52_i4 ./glm 64 4 4
```

Caveats:

- GPU results are not byte-identical to the CPU kernels (README: kernel-family
  sensitivity, issue #100); `COLI_CUDA=0` for byte-exact CPU reproducibility.
- `COLI_CUDA_TC_INT4` needs `COLI_CUDA_UNIFIED=0` (and then behaves exactly
  like a discrete GPU, including VRAM copies).
- A/B the zero-copy dispatch against the CPU expert path with
  `COLI_CUDA_UNIFIED=0 CUDA_EXPERT_GB=0` (CUDA on, experts on CPU) — whether
  GPU-serving *streamed* experts wins depends on batch shape; measure on your
  workload.
