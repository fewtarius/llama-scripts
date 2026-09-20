# llama-scripts

Scripts to configure and run [llama.cpp](https://github.com/SyntheticAutonomicMind/llama.cpp)
on AMD APU hardware. Self-contained - no system ROCm install required. We use the
[Synthetic Autonomic Mind fork](https://github.com/SyntheticAutonomicMind/llama.cpp)
of llama.cpp, which carries AMD-specific optimizations:

- **Laguna S with DFlash** - fused attention and DFlash draft model support
- **Lightning Indexer** - fused op for compressed-KV attention (DeepSeek MLA)
- **APU tuning** - `nodes_per_submit` auto-lowering, subgroup size pinning,
  quantized-KV dequant-once before flash attention

Vulkan (RADV) is the default backend for best stability on RDNA3 iGPUs.

The project provides:

- GPU/CPU auto-detection via PCI ID and CPU ISA
- Build orchestration for Vulkan, ROCm, and Metal backends
- An optimistic-first solver that reads GGUF metadata and picks the best
  configuration for your hardware
- GTT/VRAM memory configuration via kernel parameters
- Model download with split-shard validation
- A benchmark harness with TTFT measurement

---

## Quick start

```bash
git clone --recurse-submodules https://github.com/fewtarius/llama-scripts.git
cd llama-scripts

# Build Vulkan backend (default on Linux AMD)
./scripts/rebuild.sh

# Drop a GGUF model in models/, then start the server:
./llama-run.sh --server
# -> http://localhost:9090
```

---

## Using llama-scripts

```bash
# List models found in models/
./llama-run.sh --list-models

# Start server (auto-detects model, defaults to --backend vulkan)
./llama-run.sh --server

# Specific model
./llama-run.sh --server gemma-4-26b

# Interactive chat mode
./llama-run.sh --interactive gemma-4-26b

# Download a model
./llama-run.sh --download Qwen3-14B --quant Q4_K_M

# List available backends
./llama-run.sh --list-backends

# Print the solver's chosen profile for a model
./llama-run.sh --print-profile Qwen3.6-35B-A3B

# Rebuild
./scripts/rebuild.sh              # Vulkan only (default)
./scripts/rebuild.sh --rocm       # ROCm (optional)
./scripts/rebuild.sh --both       # Vulkan + ROCm
./scripts/rebuild.sh --rebuild    # Full rebuild from scratch
```

### Defaults

The server launches with sensible defaults:

| Option | Default | Override |
|--------|---------|----------|
| Backend | `vulkan` | `--backend` / `LLAMA_BACKEND` |
| Mode | `--server` | `--interactive` for chat |
| Reasoning preservation | on | `--no-preserve-reasoning` |
| KV cache | q8_0 | `--kv-cache-type` |
| Checkpoint every-N | -1 (disabled) | `--checkpoint-every-n-tokens` |
| Context size | auto (solver) | `--ctx-size` |
| GPU layers | 99 | `-ngl` / `--gpu-layers` |
| Threads | auto-detected | `--threads` |
| Port | 9090 | `--port` / `LLAMA_PORT` |

### Reasoning model support

Reasoning models - DeepSeek-R1, Qwen3.6, GLM-4.7 - emit thinking blocks before
each response. By default the runner preserves these in prior assistant messages
so conversation state is maintained across turns. To strip them, pass
`--no-preserve-reasoning`. Use `--reasoning-budget N` to cap thinking tokens
per response (default: 8192).

### Solver and profile selection

`llama-run.sh` runs an **optimistic-first solver** (in `scripts/optimize.sh`).
The solver reads actual GGUF metadata and system memory/GPU budgets to pick
the best configuration - it is the single configuration path.

**Solver pipeline:**

1. **Read GGUF metadata** via `scripts/read_gguf_kv.py` - block count,
   `head_count_kv`, `key/value_length`, `full_attention_interval` (for hybrid
   SSM models), `nextn_predict_layers` (for MTP), `expert_count` (for MoE),
   training context. Reads up to 1 MB of the GGUF to catch architecture-specific
   keys like `laguna.expert_count` or `qwen3next.expert_count`.

2. **Start optimistically** with per-archetype (batch, ubatch) defaults
   derived from llama-bench sweeps across 17 model/hardware combinations:

   | Archetype | Size | ubatch | batch |
   |-----------|------|--------|-------|
   | Dense | any | 1024 | 4096 |
   | MoE small | <60 GB | 1024 | 2048 |
   | MoE large | 60-100 GB | 2048 | 8192 |
   | MoE huge | >=100 GB | 4096 | 8192 |
   | SSM / hybrid | any | 1024 | 4096 |
   | qwen4exp | any | 2048 | 4096 |
   | MLA (DeepSeek-V2/V3/V4, GLM-4.x) | any | 4096 | 8192 |

   Plus: f16/f16 KV, training context (or 4x capped), cache-ram on, draft model
   enabled if found, threads = physical cores (batch) / half (gen), MoE strategy
   = `gpu`.

3. **Score and search** - builds a priority list of all valid combinations:
   - **Strategy**: `gpu` (300), `residency` (200), `cpu` (250)
   - **Context**: 128K (100), 96K (95), 196K (90), 262K (85), 64K (80)
   - **KV type**: f16 (30), q8_0 (20), q4_0 (10)
   - **Draft**: enabled (5), disabled (0)
   - **Batch/ubatch**: optimistic default (6), partial match (4), other (0)

   Score = `strategy*1000 + ctx*10 + kv + draft + batchub`. First combo
   fitting both GPU and system budgets with minimum cache RAM wins. Memory
   pressure is the real decision driver; the scoring is just a tiebreaker.

   **Strategy selection rules:**
   - Non-MoE/dense/SSM models: only `gpu` is valid. `cpu` and `residency`
     are skipped (no experts to offload).
   - MoE model >80% of GPU budget and fits in system RAM -> `residency`
     preferred (30% GPU / 70% RAM via `--n-cpu-moe`).
   - Otherwise (MoE): `gpu` -> `residency` -> `cpu`.

4. **Cache RAM allocation** - from system memory leftover, not GPU leftover:
   total RAM - OS reserve - config system cost, capped at 25% of total RAM
   with 10% headroom. With n_parallel=1 the prompt cache is set to 0 - the
   in-memory checkpoint ring covers the same use case at zero cost.

5. **Detune safety net** - if no fit: ctx = 65536, KV = q4_0/q4_0. If still
   no fit, exits with error. Phase 2 fine-grained detunes: reduce KV (q8->q4),
   reduce NGL, drop draft, reduce ubatch - each step once.

The solver maps its output to a profile name for `--print-profile` so users
have a stable identifier regardless of which combination was picked. See
[SOLVER.md](SOLVER.md) for the full algorithm, benchmark data, and hardware
notes.

**Override precedence** (low -> high):

| Level | Source |
|-------|--------|
| 1 | Built-in defaults (solver start) |
| 2 | System detection (GPU budget) |
| 3 | Solver output |
| 4 | User overrides (env vars / CLI flags) |

**User overrides recognized:**

| Override | CLI Flag | Env Var |
|----------|----------|---------|
| Context size | `--ctx-size` | `USER_CTX_SIZE` |
| KV cache K/V | `--kv-cache-type` | `KV_CACHE_K_OVERRIDE` / `KV_CACHE_V_OVERRIDE` |
| Threads | `--threads` | `LLAMA_THREADS_OVERRIDE` |
| Ubatch | `--ubatch-size` | `OVERRIDE_UBATCH_SIZE` / `MOE_UBATCH_OVERRIDE` |
| Cache RAM | `--cache-ram` | `OVERRIDE_CACHE_RAM` |
| GPU layers | `-ngl` / `--gpu-layers` | `OVERRIDE_NGL` |
| Fit mode | `--fit on` | `OVERRIDE_FIT` |
| Reasoning budget | `--reasoning-budget` | `OVERRIDE_REASONING_BUDGET` |
| Checkpoint every-N-tokens | `--checkpoint-every-n-tokens` | `OVERRIDE_CHECKPOINT_EVERY` |
| MoE strategy | `--cpu-moe-strategy` | `OVERRIDE_MOE_STRATEGY` |
| Spec draft p-min (DFlash) | (env) | `LLAMA_SPEC_DRAFT_P_MIN_DFLASH` |
| Spec draft p-min (DSpark) | (env) | `LLAMA_SPEC_DRAFT_P_MIN_DSPARK` |
| Spec draft p-min (MTP) | (env) | `LLAMA_SPEC_DRAFT_P_MIN_MTP` |

`LLAMA_THREADS`, `LLAMA_KV_CACHE_TYPE_K` are **defaults** (level 2), not
overrides. Use the `*_OVERRIDE` variants to win over the solver.

When the solver reduces NGL (layers offloaded to GPU) to fit the GPU budget,
`llama-run.sh` caps the launched server's `-ngl` at the solver's value so the
reduction actually applies at launch time (not just in profile output).
`OVERRIDE_FIT` sets `SOLVER_NGL=-1` to let `llama-server` auto-fit, which skips
the cap.

### Hardware support

The solver uses a single archetype (derived from Strix Halo benchmark data) for
all AMD APUs. Hardware differences are handled through actual system capability
detection - GPU memory budget, BIOS VRAM carveout, CPU ISA - not tiered
branching. See [SOLVER.md](SOLVER.md) for the full algorithm and
[AGENTS.md](AGENTS.md) for hardware-specific notes.

**Supported GPU generations:** Cezanne, Phoenix, Hawk Point, Strix Point,
Strix Halo, Sephiroth, Rembrandt, Mendocino, Renoir, Lucienne.

---

## Backends

### Vulkan (Linux/AMD) - default

Uses the Mesa RADV driver. Best stability on RDNA3 iGPUs (Phoenix, Hawk Point,
Strix Point) and earlier GCN/RDNA generations. CPU offloading works for models
that don't fit in GPU memory.

llama.cpp's Vulkan backend carries performance work relevant to this fork:

- **DeepSeek-V4 Lightning Indexer** - fused op for compressed-KV attention
- **DSV4 hyper-connection fused ops** - replaces softmax-scale-iterate sequences
- **Quantized-KV FA dequant-once** - dequantizes K/V cache once into f16 scratch
  before flash attention (with host-RAM safety gate)
- **APU `nodes_per_submit` auto-lower** - defaults to 100 on UMA to stay under
  amdgpu `lockup_timeout`
- **Subgroup size pinning** - 32-wide on RDNA3 wave64 for coopmat1 FA

### ROCm (Linux/AMD) - optional

Has known issues on RDNA3 - GLM-4.7-Flash and DeepSeek2 MLA models produce zero
generation tokens. Use Vulkan unless you have a specific reason to try ROCm.
ROCm components carry AMD's license and are downloaded by `scripts/rebuild.sh`.

### Metal (macOS)

Apple Silicon and Intel Macs with Metal-capable GPUs. Build with
`./scripts/rebuild.sh` on macOS - it auto-detects the platform and builds the
Metal backend.

---

## Hardware setup

### GPU memory

AMD APUs share system RAM between CPU and GPU. On the Nimo Axis N161 (Strix
Halo, Ryzen AI Max+ 395), the BIOS currently carves out **512 MiB** of VRAM - the
OS sees ~125 GB RAM, and GTT defaults to ~62.5 GB. GPU-visible total is ~63 GB.
`detect-gpu.sh` recognizes Strix Halo via PCI ID (`1002:1586`) regardless of the
BIOS VRAM allocation.

Use `apply-ttm-kernel-params.sh` to configure GTT for your hardware:

```bash
# Auto-detected based on system RAM (default)
sudo ./scripts/apply-ttm-kernel-params.sh

# Explicit size
sudo ./scripts/apply-ttm-kernel-params.sh 104

# Remove all params and reset to defaults
sudo ./scripts/apply-ttm-kernel-params.sh --remove
```

The script auto-detects the actual GPU VRAM carveout and system RAM to decide
whether to cap `vis_vramlimit`. On systems with a large BIOS carveout, like
Strix Halo with 96 GB, it leaves VRAM uncapped. On smaller carveouts, it caps
`vis_vramlimit` and grows GTT to fill remaining system RAM.

It writes `amdgpu.gttsize`, `ttm.pages_limit`, and `ttm.page_pool_size` to your
bootloader config (GRUB or systemd-boot). SteamFork dual-ESP installs have a
redundant `/boot` - the script probes other vfat partitions and edits the real
loader entry in place. Re-run after OS updates.

Verify after reboot:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E "amdgpu|ttm"
```

### RADV APU memory split

RADV on APUs (`has_dedicated_vram=false`) reports only 2/3 of (VRAM + GTT) as
the DEVICE_LOCAL heap and 1/3 as host heap (game-compat heuristic in
`radv_physical_device.c`). `~/.drirc` enables
`radv_enable_unified_heap_on_apu` for `llama-server`/`llama-cli`/`llama-bench`
so DEVICE_LOCAL = full VRAM + GTT. Without it, models > 2/3 of GPU-visible memory
crash with `vk::DeviceLostError` at load.

### GPU and CPU detection

`scripts/detect-gpu.sh` identifies your AMD GPU via PCI device ID and sets:

| Variable | Description |
|----------|-------------|
| `LLAMA_APU_VRAM_GB` | VRAM carveout in GB |
| `LLAMA_TOTAL_RAM_GB` | Total system RAM in GB |
| `LLAMA_THREADS` | Optimal thread count |
| `LLAMA_GFX_ARCH` | e.g. `gfx1151` (Strix Halo), `gfx1103` (7840U) |
| `LLAMA_GPU_NAME` | e.g. `Radeon 8060S`, `Radeon 780M` |

The script also detects CPU ISA level and generates cmake flags so ISA extensions
(AVX-512 BF16 on Zen 4, AVX2 on Zen 3) are enabled at build time.

Supported generations: Cezanne, Phoenix, Hawk Point, Strix Point, Strix Halo,
Sephiroth, Rembrandt, Mendocino, Renoir, Lucienne.

Overrides:
```bash
LLAMA_GFX_VERSION_OVERRIDE=11.0.3  # skip GPU detection
```

---

## Features

### Auto-profiling

Models are detected by filename and assigned a profile automatically - no manual
configuration needed. The optimistic-first solver reads GGUF metadata and
computes the best configuration for your hardware. MoE models get checkpoint
strategies and reasoning format support. SSM/Mamba models get context-shift
disabled. Large dense models get optimized batch sizes.

The profile is logged at startup, e.g.:
```
Auto profile (solver): halo-moe-small (36GB, MoE=true, SSM=false, MTP=true, qwen4exp=false)
Solver chose: ctx=131072 KV=f16/f16 ubatch=1024 batch=2048 threads=32/16
```

### Hybrid MoE and SSM architectures

Qwen3.6 (both dense and MoE variants), Qwen3-Coder-Next, and Qwen3.5-122B mix
attention layers with recurrent (Mamba-2) layers. Laguna-S-2.1 is a DFlash
target with shared experts and sliding window. llama.cpp handles these
correctly: KV cache shifting, attention-only memory clearing that preserves
recurrent state, and checkpoint overflow prevention - so same-conversation state
is always accepted regardless of size.

### MoE expert tracking

MoE models activate only a subset of experts per token - typically 3-8 out of
128-256. llama.cpp tracks which experts activate through `GET /expert-stats`
and `POST /expert-tracking` endpoints. This is instrumentation for now; future
work will use it to reorder experts for cache locality.

### Multi-agent sessions (parallel slots)

`--parallel N` (or `-np N`) configures the server with N independent slots, each
holding its own KV cache. Two agentic sessions using the same server in
parallel no longer block each other - the second agent queues onto slot 1 while
the first is generating on slot 0. The host-memory prompt cache
(`--cache-ram`, `--cache-idle-slots`) becomes useful here because slots can
hold divergent state across each other and the LCP matcher hot-swaps them
into fresh tasks.

Per-slot KV cache memory is `n_ctx * layer_count * head_dim * 2 (K+V) *
dtype_bytes`. On a 128K-context f16 model with 60 layers, that's ~2.6 GiB per
slot. With `-np 4`, that 10 GiB comes out of the GPU budget
the solver would otherwise allocate to model and cache-ram. The solver
accounts for this automatically - if `-np 2` doesn't fit, it shrinks
`n_ctx` or `cache-ram` rather than failing the launch.

CLI: `--parallel 2` or `LLAMA_PARALLEL=2 ./llama-run.sh --server ...`
(`LLAMA_PARALLEL` only applied when `--parallel` is not given on the command line).
With a single slot (`-np 1`, the default), the solver disables the host-memory
prompt cache automatically - the cache cannot accumulate state when the only
slot is also the one being saved+loaded, and the in-memory checkpoint ring
covers the same use case at zero cost.

---

## Repo structure

```
llama-scripts/
├── llama-run.sh              # Main entry point - model detection, server launch
├── scripts/
│   ├── rebuild.sh            # Build script (Vulkan default, optional ROCm)
│   ├── env.sh                # Environment setup (source before ROCm tools)
│   ├── detect-gpu.sh         # GPU/APU and CPU ISA auto-detection
│   ├── optimize.sh           # Optimistic-first solver (sourced by llama-run.sh)
│   ├── benchmark.sh          # KV cache performance testing
│   ├── apply-ttm-kernel-params.sh  # GPU memory config (capability-based)
│   ├── install-deps.sh       # Dependency installer
│   ├── lib-discover-models.sh  # HuggingFace model discovery
│   ├── read_gguf_kv.py       # GGUF metadata reader
│   └── log_analyzer.py       # Benchmark log analysis
├── llama.cpp/               # Submodule - our fork of llama.cpp
├── src/                     # Build output dirs (gitignored)
│   ├── llama-vulkan/
│   ├── llama-rocm/
│   └── llama-metal/
├── deps/                    # ROCm SDK (downloaded by rebuild.sh, gitignored)
├── models/                  # GGUF files (gitignored)
├── kv-cache/                # Persistent KV cache (gitignored)
```

---

## License

**Source code:** [GPL-3.0-or-later](LICENSE)
**Documentation:** [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

llama.cpp is licensed under its own terms - see the [llama.cpp project](llama.cpp).
