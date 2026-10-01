# Solver Algorithm and Tuning

The optimistic-first solver in `scripts/optimize.sh` picks (ctx, KV cache type,
MoE strategy, batch, ubatch) for each (model, hardware) pair from a scored
combination space. User overrides applied last.

## Decision order

1. **Built-in defaults** (solver starts here - see "Optimistic defaults" below)
2. **System detection** (GPU memory via `detect-gpu.sh`)
3. **Solver output** (chosen from a scored combination matrix)
4. **User overrides** via env vars and CLI flags

## Optimistic defaults (per archetype)

`llama-bench` sweeps across n_batch in {2048, 4096, 8192} and n_ubatch in
{256, 512, 1024, 2048, 4096} on Ayaneo Flip (7840U) and Nimo Axis (Strix
Halo) for 17 model/hardware combinations informed the per-archetype defaults
below. Decode (tg) is memory-bandwidth bound and varies <2% across the
entire (batch, ubatch) range - prefill (pp) is what tuning moves.

The solver uses hardware-tier-aware defaults. `_detect_hardware_tier()`
classifies the device as `halo`, `standard`, or `handheld` based on the
detected GPU PCI ID, VRAM carveout, and system RAM. The key exception is
Strix Halo with a 512 MiB BIOS carveout - `_detect_strix_halo()` catches
this via the PCI ID (1002:1586/1660) and forces the tier to `halo`
regardless of VRAM size, so the halo defaults below apply.

### Defaults (Halo / Strix Halo archetype)

| Archetype | Size | ubatch | batch | Why |
|-----------|------|--------|-------|-----|
| Dense | any | 1024 | 4096 | 5-8% pp loss at ub=2048 (qwen35 27B Q8_0: 127 vs 121 pp) |
| MoE small/medium | <60 GB | 2048 | 4096 | Benchmarks on Strix Halo show 2048/4096 (ub/b) is optimal across small (512-token) and large (15k-token) prefill prompts. The old 1024/2048 default was 8.7% slower at pp15000 due to insufficient batch size for large prefill contexts. |
| MoE large | 60-100 GB | 2048 | 8192 | 8192 batch + 2048 ubatch is the empirical sweet spot (Laguna 118B, qwen35moe 122B, gpt-oss 120B) |
| MoE huge | >=100 GB | 4096 | 8192 | 4096 ubatch needed to amortize the heavy per-token work |
| SSM / hybrid | any | 1024 | 4096 | Linear-attention layers don't benefit from larger batches |
| qwen4exp (Qwen3.8-Flash-Next) | any | 2048 | 8192 | PLE + hybrid attention. 8192/2048 is 0.4-0.9% faster than 4096/2048 on Strix Halo for both small and large prefill prompts; KV cache is small due to hybrid attention (12 of 48 layers store KV). |
| MLA (DeepSeek-V2/V3/V4, GLM-4.x) | any | 4096 | 8192 | Latent attention has ~1/N KV cache vs non-MLA, so ubatch can carry more; Lightning Indexer fused op in llama.cpp accelerates the indexer/attention split when subgroup_size_control gates pass |

### Context size candidates

The solver reads the model's `context_length` from GGUF metadata (via
`read_gguf_kv.py`) and uses it directly as the starting `SOLVER_CTX_SIZE`.
Phase-1 candidate context sizes are generated dynamically via
`_opt_build_ctx_candidates(context_length)`: the standard sizes
`{1048576, 524288, 262144, 196608, 131072, 98304, 65536}` filtered to
those ≤ the model's max, capped at 1M (llama.cpp hard limit).

q4_0 KV cache is **not** a phase-1 candidate - it is reserved for phase 2
as an absolute last resort. Phase 1 enumerates ctx from largest to smallest
with KV types `{f16, q8_0}` in score order. The first combo that passes
both GPU and system memory checks (including the `min_cache_ram_mib`
leftover check) wins.

If no phase-1 combo fits, phase 2 starts from the most memory-conservative
fallback (`MIN_CTX` + `q4_0/q4_0` with CPU strategy for MoE models) and
applies detune steps in priority order (q8 → q4 → ubatch → ctx → draft → NGL).

## Combination scoring

For each (strategy, ctx, KV, draft, batch, ubatch) tuple, the solver computes
a score. Higher scores are tried first. The first combo that fits the GPU
budget AND system memory wins.

```
score = strategy_score * 1000      # gpu=300, cpu=250
      + ctx_score * 10             # 1048576=110, 524288=105, 262144=100, 196608=95,
                                   # 131072=90, 98304=85, 65536=80
      + kv_score                   # f16=30, q8_0=20
      + draft_score                # enabled=5
      + batchub_score              # opt=2, partial-match=1, other=0
```

The ctx_score is monotonically increasing with context size, so the
solver prefers the largest context that fits. The kv_score prefers f16
over q8_0. q4_0 is NOT in phase 1 - it is a phase-2 detune step only,
applied when no f16 or q8_0 configuration fits at any context size.

Phase 1 enumerates (ctx from model max down to 128k, kv in {f16, q8_0})
in score order. The (batch, ubatch) component is a small tiebreaker (max
+2) so it does not override the ctx/kv preference.

The (batch, ubatch) component is a small tiebreaker (max +2) so it does
not override the ctx/kv preference. The real decision driver is the GPU
memory check + system memory check.

## Detune steps (phase 2)

If no combination fits in phase 1, phase 2 starts from the most
memory-conservative fallback (MIN_CTX + q4_0/q4_0) and applies detune
steps in priority order. The preference is: **q4_0 KV cache only when no
choice, ctx below 128k only when no choice, reducing layers from GPU only
when no choice.**

The phase-2 detune order (first fit wins, each step applies at most once):

1. Reduce KV cache to q8_0 (only downgrades from f16 - skipped if already q8 or q4)
2. Reduce KV cache to q4_0 (only downgrades from f16/q8 - skipped if already q4)
3. Reduce ubatch by half (clamped at 512)
4. Reduce ctx (cascading values 1048576 -> 524288 -> 262144 -> 196608 -> 131072 -> 98304 -> 65536)
5. Drop speculative draft model
6. Reduce NGL by 10% per step (layers moved from GPU to CPU - absolute last resort)

## Edge cases and known issues

### Hybrid SSM/MoE (qwen3next)

`qwen3next` is detected as `is_ssm=true` in `_scan_gguf_arch` (the GGUF has
`qwen3next.ssm.*` keys plus `qwen3next.expert_count > 0`). The SSM branch
takes priority over MoE in the optimistic defaults, picking ubatch=1024
which matches the empirical peak (qwen3next 80B Q8_0: 762 pp at ub=1024 vs
537 at ub=4096).

### Models with misleading filenames

Some MoE models (Laguna-S-2.1, Qwen3-Coder-Next, GLM-4.7-Flash) don't match
the filename MoE regex `moe|a3b|a8b|flash|expert|gpt-oss`. The GGUF scanner
in `llama-run.sh::_scan_gguf_arch` reads the first 1 MB of the GGUF (up from
the original 16 KB) and checks for `expert_count` to set `is_moe=true`. This
catches all known MoE architectures.

### 8192 batch for very large MoE

The data shows 8192 batch is marginally faster (0.4-0.9% pp) than 4096
for very large MoE/qwen4exp models on Strix Halo (Qwen3.8-Flash-Next
qwen4exp: 602.6 pp at 8192/2048 vs 597.2 at 4096/2048). For 60-100 GB MoE,
the solver already defaults to 8192/2048. The phase-1 candidates list
always includes 8192 as a fallback so the memory check has it on the table.

### Decode (tg) is bandwidth-bound

Across all benchmark sweeps, tg varies <2% across the entire
(batch, ubatch) range. Tuning (batch, ubatch) moves pp but not tg. So
`vram-bandwidth-limited` decode speed is a hardware characteristic, not
something the solver can tune.

## Benchmark data (llama-bench)

| Hardware | Model | Size | Peak (b/u) | Peak pp (t/s) | Notes |
|----------|-------|-----:|-----------:|--------------:|-------|
| Halo | Qwen3.6-35B-A3B Q8_K_XL (qwen35moe) | 36 GB | 4096/2048 | 1520.0 | 8.7% better at pp15000 vs 2048/1024 halo default |
| Halo | Qwen3.8-Flash-Next Q4_K_XL (qwen4exp) | 104 GB | 8192/2048 | 602.6 | 0.9% better at pp15000 vs 4096/2048 |
| Halo | Laguna-S-2.1 Q5_K_XL (MoE) | 82 GB | 8192/2048 | 619.3 | 8192/4096 is 0.07% better at pp512 but 1.2% worse at pp15000 |
| Halo | gpt-oss-120b Q8_K_XL (MoE) | 60 GB | 8192/2048 | — | Profile confirms 8192/2048 |
| Halo | GLM-4.7-Flash Q8_K_XL (MLA) | 33 GB | 8192/4096 | — | MLA branch: 4096/8192 |
| Halo | DeepSeek-V4-Flash IQ3_XXS (MLA) | 101 GB | 8192/4096 | — | MLA branch: 4096/8192 |
| 7840U | Qwen3.6-35B-A3B Q8_0 (qwen35moe) | 36 GB | 4096/2048 | — | Standard tier default |

Benchmarked with llama-bench on Nimo Axis N161 (Strix Halo, gfx1151,
Radeon 8060S, 124 GB RAM, RADV unified heap enabled). Vulkan backend,
llama-bench with -ngl 99, --flash-attn auto, -t 16, pp512 and pp15000
with -n 128. The Qwen3.6-35B-A3B pp15000 sweep was decisive: 8192 batch
is 5% slower than 4096 (dispatch overhead on 40 CUs with 35B params),
while 2048 batch is 8.7% slower. The 4096/2048 default is optimal across
both prompt sizes.

The old benchmark table (from earlier sessions, different llama.cpp
builds and test harnesses) is superseded by the data above. Absolute pp
values differ because the old data used different model variants and
test parameters.

## Solver accuracy

After the refactor, the solver picks within 5% of the empirical
peak for all tested models on Strix Halo (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next,
Laguna Q5_K_XL). The solver's (batch, ubatch) defaults now match the benchmark
peaks for all three primary models.

To override the solver's choice: use `--ubatch-size N` and `--batch-size N`
on the `llama-run.sh` command line. The overrides win.
