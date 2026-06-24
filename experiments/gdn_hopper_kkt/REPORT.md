# Porting GDN `kkt_inv_uw` to Hopper (SM90): Report

**TL;DR.** We ported sglang's Blackwell-only GDN chunked-prefill `kkt_inv_uw` kernel to Hopper
(H200) in CuteDSL, and validated the central thesis: doing the chunk matrix-inverse on **tensor
cores** (warp-mma Newton–Schulz) beats FLA's CUDA-core forward-substitution — **3.36× at 32K**.
The kernel is fully correct vs FLA/torch. End-to-end `kkt_inv_uw` is **0.77× of FLA at 32K and
climbing with sequence length**; closing the remaining gap is gated on one sub-stage (U/W) whose
best optimization is blocked by a wgmma smem-descriptor limitation.

## Background
GDN (gated delta net) chunked **prefill** is a 3-stage logical pipeline: `kkt_inv_uw` (intra-chunk) →
`h` (recurrent scan) → `o` (readout). Only `kkt_inv_uw` is ported here. It has 3 logical sub-stages:
`KKT→A` (β·K·Kᵀ·decay, strictly-lower) → `inverse` ((I+A)⁻¹) → `U/W` (Ai·(β·V), Ai·(β·eᵍ·K)).
sglang's CuteDSL version is SM100/Blackwell-only (`tcgen05`+`tmem`); Hopper fell back to FLA triton.

**Fusion boundary clarification.** FLA is not a fully unfused 3-kernel baseline for `kkt_inv_uw`: SGLang's
FLA intra path fuses `KKT→A + solve_tril` into `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, then launches
`recompute_w_u_fwd` for U/W (2 kernels total for intra). FlashInfer's `chunk_gated_delta_rule` is a single
fused GDN prefill kernel for the recurrence itself, but SGLang still performs surrounding plumbing outside
FlashInfer: Q/K L2 norm, `exp(g)`, beta conversion/slicing, state-pool gather/scatter, plus separate QKV
split/gating/conv helpers.

## Approach
Rebuilt each sub-stage in CuteDSL on Hopper-native instructions (wgmma for the 64-wide GEMMs;
warp-level `mma.sync.m16n8k16` + ldmatrix for the 16×16 inverse blocks), TDD against torch + FLA.
The inverse uses the Blackwell **blocked-16×16** strategy: four 16×16 diagonal blocks inverted in
parallel by 4 warps via Newton–Schulz (3 rounds each), then block back-substitution.

## Correctness
All stages and the full fused kernel match FLA/torch at cosine ≥ 0.999 (multi-chunk, multi-head,
GVA), bf16, L2-normalized K.

## Performance (H200, Qwen3.5-35B GDN shapes; ratio = FLA/ours, >1 = we win)
| stage | 2K | 8K | 32K |
|---|---|---|---|
| KKT→A | 0.36× | 0.46× | 0.66× |
| **inverse** | 0.58× | **1.53×** | **3.36×** |
| U/W | 0.28× | 0.44× | 0.60× |
| **end-to-end** | 0.33× | 0.54× | **0.77×** |

The inverse — the algorithmic crux and the source of the Blackwell speedup — is a decisive win at
realistic prefill lengths and scales beautifully (massively parallel, barely grows with T). KKT and
U/W are FLA's well-tuned GEMM territory; both improve with T.

## Key findings
- **Tensor-core matrix inversion beats CUDA-core forward-substitution on Hopper** (3.36×@32K). Thesis proven.
- The blocked-16×16 warp-mma NS inverse is correct and fast; it replaces FLA's `solve_tril`.
- U/W optimization: occupancy/direct-gmem-write helped (0.45→0.60×); the bigger win (feed V/K
  directly via MN-major B, eliminating the transpose-build) is **blocked** by a wgmma
  `make_gmma_smem_desc` legalization limit; load-time transpose was worse (uncoalesced).
- End-to-end parity is gated entirely on U/W's transpose-build (Amdahl: no single stage dominates).

## Reproduction
Branch `jiaguo/gdn-hopper-kkt-experiment` on `jasperjiaguo/sglang`, dir `experiments/gdn_hopper_kkt/`.
Key files: `gdn_torch_ref.py` (oracle), `gdn_bench_inv.py` (inverse vs solve_tril — the win),
`gdn_bench_uw3.py` (best U/W), `gdn_bench_stages.py` (KKT), `gdn_bench_e2e.py` (end-to-end sum),
`gdn_blocked_inv.py` (full blocked inverse), `FINDINGS.md` (chronological log). Run on an H200 with
torch 2.11 cu130 + cutlass-dsl 4.5.2, `PYTHONPATH=<sglang>/python`.

## Follow-on
1. Crack the wgmma MN-major B layout → eliminate U/W's transpose-build → likely end-to-end parity at long T.
2. Port `h` (the largest prefill kernel) and `o` for a full-prefill speedup.
3. Fuse all three `kkt_inv_uw` sub-stages into one launch (avoids inter-stage gmem traffic).

---

## Update — Session 2 (2026-06-15): MN-major wall CRACKED + fusion findings

**Follow-on #1 (crack the wgmma MN-major B layout) is done.** The fix: B operand shape `(N,K)`;
`atom = warpgroup.make_smem_layout_atom(get_smem_layout_atom(LayoutEnum.COL_MAJOR, dtype, N), dtype)`;
`cute.tile_to_shape(atom, (N,K), order=(1,0))`; `b_leading_mode = OperandMajorMode.MN`; load
`sV[n,k] = gV[k,n]` with n inner (coalesced). This feeds V/K **directly** into the wgmma with no
transpose-build — the same thing Blackwell does via `transpose_B` and FLA via `tl.dot`.
Files: `gdn_mnmajor_probe.py` (probe), `gdn_bench_uw6.py` (U/W), `gdn_bench_uw7.py` (bf16-out variant).

**Result:** U/W **0.60× → 0.67×** @32K (0.738 → 0.657 ms), cosine 1.0. End-to-end **0.77× → 0.84×** @32K
(0.33 / 0.54 / 0.84 at 2K / 8K / 32K).

**But the win was small, which was informative:**
- bf16 output (`gdn_bench_uw7.py`) made no difference → U/W is **not** bandwidth-bound.
- Removing the transpose-build only bought ~11% → U/W is **occupancy/latency-bound**: 16,384 tiny
  one-(chunk,head)-per-CTA launches, each a serial barrier chain; actual wgmma compute is only ~34µs total.

**Naive fusion is the *wrong* fusion.** `gdn_fused_invuw.py` fuses inverse+U/W (Ai stays in smem, kills
the Ai gmem round-trip) — but it **regressed to 0.59×**: smem balloons to ~80KB (→ ~2 CTAs/SM occupancy)
and the serial chain lengthens, outweighing the saved round-trip. The vLLM PR's 3× comes from fusion
**+ async multi-stage pipelining** (warp-specialized producer/consumer overlapping the next chunk's loads
with the current chunk's compute), *not* sequential fusion.

### Updated verdict
- Best end-to-end is now **0.84× @32K** (was 0.77×), climbing with T.
- U/W (now 0.657 ms = ~62% of my total) remains the gate; it is occupancy-bound, not transpose-bound.
- **Path to >1× / 1.5×:** async `cp.async` double-buffered + warp-specialized pipeline (the Blackwell
  approach). The separate non-pipelined per-stage kernels are near their practical ceiling.

### Revised follow-on
1. ~~Crack the wgmma MN-major B layout~~ ✅ done (U/W transpose-build eliminated; U/W 0.60→0.67×).
2. **Async multi-stage pipeline** (cp.async double-buffer + warp specialization) — the real lever for >1×.
3. Port `h` (the largest prefill kernel) and `o` for a full-prefill speedup.

---

## Update — Session 3 (2026-06-21): FlashInfer SM90 GDN in-place tuning

We also tuned the installed FlashInfer SM90 GDN kernel used by SGLang. Newton-Schulz ports were correct but
slower in FlashInfer's in-kernel context:

| variant | 1K | 2K | 8K |
|---|---:|---:|---:|
| FlashInfer baseline | 0.0541 ms | 0.1033 ms | 0.4021 ms |
| vLLM-style NS16 port | 0.0749 ms | 0.1490 ms | 0.5929 ms |
| direct-store beta-path patch | **0.0446 ms** | **0.0846 ms** | **0.3304 ms** |

The win is not NS. For beta-enabled kernels, the full inverse smem round-trip is bypassed: we quantize the
KKT accumulator to `InverseType`, apply the needed 8x8 diagonal/upper correction in the final register
transform, beta-scale, then store once. Stress checks were bitwise-identical to the restored baseline for the
tested beta/scale grid. A follow-up Hopper micro-tune changed the 8x8-block predicate from division to an
unsigned xor mask; this was bitwise-identical and slightly improved the 2K median. Patch:
`flashinfer_gdn_direct_store_transform.patch`.

---

## Update — Session 4 (2026-06-24): FlashInfer vs FLA fusion boundary + plumbing

A fair comparison is **not** "FlashInfer one kernel vs completely unfused FLA." The boundaries are:

- **FlashInfer GDN prefill:** one fused FI custom op (`flashinfer::gdn_prefill` /
  `FlatKernelTmaWarpSpecializedDeltaRule`) that consumes already-prepared `q/k/v/g/beta/initial_state` and
  writes output/final state.
- **FLA intra:** two Triton kernels for `kkt_inv_uw`: fused `KKT→A + solve_tril`, then `recompute_w_u_fwd`.
- **Shared SGLang plumbing outside the recurrence:** causal conv, QKV split/projection handling,
  fused GDN gating (`fused_gdn_gating`), Q/K normalization, `exp(g)`/beta conversion, and state-pool
  gather/scatter are not part of the FI prefill kernel. SGLang has its own helper fusions such as
  `fused_qkv_split_gdn_prefill`, but those are separate from both FI and FLA recurrence kernels.

With the optimized FI direct-store beta-path patch active, direct prefill microbenchmarks on H200
(`H=16,D=128,beta=0.01`, finite stable inputs) showed:

| T | baseline FI | optimized FI | optimized FLA | optimized FI speedup vs baseline | FLA / optimized FI |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.0567 ms | 0.0483 ms | 0.2518 ms | 1.17× | 5.21× |
| 2K | 0.1063 ms | 0.0891 ms | 0.2440 ms | 1.19× | 2.74× |
| 8K | 0.4072 ms | 0.3371 ms | 0.4782 ms | 1.21× | 1.42× |
| 32K | 1.6147 ms | 1.3309 ms | 1.8146 ms | 1.21× | 1.36× |

A follow-up predicate micro-tune inlined the 8x8-block test to avoid extra live locals:
`(uint32_t(s) ^ uint32_t(t)) < 8u && uint32_t(s) <= uint32_t(t)`. It was neutral/slightly faster
than the named-local xor version (8K: 0.3353 ms vs 0.3368 ms; 32K: 1.3247 ms vs 1.3321 ms) and is the
current patch form.

Nsight tooling note: `ncu` is available under `/usr/local/cuda/bin/ncu` and `/opt/nvidia/nsight-compute/2025.3.1/ncu`,
but NCU collection on this JIT-loaded FI custom op currently hangs or fails (`malloc(): unsorted double linked list corrupted`).
`nsys` does capture the expected FI kernel name, but reliable NCU occupancy / SpeedOfLight numbers are still pending.

