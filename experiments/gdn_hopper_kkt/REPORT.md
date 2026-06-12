# Porting GDN `kkt_inv_uw` to Hopper (SM90): Report

**TL;DR.** We ported sglang's Blackwell-only GDN chunked-prefill `kkt_inv_uw` kernel to Hopper
(H200) in CuteDSL, and validated the central thesis: doing the chunk matrix-inverse on **tensor
cores** (warp-mma Newton–Schulz) beats FLA's CUDA-core forward-substitution — **3.36× at 32K**.
The kernel is fully correct vs FLA/torch. End-to-end `kkt_inv_uw` is **0.77× of FLA at 32K and
climbing with sequence length**; closing the remaining gap is gated on one sub-stage (U/W) whose
best optimization is blocked by a wgmma smem-descriptor limitation.

## Background
GDN (gated delta net) chunked **prefill** is a 3-kernel pipeline: `kkt_inv_uw` (intra-chunk) →
`h` (recurrent scan) → `o` (readout). Only `kkt_inv_uw` is ported here. It has 3 sub-stages:
`KKT→A` (β·K·Kᵀ·decay, strictly-lower) → `inverse` ((I+A)⁻¹) → `U/W` (Ai·(β·V), Ai·(β·eᵍ·K)).
sglang's CuteDSL version is SM100/Blackwell-only (`tcgen05`+`tmem`); Hopper fell back to FLA triton.

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
