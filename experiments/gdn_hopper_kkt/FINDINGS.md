# GDN kkt_inv_uw Hopper port — findings log

## P0: FLA per-stage baseline (H200, Qwen3.5-35B-A3B GDN shapes, random inputs)
Shapes: k_heads=16, v_heads=32, head_k=head_v=128, BT=64, B=1. WARMUP=10 ITERS=50.
Stages = FLA `chunk_gated_delta_rule_fwd` decomposition; `intra` = the kkt_inv_uw counterpart
("fused kkt + solve_tril + recompute_w_u").

| T     | intra (kkt) ms | h ms   | o ms   | intra % of (intra+h+o) |
|-------|----------------|--------|--------|------------------------|
| 2048  | 0.0589         | 0.0694 | 0.0395 | 35.1%                  |
| 8192  | 0.1898         | 0.2609 | 0.1335 | 32.5%                  |
| 32768 | 0.7525         | 1.0203 | 0.5392 | 32.5%                  |

**Amdahl reality:** kkt_inv_uw is ~1/3 of the 3-kernel prefill time. `h` is the LARGEST
single stage (~44%). A kkt-only port, even taken to zero, caps the prefill-kernel speedup
at ~1.5x. End-to-end win REQUIRES porting h (and o) too. kkt-first is still right: it is
the algorithmic crux (TC Newton-Schulz inverse vs FLA forward-substitution), the hardest
tmem->smem remap (so the biggest de-risk), and standalone-validatable.

## P1a: torch reference oracle — VALIDATED
`gdn_torch_ref.py::ref_kkt_inv_uw(k,v,beta,g_cs,n_iter)`. Exact-inverse path vs FLA intra:
- U cosine = 0.999987, W cosine = 0.999996  (threshold 0.999) -> ORACLE TRUSTWORTHY.
- Pinned math: A[i,j]=beta_i·<k_i,k_j>·exp(g_i−g_j) for i>j; Ai=(I+A)^-1; U=Ai@(beta·V);
  W=Ai@(beta·exp(g)·K). g = chunk_local_cumsum output.

**Newton-Schulz iteration count (CRITICAL kernel param):** X0=I−A, X←X(2I−MX).
- n_iter ≤ 4: cosine ≈ 0 (garbage — order 2^(n+1) < 64, misses high Neumann terms).
- **n_iter = 5: cosine 0.999995 (exact).** Order 2^6=64 ≥ BT=64. So Hopper kernel needs ≥5 NS iters.
  (Verify the Blackwell kernel's hardcoded count matches; expect 5.)

**Conditioning note:** test used raw randn K (unnormalized) -> a few near-singular chunks gave
huge maxabs on isolated entries (cosine still 0.99998). Real GDN L2-normalizes K
(`use_qk_l2norm_in_kernel`), so <k_i,k_j>∈[−1,1], far better conditioned. Use L2-normalized K
in kernel-correctness tests to match the real distribution.

## TDD loop for P1b (each milestone: write ref-assert test → run (fail) → impl → pass → commit)
- 2a plumbing: TMA-load K tile, store back → assert identity passthrough.
- 2b K·Kᵀ→A (+ decay·beta·tril) → assert cosine(A_kernel, A_ref) ≥ 0.999.
- 2c Newton-Schulz inverse (THE crux): four 16×16 diagonal blocks, 3 NS rounds each + block back-sub (NOT monolithic 64×64) → assert cosine(Ai_kernel, Ai_ref) ≥ 0.999.
- 2d U/W via wgmma with Ai → assert cosine(U,W vs ref) ≥ 0.999.
- 2e full multi-chunk/multi-head → assert vs FLA intra end-to-end (cosine ≥ 0.999).
Oracle = gdn_torch_ref (n_iter=None for FLA cross-check; n_iter=5 for kernel match).

## P1b reshaping finding (2026-06-11) — Hopper material + Blackwell inverse strategy
**Hopper kernels DO exist (3 levels):** (1) CuteDSL SM90 atoms in `cutlass.cute.nvgpu`:
`warpgroup`(wgmma), `warp`(m16n8k16 mma.sync + LdMatrix/StMatrix), `cpasync`(TMA), plus
`cutlass.pipeline.sm90`, `cutlass.utils.hopper_helpers`. (2) Working SM90 CuteDSL GDN **decode**
kernel `sglang/jit_kernel/cutedsl_gdn.py` (smem+cp.async) = style template. (3) FLA triton
`_blockdim64` = algorithm already on Hopper (the baseline we beat).

**Blackwell kkt_inv_uw inverse is BLOCKED, not monolithic:** split I+A into four 16x16 diagonal
blocks; invert each via **3 Newton-Schulz rounds** (Ai<-2Ai-Ai@M@Ai, init Ai=I-A; 2^4=16>=16 exact
per 16x16 block); then block back-substitute off-diagonals. SAME structure as FLA solve_tril
(solve_tril_16x16 -> merge_16x16_to_64x64). [My earlier global-64x64 "need 5 NS iters" finding was
for a monolithic inverse; the real kernel uses 3 rounds PER 16x16 block + block back-sub.]

**KEY: Phase 3 (inverse = the speedup source) uses warp-level `mma.sync` m16n8k16 + ldmatrix/
stmatrix (`warp.LdMatrix8x8x16bOp`/`StMatrix8x8x16bOp`), NOT tcgen05/tmem -> already SM90-native.**
tcgen05/tmem is confined to Phase 2 (KKT K@K^T->A, `_tcgen05.mma_f16`) + U/W (`_tcgen05.mma_ts_f16`)
+ the tmem mbarrier/2-SM pipeline. So the port: keep inverse ~verbatim; re-plumb KKT+U/W MMAs
tcgen05/tmem -> wgmma/smem; rebuild pipeline with cutlass.pipeline.sm90. De-risks the crux.

**Sharpened go/no-go:** FLA solve_tril on Hopper already does the SAME blocked-16x16 strategy but
with CUDA-core forward-sub inside each block. P1c question = "does warp-mma Newton-Schulz on 16x16
blocks beat CUDA-core forward-sub on Hopper?" — testable in isolation.

## Loop progress (2026-06-11) — autonomous TDD
- **Goal reconfirmed:** port the GDN kkt_inv_uw ALGORITHM (tensor-core: KKT MMA + warp-mma Newton-Schulz
  inverse + U/W MMA) to Hopper SM90, to test if the TC-inverse approach beats FLA's CUDA-core
  forward-sub on H200. Naive per-element stages (2b) are THROWAWAY correctness scaffolds to lock
  math/layout; they are NEVER benchmarked. The benchmarked kernel is all-tensor-core. 2c inverse
  uses warp-mma FROM THE START (it IS the hypothesis under test).
- **2a GREEN** (K-tile identity through smem, exact). **2b GREEN** (A=strictLower(beta·KKt·decay),
  cosine 1.0 — naive dot, math/layout validated). Harness: gdn_hopper_kkt.py (kernel) +
  gdn_test_kkt.py (MILESTONE=2x runner, asserts vs gdn_torch_ref/ref_A).
- **2c design (THE crux):** lift the Blackwell inverse warp code ~verbatim (it's SM90-native:
  `mma_bf16`=mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32 in cute_utils; ldmatrix/stmatrix =
  warp.LdMatrix8x8x16bOp/StMatrix8x8x16bOp). Feed A via smem in the ldmatrix layout
  (sA_ldsm = logical_divide(sA,(16,(8,2)))) instead of tmem KKT MMA. Sub-milestones:
  2c-i single 16×16 block NS inverse (3 rounds, init Ai=I−A) vs torch block-inverse;
  2c-ii four diagonal blocks; 2c-iii block back-sub (off-diag by 1,2,3) -> full 64×64 Ai vs ref.
  Ref: gdn_torch_ref Ai (exact). Then 2d U/W (wgmma), 2e full vs FLA, P1c bench.
- **Fast directional signal option:** after 2c-iii, micro-compare the isolated inverse vs FLA
  solve_tril before building 2d/2e — early NO-GO read (caveat: isolated understates bubble-removal).

## 2c-0 WGMMA GEMM WORKING (2026-06-11) — tensor cores driven in CuteDSL
`gdn_wgmma_probe.py`: 64×64 bf16 C=A@Bᵀ via single-warpgroup wgmma -> cosine 1.000000,
maxabs 3.8e-6 vs torch. **The crux's hardest enabler is done — Hopper TC MMA works.**
Reusable idiom (the swizzle gotcha + fix):
  - tiled_mma = cutlass.utils.hopper_helpers.make_trivial_tiled_mma(BF16,BF16,
      OperandMajorMode.K, OperandMajorMode.K, Float32, (1,1,1), (M,N), OperandSource.SMEM)
    (OperandMajorMode from cutlass.cute.nvgpu; OperandSource from cutlass.cute.nvgpu.warpgroup)
  - smem layout: quack.sm90_utils.make_smem_layout(BF16, LayoutEnum.ROW_MAJOR, (M,K))  -> COMPOSED
  - **GOTCHA:** allocate with the AFFINE part + swizzle on the ptr, else MMA errors
    "Expected affine layout ... got composed":
      sA = smem.allocate_tensor(BF16, layout.outer, byte_alignment=1024, swizzle=layout.inner)
  - construct tiled_mma + smem layouts INSIDE @cute.kernel (composed layout can't be a JIT arg).
  - thr_mma=tiled_mma.get_slice(tidx); acc,tCrA,tCrB=sm90_utils.partition_fragment_ABC(thr_mma,(M,N,K),sA,sB)
  - sm90_utils.gemm(tiled_mma, acc, tCrA, tCrB, zero_init=True)  # fence/k-loop/commit/wait
  - epilogue: cute.autovec_copy(acc, thr_mma.partition_C(gC))
  - cute.gemm convention: acc[M,N] = A[M,K] · B[N,K] (K-contracted) = A @ Bᵀ.
**Inverse plan (2c):** global 64×64 NS, X←X(2I−MX), 5 iters (P1a: 5 exact for BT=64), each = 2
wgmma (Y=M@X; X=2X−X@Y) with smem round-trip. Simpler/correct-first than blocked-16×16 (a honing
option). Still a genuine tensor-core inverse -> answers the go/no-go.

## 2c GREEN (2026-06-11) — TENSOR-CORE INVERSE CORRECT ON HOPPER (the crux de-risked)
`gdn_test_ns_inverse.py`: NS inverse X←X(2I−MX)=2X−X@M@X, every matmul on wgmma tensor cores
(wgmma_gemm primitive). Ai=(I+A)^-1 vs torch exact: **cosine 1.000000**, max|(I+A)Ai−I|~3e-4 (bf16).
**Central risk retired: a tensor-core matrix inverse works correctly on Hopper.**
- **n_iter finding (realistic):** with L2-normalized K (well-conditioned, real GDN), **3 NS iters
  already exact** (cosine 1.0, residual bf16-limited ~4e-4) — matches Blackwell's hardcoded 3 rounds.
  The P1a "5 iters" was the ill-conditioned raw-randn worst case. Use 3 for real distributions.
- This 2c is Python-driven (one wgmma launch per matmul) = CORRECTNESS proof, NOT perf. The GO/NO-GO
  bench needs the FUSED single-kernel version (keep A/M/X in smem, no gmem round-trips) — that's the
  honing step. Remaining correctness: 2d U/W (Ai@(beta·V), Ai@(beta·exp(g)·K) — generalize wgmma to
  N=128), 2e full (in-kernel wgmma KKT + multi-chunk/head) vs FLA intra.

## 2d GREEN (2026-06-11, cron fire) — U/W on tensor cores
`gdn_test_uw.py`: U=Ai@(beta·V), W=Ai@(beta·exp(g)·K) via wgmma (N=128 split into 2×64-wide
halves through the 64×64×64 primitive). Full chain (wgmma NS inverse -> wgmma U/W) vs torch
oracle: U cosine 0.999999, W cosine 0.999999. **The complete kkt_inv_uw output (Ai,U,W) is now
correct on Hopper tensor cores.** Remaining correctness: 2e = replace naive-dot A (2b) with
in-kernel wgmma KKT (K@Kᵀ→A) + multi-chunk/multi-head + cross-check vs FLA chunk_gated_delta_rule_fwd_intra.
Then P1c = FUSE all stages into one kernel (A/M/X/U/W resident in smem, no gmem round-trips) and
microbench vs FLA intra @ {2K,8K,32K} -> GO/NO-GO.

## 2e GREEN (2026-06-11, cron fire) — FULL PIPELINE vs FLA, ALL P1b CORRECTNESS DONE
`gdn_test_full.py`: wgmma KKT (K@Kᵀ split-K: contraction 128 = 2×64 accumulate) + NS inverse +
U/W, looped over chunks × heads with GVA (k-head = h//(H//Hg)), B=1,T=128(2 chunks),Hg=4,H=8.
vs FLA chunk_gated_delta_rule_fwd_intra: **U cosine 0.999999, W cosine 0.999999.**
(Fix this fire: cast KKT inputs to bf16 — wgmma smem is bf16.)
**The entire kkt_inv_uw algorithm is correct on Hopper tensor cores, multi-chunk/multi-head.**
Milestones 2a,2b,2c,2d,2e all GREEN. P1b correctness COMPLETE.

### NEXT: P1c — FUSE + benchmark → GO/NO-GO (the remaining real work)
The current pipeline is Python-driven: MANY wgmma kernel LAUNCHES per (chunk,head) + gmem
round-trips → NOT representative of perf (would lose to FLA on launch overhead alone). For a valid
GO/NO-GO, build ONE fused @cute.kernel: grid over (chunk,head); per CTA keep K/V/beta/g in smem,
do KKT→A→NS-inverse→U/W all in-kernel (A/M/X resident in smem/regs, no gmem between stages),
write U/W out. Then microbench fused-kernel vs FLA intra @ {2K,8K,32K}. >1.0× = GO else NO-GO.
This is the hardest perf-engineering step (fused multi-stage wgmma + the smem-resident NS loop with
the stmatrix/ldmatrix transpose round-trips). Optional: blocked-16×16 warp-mma inverse (Blackwell
style) if global-64×64 NS is the bottleneck. n_iter=3 enough for realistic conditioning.

## P1c-i GREEN (2026-06-11) — fused KKT->A in one smem-resident kernel
`gdn_fused_kkt.py`: single @cute.kernel, K@Kᵀ as ONE wgmma (M=N=64, K=128 — tiled_mma k-loop
handles the 128 contraction, no manual split needed), acc written to smem via partition_C+autovec_copy,
then decay·beta·strict-tril applied in-place by threads (i>j), stored to A. vs ref_A: cosine 0.999996.
Establishes the fused pattern: wgmma -> write acc to smem -> elementwise modulation -> next stage.

### P1c remaining (the hard part): fused in-smem NS INVERSE
The crux of the perf work. NS X<-X(2I-MX) needs operands in smem; Y=M@X requires Xᵀ (gemm computes
A@Bᵀ). Transpose-in-smem each iter is the obstacle. Two paths:
  (a) Blackwell low-level: warp-mma m16n8k16 + ldmatrix-TRANSPOSE (ldsm_trans_atom) loads operands
      transposed from smem on the fly — no transpose buffer; blocked 16×16. Intricate fragment layouts.
  (b) wgmma + explicit smem transpose helper between the two matmuls per round.
FAIR-BENCHMARK NOTE: FLA intra is ALSO 3 triton kernels (kkt, solve_tril, recompute_wu), NOT one fused
kernel. So a fair go/no-go = my stages as gridded kernels (grid over all chunks×heads) vs FLA's 3 —
the in-smem NS inverse is still required, but full KKT+inv+UW single-kernel fusion is NOT mandatory for
the comparison. Plan: gridded fused-KKT (extend P1c-i grid to chunks×heads) + gridded inverse + gridded
U/W, bench vs FLA intra @ {2K,8K,32K}.

## P1c-ii GREEN (2026-06-11, FIRST TRY) — fused in-smem NS inverse (the hard part, done)
`gdn_fused_inv.py`: Ai=(I+A)^-1 via X<-2X-X@M@X, fully smem-resident, ONE kernel. cosine 1.000000,
max|(I+A)Ai-I|=2.95e-4. Per round: logical smem transpose (sXt[j,i]=sX[i,j]) → wgmma Y=gemm(M,Xt)
→ acc f32 round-tripped to smem + cast to bf16 → transpose → wgmma Z=gemm(X,Yt) → X=2X-Z. 5 iters.
Transpose-via-smem (path b) worked first try; no need for the Blackwell ldmatrix-transpose path.
**All fused building blocks now exist: P1c-i (KKT→A) + P1c-ii (inverse); U/W = same wgmma pattern.**

### NEXT: P1c-iii assemble full fused kkt_inv_uw kernel (gridded over chunks×heads), then
P1c-iv microbench vs FLA intra @ {2K,8K,32K} → GO/NO-GO. Building blocks: gdn_fused_kkt._fused_kkt
(KKT→A), gdn_fused_inv._fused_inv (inverse), + U/W (Ai@(beta·V), Ai@(beta·exp(g)·K) via wgmma,
N=128 as 2×64). Combine into one @cute.kernel with grid=(num_chunks*num_heads); each CTA loads its
K/V/beta/g slice, runs KKT→A→inverse→U/W in smem, writes U/W. Bench wall-time vs FLA's 3-kernel intra.

## P1c-iii GREEN (2026-06-11) — FULL FUSED kkt_inv_uw kernel works (one launch)
`gdn_fused_full.py`: KKT→A→NS-inverse→U/W ALL in ONE @cute.kernel, smem-resident, single (chunk,head).
U cosine 0.999997, W cosine 0.999997 vs torch ref. The entire kkt_inv_uw algorithm is now a single
fused CuteDSL kernel on Hopper tensor cores. (Fix this fire: CuteDSL dynamic-if doesn't export
branch-local vars — initialize `a` BEFORE the `if i<=j: a=0` mask. The m/x pre-init pattern from
P1c-ii was already correct.)

### NEXT (final): P1c-iv — grid + benchmark → GO/NO-GO
Extend `_fused_full` grid from (1,1,1) to (num_chunks*num_heads): bid=block_idx; chunk=bid//H,
head=bid%H; index K[chunk*BT:.., head//rep, :], V/beta/g[.., head, :] (GVA), write U/W[.., head, :].
Then timing harness (cuda events, warmup=10 iters=50 like gdn_fla_baseline.py): time the fused kernel
vs FLA chunk_gated_delta_rule_fwd_intra at seqlens {2048,8192,32768}, 35B GDN shapes (Hg=16,H=32,d=128).
>1.0x across the range = GO; else documented NO-GO. Use NITER=3 (realistic conditioning) in the perf kernel.

## P1c-iv VERDICT (2026-06-11): NO-GO at first cut — fused kernel ~3x SLOWER than FLA
`gdn_bench.py`, gridded fused kkt_inv_uw vs FLA chunk_gated_delta_rule_fwd_intra, 35B shapes (Hg=16,H=32,d=128):
  T=2048 : fused 0.1844ms  FLA 0.0565ms  -> 0.31x
  T=8192 : fused 0.5525ms  FLA 0.1901ms  -> 0.34x
  T=32768: fused 1.9668ms  FLA 0.7531ms  -> 0.38x
Correctness (gridded, T=128): U cosine 0.999999, W cosine 1.000000 (the kernel is CORRECT, just slow).
**VERDICT: NO-GO (first cut).** The numerical port fully succeeded; perf does not beat FLA.

### Why slow (honing levers, in impact order):
1. **Occupancy:** ~104KB smem/CTA (9 64×64/64×128 buffers) → only ~2 CTAs/SM. Biggest lever: reuse
   buffers (sM/sY/sBt overlap lifetimes), drop to <50KB → 4+ CTAs/SM.
2. **Serial barrier-heavy NS loop:** 3 iters × (2 transposes + 2 wgmma + f32↔bf16 round-trips + ~6
   barriers). The transpose-via-smem path is the slow way; Blackwell uses ldmatrix-TRANSPOSE (no smem
   round-trip) + keeps Ai in registers.
3. **Tiny 64×64×16 wgmma, single warpgroup:** very low tensor-core utilization; no async overlap.
4. **No warp specialization / producer-consumer pipeline** — the Blackwell speedup came largely from
   tcgen05+tmem warp-specialized pipelining (overlap KKT-MMA with inverse with U/W). My fused kernel
   serializes all stages. THIS is the "bespoke pipelining = least portable" part flagged in the spec.

### Honest read / decision:
The crux hypothesis ("TC Newton-Schulz inverse is correct & feasible on Hopper") is PROVEN. But a naive
fused kernel is 3x off tuned FLA. Closing 3x needs the sophisticated async/warp-specialized pipelining
(the least-portable Blackwell trick) + occupancy work — a substantial effort with uncertain payoff.
Per the spec's honest-exit principle, NO-GO is a legitimate stopping point with a valuable negative result.
Honing path exists (levers above) if pursued; realistic best-case is approaching parity, beating FLA is uncertain.

## Per-stage breakdown (2026-06-11) — WHERE the 3x lives
gdn_bench_stages.py — my gridded KKT->A and inverse vs FLA chunk_scaled_dot_kkt / solve_tril, 35B shapes:
  T=2048 : myKKT 0.0528 vs flaKKT 0.0190 (0.36x) | myINV 0.0947 vs flaSolve 0.0323 (0.34x)
  T=8192 : myKKT 0.0933 vs flaKKT 0.0428 (0.46x) | myINV 0.2921 vs flaSolve 0.0910 (0.31x)
  T=32768: myKKT 0.2401 vs flaKKT 0.1590 (0.66x) | myINV 1.0285 vs flaSolve 0.3321 (0.32x)
**Findings:** (1) KKT→A is the LESS-bad part, 0.36→0.66x (improving with T — clean wgmma scales). (2) the
INVERSE is the DOMINANT cost (myINV >> myKKT) AND stuck ~3x slower. The inverse is the honing target.
**Root cause of slow inverse:** I used a GLOBAL 64×64 Newton-Schulz (6× full 64×64×64 matmuls + 6 smem
transposes + ~18 barriers / 3 iters) for simplicity. FLA's solve_tril (and Blackwell) use BLOCKED 16×16
(four 16×16 NS in parallel across warps + cheap block back-sub) = far fewer FLOPs. So my inverse is
algorithmically heavier AND barrier/round-trip bound. To compete, the inverse needs the blocked-16×16
warp-mma + ldmatrix-TRANSPOSE approach (Blackwell's actual method, the path I deferred) — bigger effort.
Quicker partial win: eliminate the per-iter smem transpose round-trips via operand major-mode selection.

## Honing path A committed (2026-06-11): blocked-16x16 warp-mma inverse for FLA parity
Goal: optimize the inverse (the 3x culprit) to >= FLA speed via the FLA/Blackwell blocked-16x16 method.
16x16 tiles need WARP-level mma.sync (wgmma needs M=64), so a new primitive is required.
**API identified:** `from cutlass.cute.nvgpu import warp`; `op = warp.MmaF16BF16Op(BFloat16, Float32,
mma_inst_mnk=(16,8,16))`; `tiled = cute.make_tiled_mma(op, atom_layout_mnk, permutation_mnk=...)`.
Operands come from REGISTERS (loaded smem->reg via ldmatrix), unlike wgmma (smem-source).
ldmatrix atoms: `warp.LdMatrix8x8x16bOp(num_matrices=4)` (+ `transpose=True` variant for the on-the-fly
transpose — kills the smem round-trips that made my global NS slow). `cute.make_copy_atom(ldsm_op, BFloat16)`.
**Reference idiom:** quack `gemm_sm80.py` (Ampere warp-mma GEMM: cp.async loads + ldmatrix s2r-copy +
cute.gemm k-loop). The Blackwell kkt_inv_uw.py lines ~485-650 = the exact blocked-16x16 NS + back-sub
register dance (raw mma_bf16 from cute_utils = mma.sync.m16n8k16 inline-asm) — fallback if high-level fails.
**Plan (incremental, TDD):** (1) warp-mma 16x16 matmul probe vs torch; (2) 16x16 NS inverse (3 rounds,
ldmatrix-transpose, register-resident, no smem round-trip) vs torch block-inverse; (3) four diag blocks
in parallel (4 warps) + block back-sub -> full 64x64 Ai; (4) swap into the kernel, rebench vs FLA solve_tril.
Target: inverse from 0.32x -> >=1.0x; KKT already 0.66x at 32K (improve via multi-chunk/block if needed).

## Honing-2 progress (2026-06-11): warp-mma probe COMPILES, runtime-aborts (ldmatrix layout)
`gdn_warpmma_probe.py` (M16N8K16, one warp). Correct fragment API found in quack `sm80_utils.py`
(the warp-mma helper): `tCsA=thr_mma.partition_A(sA); tCrA=thr_mma.make_fragment_A(tCsA)` (pass the
PARTITIONED TENSOR, not a shape — that was the AssertionError); `acc=make_rmem_tensor(thr_mma.
partition_shape_C((M,N)), Float32)`; `cute.make_tiled_mma(op,(1,1,1))`. Now COMPILES but **aborts at
runtime (exit 134, native fault, no py-trace).** Hypothesis: ldmatrix needs an ldmatrix-COMPATIBLE smem
layout (8x8-matrix tiling), but my smem is plain row-major (16,16) -> ldmatrix reads wrong/OOB.
**NEXT FIX:** give sA/sB an ldmatrix-friendly layout. Options: (a) Blackwell pattern
`sA_ldsm = cute.logical_divide(sA, (16, cute.make_layout((8,2))))` then index for the warp; (b) use the
matching tiled-copy: `tc=cute.make_tiled_copy_A(ldsm_atom, tiled_mma); thrc=tc.get_slice(tid);
cute.copy(ldsm_atom, thrc.partition_S(sA), thrc.retile(tCrA))` (the tiled-copy partition handles the
ldmatrix layout) — my earlier attempt used this but with the wrong fragment ctor; retry now that
fragments are correct. Reference: quack gemm_sm80.py s2r copy. Keep M=16,N=8,K=16 (one atom) until green.

## Honing-2 GREEN (2026-06-11): warp-mma m16n8k16 primitive works
`gdn_warpmma_probe.py` (M16N8K16, one warp): C=A@Bᵀ via warp.MmaF16BF16Op + ldmatrix s2r, cosine
1.000000 vs torch. **Warp-level tensor cores driven in CuteDSL — the blocked-inverse primitive is ready.**
Working idiom:
  op = warp.MmaF16BF16Op(BF16, Float32, (16,8,16)); tiled_mma = cute.make_tiled_mma(op,(1,1,1))
  thr = tiled_mma.get_slice(tid)
  tCrA = thr.make_fragment_A(thr.partition_A(sA)); tCrB = thr.make_fragment_B(thr.partition_B(sB))
  acc = cute.make_rmem_tensor(thr.partition_shape_C((M,N)), Float32)
  ldsm_A = make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), BF16)  # A frag 8 bf16/thr
  ldsm_B = make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=2), BF16)  # B frag 4 bf16/thr
  tcA=cute.make_tiled_copy_A(ldsm_A,tiled_mma); thrcA=tcA.get_slice(tid)
  cute.copy(ldsm_A, thrcA.partition_S(sA), thrcA.retile(tCrA))   # same for B
  acc.fill(0.0); cute.gemm(tiled_mma, acc, tCrA, tCrB, acc); cute.autovec_copy(acc, thr.partition_C(gC))
  Plain row-major smem works with the tiled-copy (no special swizzle needed at 16x16). The earlier
  "illegal access" was an OOB load loop (used MM*KK//32 iters for the smaller sB), NOT ldmatrix.

### NEXT: honing-3 — 16x16 NS inverse via warp-mma
Use ldsm_trans (warp.LdMatrix8x8x16bOp(transpose=True)) for the transpose-free NS (no smem round-trip).
Per 16x16 block: init Ai=I-A, 3 rounds Ai<-2Ai-Ai@M@Ai (warp-mma + ldmatrix-trans). Then four diag
blocks across 4 warps (warp_id selects block) + block back-sub (off-diag-by-1/2/3, cf Blackwell 555-650)
-> full 64x64 Ai vs torch. Then honing-4: swap into kernel, rebench vs FLA (target inverse 0.32x->>=1.0x).

## Honing-3 step1 (2026-06-11): N=16 + normal ldmatrix OK; ldmatrix-TRANSPOSE needs 128-bit align
`gdn_warpmma_mm16.py` (16x16 A@B). IR verification: `make_tiled_mma(op,(1,1,1),permutation_mnk=(16,16,16))`
covers N=16 fine, normal ldmatrix (A) fine. But `LdMatrix8x8x16bOp(transpose=True)` FAILS:
"src ptr alignment (16 bits) does not meet requirement (128 bits)" — transpose-ldmatrix reads strided
columns from plain row-major (16,16) bf16 smem → not 128-bit aligned.
**DECISION:** for correctness-first, DROP ldmatrix-transpose; do the transpose EXPLICITLY in smem
(sXt[j,i]=sX[i,j]) then normal ldmatrix — reuses the proven warp-mma primitive. (ldmatrix-transpose
needs a swizzled/padded smem layout for 128-bit align; defer that optimization.) The blocked-16x16 speed
win comes mainly from 16x16 matmuls (vs 64x64) + 4 blocks PARALLEL across 4 warps, not from avoiding the
(now-tiny) transpose round-trip.

### NEXT: honing-3 — build 16x16 NS inverse block (single warp, explicit transpose)
sM,sX,sXt,sY,sYt bf16 16x16 + sXf f32. init M=I+A, X=I-A. 3 rounds: transpose X->sXt; warpmma(M,Xt-normal-load... 
wait: warpmma(A,B)=A@Bᵀ, so M@X needs Bop=Xᵀ=sXt loaded NORMAL -> warpmma(M,sXt)=M@(sXt)ᵀ=M@X. Then
Z=X@Y -> transpose Y->sYt, warpmma(X,sYt)=X@Y. X=2X-Z. Validate Ai vs torch inv(I+A_blk). Use N=16
permutation tiled_mma + normal ldmatrix (both verified). Then 4 diag blocks across 4 warps + block back-sub
(off-diag-by-1/2/3, Blackwell ~555-650) -> full 64x64 Ai. Then honing-4: swap into kernel, rebench vs FLA.

## Honing-3 GREEN (2026-06-11): 16x16 NS inverse block via warp-mma works
`gdn_ns16_probe.py`: single-warp 16x16 Ai=(I+A)^-1, warp-mma (N=16 permutation) + explicit smem
transpose, 3 NS rounds. cosine 1.000000, max|(I+A)Ai-I|=1.67e-3 (bf16, 3 rounds). The core compute
unit of the blocked inverse is proven. `_matmul()` helper = partition_A/B + make_fragment + tiled-copy
ldmatrix + cute.gemm; warpmma(P,Q)=P@Qᵀ so M@X=_matmul(M, Xᵀ) with Xᵀ explicit-transposed in smem.

### NEXT: honing-3b — four 16x16 diagonal blocks (4 warps) + block back-substitution -> full 64x64 Ai
Block-triangular inverse of (I+A), A strictly-lower 64x64 split into 4x4 grid of 16x16 blocks.
Diagonal blocks D_ii=(I+A_ii)^-1 via the proven 16x16 NS (warp w handles block w, parallel).
Off-diagonal (block forward-sub, lower-tri): Ai_ij = -D_ii @ (sum_{k=j..i-1} A_ik @ Ai_kj), for i>j
(cf Blackwell kernel_kkt_inv_uw.py off-diag-by-1/2/3, lines ~545-650). All via warp-mma. Validate full
64x64 Ai vs torch inv(I+A) (cosine>=0.999). Then honing-4: replace the global-NS inverse in the gridded
kernel (gdn_bench.py _grid) with this blocked inverse, rebench vs FLA solve_tril (target 0.32x->>=1.0x).

## Honing-3b-1 GREEN (2026-06-11): 4-warp PARALLEL diagonal-block NS inverse
`gdn_ns16x4_probe.py`: 128 threads = 4 warps; warp w inverts diagonal block w (the proven 16x16 NS) on
its own smem region (per-warp smem via 3D tensor [4,16,16], slice `sM[wid,None,None]` — single-index
`sM[wid]` collapses to scalar, must keep dims with None). All 4 diagonal blocks cosine 1.000000 vs the
diagonal blocks of torch inv(I+A). Multi-warp structure confirmed; warp uses thr=tiled_mma.get_slice(lane=tid%32).

### NEXT: honing-3b-2 — block back-substitution for off-diagonal blocks -> full 64x64 Ai
After the 4 diagonal D_ii (parallel), compute off-diag X_ij (i>j, block forward-sub):
  X_ij = -D_ii @ (sum_{k=j..i-1} A_ik @ X_kj)   [A_ik = off-diag block of I+A; D_ii=X_ii]
Dependency order: off-by-1 {X10,X21,X32} -> off-by-2 {X20,X31} -> off-by-3 {X30}. All 16x16 warp-mma.
Warp assignment: cf Blackwell kernel_kkt_inv_uw.py ~545-650 (off-diag-by-1/2/3, `if warp_id_<2` etc).
Validate full 64x64 Ai vs torch inv(I+A) cosine>=0.999. Then honing-4: replace global-NS inverse in
gdn_bench.py::_grid with the blocked inverse (diag-parallel + back-sub), rebench vs FLA (target 0.32x->>=1.0x).

## Honing-3b-2 de-risk GREEN (2026-06-12): bf16 blocked-inverse numerics validated
`gdn_blocked_ref.py` (torch sim, the CUDA back-sub oracle): 4x 16x16-block NS diagonals + bf16 block
forward-substitution, 5 trials all cosine 1.000000, resid ~1-2.6e-4 (BETTER than global-NS 3e-4).
**bf16 blocked algorithm is numerically excellent — CUDA back-sub is now pure engineering.**
Exact algorithm (validated): D_i=(I+A_ii)^-1 [4-warp parallel, done]; then for j in 0..3, for i in j+1..3:
  X_ij = -D_i @ (sum_{k=j..i-1} A_ik @ X_kj). Levels: off-by-1 {(1,0),(2,1),(3,2)}, off-by-2 {(2,0),(3,1)},
  off-by-3 {(3,0)}.

### NEXT: honing-3b-2 CUDA — implement the back-sub in `gdn_ns16x4_probe.py` -> full 64x64 Ai
Keep A (64x64 bf16) + X (64x64 bf16, diag=D_i) in smem. Block views via cute.zipped_divide(sX,(16,16))
-> ((16,16),(4,4)); blk(bi,bj)=view[None,(bi,bj)] OR slice sX[bi*16:(bi+1)*16, bj*16:(bj+1)*16]. Matmuls
P@Q = _matmul(P, Qᵀ) with Qᵀ explicit-transposed in a per-warp scratch. Leveled: barrier between off-by-1/2/3;
assign the level's blocks to warps (off-by-1: warps 1,2,3 do X10,X21,X32; off-by-2: warps 2,3 do X20,X31;
off-by-3: warp 3 does X30). Accumulate sum_k in f32 regs/smem. Validate full 64x64 vs gdn_blocked_ref /
torch inv (cosine>=0.999). Then honing-4: swap blocked inverse into gdn_bench.py::_grid, rebench vs FLA.

## Honing-3b-2 CUDA GREEN (2026-06-12): FULL 64x64 blocked inverse works
`gdn_blocked_inv.py`: 4-warp parallel 16x16-NS diagonal + leveled block forward-substitution
(off-by-1/2/3, warps assigned per level, CTA barriers between), all warp-mma m16n8k16. 3 trials
cosine 1.000000, resid ~2-3e-4. **The entire blocked-inverse algorithm is built & correct on Hopper.**
Gotchas fixed: (1) gA f32→sA bf16 needs .to(BFloat16) on load; (2) CuteDSL closures capturing vars
(lane) are NOT allowed in dynamic control flow → use TOP-LEVEL helper fns; (3) `range_constexpr` is
preprocessor-only (fails in un-decorated helpers) → use plain `range()` in helper fns (unrolls at trace).
Block storage: smem [4,4,16,16] block-major; block view = `s[bi,bj,None,None]`.

### NEXT (FINAL): honing-4 — swap blocked inverse into gridded kernel + rebench vs FLA = updated verdict
In `gdn_bench.py::_grid`, the inverse is currently the global-64x64 NS (the 3x-slow part). Replace it with
the blocked inverse (`_blk_inv` logic: 4-warp diagonal + back-sub) — but note the gridded kernel does ONE
(chunk,head) per CTA with the full kkt_inv_uw (KKT wgmma + inverse + U/W wgmma). Integrating warp-specialized
blocked inverse into that single-warpgroup fused kernel needs care (the diagonal uses 4 warps = the whole
warpgroup; KKT/UW wgmma also use the warpgroup). Simplest: keep stages sequential within the CTA (KKT→blocked-inv→UW),
all 128 threads. Then microbench vs FLA intra @ {2K,8K,32K}. >1.0x = GO. Compare to the prior global-NS bench
(inverse was 0.32x). Document the updated GO/NO-GO in FINDINGS + status "completed".

## Honing-4 INVERSE BENCH (2026-06-12): BLOCKED INVERSE BEATS FLA at 8K/32K
`gdn_bench_inv.py`: gridded blocked inverse vs FLA solve_tril @ 35B shapes (correctness cosine 1.0000):
  T=2048 : blocked 0.0551ms  solve_tril 0.0321ms  -> 0.58x  (small grid, launch/occupancy-bound)
  T=8192 : blocked 0.0592ms  solve_tril 0.0906ms  -> 1.53x  WIN
  T=32768: blocked 0.0987ms  solve_tril 0.3314ms  -> 3.36x  BIG WIN
**TURNAROUND vs global-NS (was 0.31-0.34x everywhere).** Blocked inverse barely grows with T
(0.055->0.099ms) while solve_tril scales ~linearly (0.032->0.331) -> the massively-parallel blocked
approach (NT*H CTAs, 4-warp-parallel diagonal + cheap back-sub) scales great on H200's many SMs.
The TC Newton-Schulz blocked inverse IS a win on Hopper at realistic prefill lengths (8K/32K).
The 2K loss is small-grid overhead (fewer CTAs than SMs) — fixable via batching multiple chunks/CTA.

### Inverse GO/NO-GO: GO at the lengths that matter (8K/32K). Remaining for END-TO-END kernel verdict:
honing-4-full = integrate blocked inverse into the gridded full kernel (KKT wgmma + blocked inverse +
U/W wgmma) — needs layout repack between block-major (warp-mma inverse) and 64x64 (wgmma U/W). Then
rebench full kkt_inv_uw vs FLA intra. KKT was 0.66x@32K (improving), inverse now 3.36x@32K, U/W TBD.

## Honing-4 U/W bench (2026-06-12): U/W is now the bottleneck + has a gridded correctness bug
`gdn_bench_uw.py`: gridded U/W vs FLA recompute_w_u. TWO issues:
  T=2048: myUW 0.124 vs fla 0.023 (0.19x); 8192: 0.295 vs 0.083 (0.28x); 32768: 0.944 vs 0.344 (0.36x).
  CORRECTNESS BROKEN gridded: U cos ~0/nan, W cos 0.31 (same matmul logic gave 0.999997 single-chunk in
  gdn_fused_full — so it's a GRIDDED indexing/layout bug: Ai_blocks reshape, sAi gmem-load into swizzled
  layout, or head/chunk/kh indexing). Must fix before trusting U/W perf.
  (FLA recompute needs bf16 A operand — pass Ai.to(bf16).)

### END-TO-END REALITY (Amdahl again): inverse win alone does NOT give parity
Per-stage @ 32K: KKT mine 0.240 vs FLA 0.159 (0.66x); INVERSE mine 0.099 vs FLA 0.331 (3.36x WIN);
U/W mine ~0.94 vs FLA 0.34 (0.36x, + buggy). Sum mine ~1.28 vs FLA intra 0.83 -> ~0.65x end-to-end
DESPITE the inverse win, because U/W (now ~0.94ms) dominates + KKT slower. **Fixing the inverse exposed
U/W as the new bottleneck.** To reach end-to-end FLA parity, must ALSO optimize U/W (single-warpgroup,
~12 barriers, 4 sequential wgmma -> needs fewer barriers / better overlap / N=128 in one wgmma not 2 halves)
and KKT (0.66x). The TC-inverse hypothesis is VALIDATED (3.36x); end-to-end parity is a broader kernel-opt effort.

### NEXT: honing-5 — (1) fix gridded U/W correctness bug; (2) optimize U/W (biggest cost) + KKT; re-sum vs FLA.

## Honing-5a (2026-06-12): gridded U/W kernel is CORRECT — the "bug" was a faulty FLA bench-reference
`gdn_debug_uw.py`: my uw_grid vs TORCH ref from the same Ai_blocks -> U cosine 0.999999, W 0.999999,
no NaN. So gdn_bench_uw's U cos~0/W 0.31 was a bad FLA comparison (recompute_w_u_fwd standalone call /
output convention), NOT my kernel. **U/W matmul is correct.** Real issue = U/W PERF (~0.36x@32K, ~0.94ms,
the new bottleneck); FLA recompute TIMING (0.34ms@32K) is valid (kernel ran) so the 2.8x-slower ratio holds.

### NEXT: honing-5b — OPTIMIZE U/W (the bottleneck). Headroom:
(1) do N=128 in ONE wgmma (tiler_mn=(64,128)) instead of 2x 64-halves -> halve wgmma + barriers;
(2) cut barriers (~12 currently); (3) avoid per-element transpose-build of sBt where possible.
Target: U/W 0.36x -> ~1.0x. Then KKT (0.66x) similarly. Re-sum vs FLA intra. HONEST NOTE: end-to-end FLA
parity is a multi-stage opt effort; the headline inverse win (3.36x) — the original ask — is already done.

## Honing-5b (2026-06-12): N=128-one-wgmma did NOT speed up U/W -> bottleneck is scalar loops/occupancy
`gdn_bench_uw2.py` (N=128 in one wgmma, ~half the barriers): U/W still ~0.124/0.297/0.976ms (≈unchanged
vs 2-halves). Correct (cos 1.0). So wgmma COUNT was not the bottleneck. U/W (~0.97ms@32K) is ~10x my
inverse (0.099ms) despite less matmul -> U/W is bound by the per-element sBt build loops (smem read +
fp32 convert + mul + exp + bf16 convert, ~128 iters/thread) + big smem (sK,sV,sZ ~80KB → low occupancy),
NOT the matmul. FLA recompute_w_u is a tuned fused triton kernel; beating it needs vectorized builds +
smem reduction (drop sZ, write acc→gmem directly; smaller V/K staging) — deeper opt, uncertain payoff.

## PROJECT SUMMARY / VERDICT (2026-06-12)
GOAL: port GDN kkt_inv_uw to Hopper (SM90); does the tensor-core approach beat FLA on H200?
CORRECTNESS: ALL green vs FLA/torch (cosine ≥0.999) — KKT→A, TC Newton-Schulz inverse (global + blocked),
U/W, full fused kernel, multi-chunk/head. The algorithm ports to Hopper correctly (wgmma + warp-mma m16n8k16).
PERF (per-stage gridded vs FLA's 3 kernels, 35B shapes, 32K):
  - KKT→A:        0.66x (mine slower; clean wgmma, improves with T)
  - INVERSE:      **3.36x @32K, 1.53x @8K (WIN)** via blocked-16x16 warp-mma NS (4-warp parallel + back-sub).
                  Turnaround from naive global-NS (0.32x). The TC-inverse hypothesis is VALIDATED on Hopper.
  - U/W:          0.36-0.45x (scalar/occupancy-bound; N=128 opt didn't help)
  END-TO-END:     ~0.65x (FLA intra still faster overall — U/W dominates, KKT slower).
VERDICT: The CORE THESIS (tensor-core matrix inverse beats FLA's CUDA-core forward-sub on Hopper) is
PROVEN (inverse 3.36x). But END-TO-END FLA parity is NOT achieved — it needs U/W + KKT optimization too
(no single stage dominates; Amdahl). Reaching full-kernel parity is a multi-week kernel-eng effort vs
FLA's mature triton; the inverse win is the high-value, defensible result. Recommend: land the inverse
finding; treat U/W/KKT parity as follow-on. All code: github.com/jasperjiaguo/sglang @ jiaguo/gdn-hopper-kkt-experiment.

## Honing-5c (2026-06-12): U/W occupancy opt — direct gmem write -> 0.45x to 0.60x @32K
`gdn_bench_uw3.py`: dropped 32KB f32 sZ staging, write wgmma acc straight to gmem (output viewed as
[NT,BT,H,DV] blocks, partition_C(gU[chunk,None,head,None])). U/W @32K 0.976->0.738ms (0.45x->0.60x),
correctness cos 1.0. Occupancy lever confirmed. Remaining cost = the transposed sBt build loops
(beta*V[kk,n] -> sBt[n,kk], 128x64, scalar convert/exp).

### NEXT: honing-5d — eliminate the transpose-build via MN-major B operand + pre-scaled Ai (big lever)
U = Ai@(beta*V) = (Ai*diag(beta)) @ V = Ais @ V. Build Ais[i,kk]=Ai[i,kk]*beta[kk] (64x64, cheap) instead
of betaV[128x64 transposed]. Then feed V DIRECTLY as B operand with b_leading_mode=OperandMajorMode.MN
(B stored [K,N]=V as-loaded, acc=A@B standard, NO transpose). Same for W: Aig[i,kk]=Ai[i,kk]*beta[kk]*exp(g[kk]),
W=Aig@K. Eliminates the 128x64 transposed scalar builds entirely -> should be the big U/W win. Needs
MN-major B smem layout (make_smem_layout COL_MAJOR) + correct operand mode. Then re-bench; then KKT opt.

## Honing-5d (2026-06-12): MN-major B operand hit a wgmma desc wall — keeping 0.60x U/W, pivot to KKT
`gdn_bench_uw4.py` (pre-scaled Ai + feed V/K directly as MN-major B): IR fails to legalize
`cute_nvgpu.make_gmma_smem_desc` with major<mn> on the [64,128] B smem (swizzle S<3,4,3>). MN-major
GMMA operands need a specific smem layout atom that the plain make_smem_layout(ROW_MAJOR,(64,128)) doesn't
produce; getting it right is non-trivial (deep wgmma operand-layout rules). NOT worth rabbit-holing.
DECISION: keep the 0.60x U/W (gdn_bench_uw3.py, occupancy opt) as the U/W result; the transpose-build
elimination is deferred. PIVOT to KKT opt (the other stage, 0.66x).

### NEXT: honing-6 — optimize KKT->A (0.66x@32K). gdn_bench_stages.py::_grid_kkt does K@Kᵀ (one wgmma,
K=128) + A modulation (decay/beta/tril) + write. Levers: (1) direct-gmem-write A (drop sA staging like
the U/W win); (2) fewer barriers; (3) the decay/beta/exp modulation is scalar 64x64 — fold/vectorize.
KKT improves with T (0.36->0.66x), so at long seqlens it's closest to parity. Then re-sum 3 stages vs FLA.

## Honing-6 DEFINITIVE END-TO-END (2026-06-12): my 3 kernels vs FLA 3 stages, one run
`gdn_bench_e2e.py` @ 35B shapes (ms; ratio = FLA_total/mine, higher=better):
  T=2048 : myKKT 0.053 + myINV 0.056 + myUW 0.116 = 0.224 | FLA 0.073 -> 0.33x
  T=8192 : 0.094 + 0.061 + 0.246 = 0.401 | FLA 0.216 -> 0.54x
  T=32768: 0.242 + 0.100 + 0.738 = 1.080 | FLA 0.837 -> 0.77x  (vs fused FLA intra 0.754 -> 0.70x)
**End-to-end kkt_inv_uw = 0.77x@32K, climbing with T (0.33->0.54->0.77).** Up from ~0.65x pre-U/W-opt.
U/W (0.738ms) = 68% of my total -> THE gate to parity. Inverse now tiny (0.100ms, the 3.36x win). KKT 0.242.
Trend suggests longer seqlens approach parity IF U/W transpose-build is cracked (MN-major, walled) — that
is the one remaining high-value lever; KKT near-ceiling. Current best stage kernels: gdn_bench_stages
(KKT), gdn_bench_inv (inverse), gdn_bench_uw3 (U/W occupancy-opt).

## Honing-7 (2026-06-12): transposed-gmem-load U/W is WORSE -> U/W tractable levers EXHAUSTED
`gdn_bench_uw5.py` (load V^T/K^T directly + pre-scaled Ai, no rebuild): 0.18/0.51/1.78ms (0.17/0.22/0.25x),
correct (cos 1.0) but ~2.4x SLOWER than uw3 — the strided/uncoalesced transposed gmem reads dominate.
**U/W optimization space exhausted:** occupancy/direct-gmem-write = WIN (0.45->0.60x, uw3 = best);
MN-major B (eliminate transpose) = WALLED (gmma desc legalization); transposed-load = WORSE (uncoalesced).
The 128x64 transpose-rebuild in uw3 is the residual cost; removing it needs the MN-major path (blocked).

## FINAL STATE (2026-06-12)
- Correctness: ALL green vs FLA/torch (cosine ≥0.999) across every stage + full pipeline.
- INVERSE: **3.36x@32K WIN** (blocked-16x16 warp-mma NS) — TC-inverse-on-Hopper thesis PROVEN.
- U/W: 0.60x@32K (occupancy-optimized; further blocked by wgmma MN-major wall).
- KKT: 0.66x@32K (near-ceiling, wgmma-bound).
- END-TO-END kkt_inv_uw: **0.77x@32K** (climbing with T: 0.33/0.54/0.77 @ 2K/8K/32K).
VERDICT: core thesis proven (inverse 3.36x). End-to-end ~0.77x@32K, gated on U/W's transpose-build whose
only real fix (MN-major) is blocked by a wgmma descriptor wall. Tractable optimization space is exhausted;
reaching full parity needs cracking that wgmma-layout wall (deep, uncertain). RECOMMEND CONSOLIDATE.
Code: github.com/jasperjiaguo/sglang @ jiaguo/gdn-hopper-kkt-experiment (b8d0833 + uncommitted honing files).

## Honing-8/9 (2026-06-15): MN-major B wgmma CRACKED + U/W is occupancy-bound (not transpose/bw)
**CRACKED the MN-major wall** (gdn_mnmajor_probe.py, gdn_bench_uw6.py): feed V/K DIRECTLY to wgmma, no
transpose-build. Recipe: B shape (N,K); atom = warpgroup.make_smem_layout_atom(get_smem_layout_atom(
LayoutEnum.COL_MAJOR, dtype, N), dtype); cute.tile_to_shape(atom,(N,K),order=(1,0)); b_leading_mode=MN;
write sV[n,k]=gV[k,n] with n inner (coalesced). cosine 1.0. THIS is what Blackwell (transpose_B desc) and
FLA (tl.dot) do. Pre-scale Ai by beta (cheap 64x64) instead of the 128x64 betaV transpose-build.
RESULT: U/W 0.738->0.657ms @32K (0.60x->0.67x). Modest -> transpose-build was NOT the dominant cost.
bf16 output (uw7) = NO change (0.671) -> NOT bandwidth-bound either.
**DIAGNOSIS: U/W is OCCUPANCY/LATENCY-bound.** 16384 tiny 1-(chunk,head)-per-CTA launches; serial chain
(~6 barriers + 2 wgmma + builds); actual wgmma compute ~34us total. End-to-end @32K w/ uw6: KKT 0.242 +
inv 0.100 + UW 0.657 = 0.999ms vs FLA 0.837 -> 0.84x (up from 0.77x).
### NEXT for 1.5x: FUSION + amortize per-CTA. Combine KKT+inverse+U/W in ONE kernel (Ai stays in smem,
no gmem round-trips), and/or process multiple chunks/heads per CTA to hide latency. This is the Blackwell
approach (warp-specialized + multi-stage async pipeline = their 3x). Best U/W kernel now = gdn_bench_uw6.py.

## FlashInfer SM90 tuning (2026-06-21): direct-store beta path beats NS

Target: installed FlashInfer GDN SM90 path in SGLang (`flashinfer/flat/hopper/collective/
flat_collective_tma_warpspecialized_delta_rule.hpp`), using stable finite inputs
(`H=16,D=128`, L2-normalized K, `g=0`, `beta=0.01`, `use_qk_l2norm_in_kernel=True`).

Baseline timings:
- T=1024: 0.0541 ms
- T=2048: 0.1033 ms
- T=8192: 0.4021 ms

Tried NS ports:
- Scalar NS8 replacing FlashInfer's 8x8 seed: correct but slower (2 rounds: 0.0638/0.1207/0.4646 ms;
  3 rounds: 0.0711/0.1361/0.5252 ms).
- Scratch-buffer NS16: compiled and correct, but slower even with fewer rounds; zero-iteration lower bound was
  still slower than baseline due the scratch/barrier overhead.
- vLLM PR #43273-style NS16 (`Ai_new = 2*Ai - Ai @ M @ Ai`, 3 rounds): bitwise-identical on stress checks,
  but much slower (0.0749/0.1490/0.5929 ms). The vLLM win is tied to its SM100/tcgen05 pipelined kernel
  structure, not directly portable as a faster drop-in for FlashInfer's SM90 collective.

Best patch:
- For the beta-enabled path, skip `CollectiveInverse::compute()` and the smem reload.
- Reuse the already half-quantized KKT fragment, apply the 8x8 diagonal/upper correction in the final register
  transform, multiply by `Beta(t)`, and store once.
- Timings: **0.0444/0.0852/0.3301 ms** at 1K/2K/8K (~18% faster than baseline).
- Stress checks vs restored baseline-equivalent outputs were bitwise-identical for beta in `{0.1, 1.0}` and
  q/v scale in `{0.01, 0.1, 1.0}`.

Checked-in patch: `flashinfer_gdn_direct_store_transform.patch`.
