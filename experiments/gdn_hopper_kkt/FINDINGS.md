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
