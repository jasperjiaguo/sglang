"""Fusion step 1: fuse INVERSE + U/W into one kernel (Ai stays in smem — kills the Ai gmem round-trip).
KKT stays separate (feeds A). End-to-end = my_kkt + fused_invuw vs FLA's 3 stages."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hh
from cutlass.cute.nvgpu import OperandMajorMode, warp, warpgroup
from cutlass.cute.nvgpu.warpgroup import OperandSource
import quack.sm90_utils as smu

import gdn_shapes as S
from gdn_bench_stages import my_kkt
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.fla.solve_tril import solve_tril
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd

B, BT, NB = 16, 64, 4
HG, H = S.NUM_K_HEADS, S.NUM_V_HEADS
REP = H // HG
DK = DV = 128
NITER = 3
DEV = "cuda"


def mn_layout(dtype, n, k):
    atom = warpgroup.make_smem_layout_atom(
        smu.sm90_utils_og.get_smem_layout_atom(LayoutEnum.COL_MAJOR, dtype, n), dtype)
    return cute.tile_to_shape(atom, (n, k), order=(1, 0))


def _mma_acc(tiled_mma, thr, ldsm, tcA, tcB, lane, sA, sBt, acc):
    tCrA = thr.make_fragment_A(thr.partition_A(sA))
    tCrB = thr.make_fragment_B(thr.partition_B(sBt))
    cute.copy(ldsm, tcA.get_slice(lane).partition_S(sA), tcA.get_slice(lane).retile(tCrA))
    cute.copy(ldsm, tcB.get_slice(lane).partition_S(sBt), tcB.get_slice(lane).retile(tCrB))
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)


def _offdiag(mma, thr, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, i, j, nk):
    accS = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
    accS.fill(0.0)
    for s in range(nk):
        k = j + s
        Xkj = sX[k, j, None, None]
        for it in range(B * B // 32):
            idx = it * 32 + lane
            mT1[idx % B, idx // B] = Xkj[idx // B, idx % B]
        cute.arch.sync_warp()
        _mma_acc(mma, thr, ldsm, tcA, tcB, lane, sA[i, k, None, None], mT1, accS)
    cute.autovec_copy(accS, thr.partition_C(mSf))
    cute.arch.sync_warp()
    for it in range(B * B // 32):
        idx = it * 32 + lane
        mT2[idx % B, idx // B] = mSf[idx // B, idx % B].to(cutlass.BFloat16)
    cute.arch.sync_warp()
    accX = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
    accX.fill(0.0)
    _mma_acc(mma, thr, ldsm, tcA, tcB, lane, sX[i, i, None, None], mT2, accX)
    cute.autovec_copy(accX, thr.partition_C(mSf))
    cute.arch.sync_warp()
    for it in range(B * B // 32):
        idx = it * 32 + lane
        sX[i, j, idx // B, idx % B] = (cutlass.Float32(0.0) - mSf[idx // B, idx % B]).to(cutlass.BFloat16)


@cute.kernel
def _fused_invuw(gA, gK, gV, gBeta, gG, gU, gW):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    chunk = bid // H
    head = bid % H
    kh = head // REP
    r0 = chunk * BT
    wid = tid // 32
    lane = tid % 32

    mma_inv = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
                                  (1, 1, 1), permutation_mnk=(16, 16, 16))
    thr_inv = mma_inv.get_slice(lane)
    ldsm = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16)
    tcA = cute.make_tiled_copy_A(ldsm, mma_inv)
    tcB = cute.make_tiled_copy_B(ldsm, mma_inv)
    mma_uw = hh.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K,
        OperandMajorMode.MN, cutlass.Float32, (1, 1, 1), (BT, DV), OperandSource.SMEM)
    thr_uw = mma_uw.get_slice(tid)

    sm = cutlass.utils.SmemAllocator()
    LB = cute.make_layout((NB, NB, B, B), stride=(NB * B * B, B * B, B, 1))
    L1 = cute.make_layout((NB, B, B), stride=(B * B, B, 1))
    layA = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT))
    layB = mn_layout(cutlass.BFloat16, DV, BT)
    sA = sm.allocate_tensor(cutlass.BFloat16, LB, byte_alignment=128)
    sX = sm.allocate_tensor(cutlass.BFloat16, LB, byte_alignment=128)
    sM = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sXw = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sT1 = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sT2 = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sZ = sm.allocate_tensor(cutlass.Float32, L1, byte_alignment=128)
    sXf = sm.allocate_tensor(cutlass.Float32, L1, byte_alignment=128)
    sAi = sm.allocate_tensor(cutlass.BFloat16, layA.outer, byte_alignment=1024, swizzle=layA.inner)
    sAs = sm.allocate_tensor(cutlass.BFloat16, layA.outer, byte_alignment=1024, swizzle=layA.inner)
    sV = sm.allocate_tensor(cutlass.BFloat16, layB.outer, byte_alignment=1024, swizzle=layB.inner)
    sK = sm.allocate_tensor(cutlass.BFloat16, layB.outer, byte_alignment=1024, swizzle=layB.inner)
    sBeta = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)
    sG = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)

    # load A (block-major), V/K (MN-major), beta, g
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        r = idx // BT
        c = idx % BT
        sA[r // B, c // B, r % B, c % B] = gA[bid, r, c].to(cutlass.BFloat16)
    for it in cutlass.range_constexpr(DV * BT // 128):
        idx = it * 128 + tid
        k = idx // DV
        n = idx % DV
        sV[n, k] = gV[r0 + k, head, n]
        sK[n, k] = gK[r0 + k, kh, n]
    if tid < BT:
        sBeta[tid] = gBeta[r0 + tid, head]
        sG[tid] = gG[r0 + tid, head]
    cute.arch.barrier()

    # ===== blocked inverse: sA -> sX (Ai) =====
    mM, mXw, mT1, mT2 = sM[wid, None, None], sXw[wid, None, None], sT1[wid, None, None], sT2[wid, None, None]
    mZ, mXf = sZ[wid, None, None], sXf[wid, None, None]
    Aii = sA[wid, wid, None, None]
    for it in cutlass.range_constexpr(B * B // 32):
        idx = it * 32 + lane
        i = idx // B
        j = idx % B
        a = Aii[i, j].to(cutlass.Float32)
        m = a
        x = cutlass.Float32(0.0) - a
        if i == j:
            m = m + cutlass.Float32(1.0)
            x = x + cutlass.Float32(1.0)
        mM[i, j] = m.to(cutlass.BFloat16)
        mXw[i, j] = x.to(cutlass.BFloat16)
        mXf[i, j] = x
    cute.arch.sync_warp()
    for _ in cutlass.range_constexpr(NITER):
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mT1[idx % B, idx // B] = mXw[idx // B, idx % B]
        cute.arch.sync_warp()
        aY = cute.make_rmem_tensor(thr_inv.partition_shape_C((B, B)), cutlass.Float32)
        aY.fill(0.0)
        _mma_acc(mma_inv, thr_inv, ldsm, tcA, tcB, lane, mM, mT1, aY)
        cute.autovec_copy(aY, thr_inv.partition_C(mZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mT2[idx // B, idx % B] = mZ[idx // B, idx % B].to(cutlass.BFloat16)
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mT1[idx % B, idx // B] = mT2[idx // B, idx % B]
        cute.arch.sync_warp()
        aZ = cute.make_rmem_tensor(thr_inv.partition_shape_C((B, B)), cutlass.Float32)
        aZ.fill(0.0)
        _mma_acc(mma_inv, thr_inv, ldsm, tcA, tcB, lane, mXw, mT1, aZ)
        cute.autovec_copy(aZ, thr_inv.partition_C(mZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            i = idx // B
            j = idx % B
            xn = cutlass.Float32(2.0) * mXf[i, j] - mZ[i, j]
            mXf[i, j] = xn
            mXw[i, j] = xn.to(cutlass.BFloat16)
        cute.arch.sync_warp()
    for it in cutlass.range_constexpr(B * B // 32):
        idx = it * 32 + lane
        sX[wid, wid, idx // B, idx % B] = mXw[idx // B, idx % B]
    cute.arch.barrier()
    mSf = sXf[wid, None, None]
    if wid >= 1:
        _offdiag(mma_inv, thr_inv, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, wid, wid - 1, 1)
    cute.arch.barrier()
    if wid >= 2:
        _offdiag(mma_inv, thr_inv, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, wid, wid - 2, 2)
    cute.arch.barrier()
    if wid >= 3:
        _offdiag(mma_inv, thr_inv, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, wid, wid - 3, 3)
    cute.arch.barrier()

    # ===== repack sX (block-major Ai) -> sAi (K-major 64x64), lower-tri only =====
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        i = idx // BT
        j = idx % BT
        if i // B >= j // B:
            sAi[i, j] = sX[i // B, j // B, i % B, j % B]
        else:
            sAi[i, j] = cutlass.BFloat16(0.0)
    cute.arch.barrier()

    # ===== U/W (MN-major, Ai resident) =====
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        sAs[idx // BT, idx % BT] = (sAi[idx // BT, idx % BT].to(cutlass.Float32) * sBeta[idx % BT]).to(cutlass.BFloat16)
    cute.arch.barrier()
    accU, tA, tB = smu.partition_fragment_ABC(thr_uw, (BT, DV, BT), sAs, sV)
    smu.gemm(mma_uw, accU, tA, tB, zero_init=True)
    aUb = cute.make_rmem_tensor(accU.shape, cutlass.BFloat16)
    aUb.store(accU.load().to(cutlass.BFloat16))
    cute.autovec_copy(aUb, thr_uw.partition_C(gU[chunk, None, head, None]))
    cute.arch.barrier()
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        sAs[idx // BT, idx % BT] = (sAi[idx // BT, idx % BT].to(cutlass.Float32) * sBeta[idx % BT] * cute.math.exp(sG[idx % BT])).to(cutlass.BFloat16)
    cute.arch.barrier()
    accW, tA2, tB2 = smu.partition_fragment_ABC(thr_uw, (BT, DK, BT), sAs, sK)
    smu.gemm(mma_uw, accW, tA2, tB2, zero_init=True)
    aWb = cute.make_rmem_tensor(accW.shape, cutlass.BFloat16)
    aWb.store(accW.load().to(cutlass.BFloat16))
    cute.autovec_copy(aWb, thr_uw.partition_C(gW[chunk, None, head, None]))


@cute.jit
def _run(mA, mK, mV, mB, mG, mU, mW):
    nb = cute.size(mA, mode=[0])
    _fused_invuw(mA, mK, mV, mB, mG, mU, mW).launch(grid=(nb, 1, 1), block=(128, 1, 1))


_c = {}

def fused_invuw(A, k, v, beta, g, T):
    NT = T // BT
    U = torch.empty(T, H, DV, device=DEV, dtype=torch.bfloat16)
    W = torch.empty(T, H, DK, device=DEV, dtype=torch.bfloat16)
    args = [from_dlpack(A, assumed_align=16), from_dlpack(k, assumed_align=16), from_dlpack(v, assumed_align=16),
            from_dlpack(beta, assumed_align=16), from_dlpack(g, assumed_align=16),
            from_dlpack(U.view(NT, BT, H, DV), assumed_align=16), from_dlpack(W.view(NT, BT, H, DK), assumed_align=16)]
    if T not in _c:
        _c[T] = cute.compile(_run, *args)
    _c[T](*args)
    torch.cuda.synchronize()
    return U, W


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    print("Device:", torch.cuda.get_device_name(0), "| FUSED inv+uw : end-to-end (my_kkt + fused_invuw) vs FLA")
    print("%7s | %8s %10s %9s | %8s | %7s" % ("T", "myKKT", "fused_iuw", "myTOT", "flaTOT", "ratio"))
    for T in S.SEQLENS:
        NT = T // BT
        k = torch.randn(1, T, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
        v = torch.randn(1, T, H, DV, device=DEV, dtype=torch.bfloat16)
        beta = torch.rand(1, T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
        g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)
        A_raw = chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT)
        A_blocks = A_raw[0].reshape(NT, BT, H, BT).permute(0, 2, 1, 3).reshape(NT * H, BT, BT).contiguous()
        Ai_fla = solve_tril(A_raw); Ai_bf = Ai_fla.to(torch.bfloat16)
        kc, vc, bc, gc = k[0].contiguous(), v[0].contiguous(), beta[0].contiguous(), g[0].contiguous()

        # correctness: fused vs torch (block 0)
        U, W = fused_invuw(A_blocks, kc, vc, bc, gc, T)
        Ai0 = torch.linalg.inv(torch.eye(BT, device=DEV) + A_blocks[0].float())
        Uref0 = Ai0 @ (bc[:BT, 0, None].float() * vc[:BT, 0, :].float())
        cU = cos(U[:BT, 0, :], Uref0)

        t_kkt = time_fn(lambda: my_kkt(kc, bc, gc, NT))
        t_fiuw = time_fn(lambda: fused_invuw(A_blocks, kc, vc, bc, gc, T))
        mine = t_kkt + t_fiuw
        t_fla = (time_fn(lambda: chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT))
                 + time_fn(lambda: solve_tril(A_raw))
                 + time_fn(lambda: recompute_w_u_fwd(k, v, beta, g, Ai_bf, None)))
        print("%7d | %8.4f %10.4f %9.4f | %8.4f | %6.2fx  (U cos=%.4f)" % (T, t_kkt, t_fiuw, mine, t_fla, t_fla / mine, cU))


if __name__ == "__main__":
    main()
