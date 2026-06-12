"""Honing-4: GO/NO-GO on the inverse. Gridded blocked inverse (4-warp 16x16-NS diag + back-sub)
vs FLA solve_tril, at 35B shapes. The inverse was the 3x bottleneck (global-NS = 0.32x); this
measures whether the blocked rework reaches/beats FLA solve_tril parity."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import warp

import gdn_shapes as S
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.fla.solve_tril import solve_tril

B, BT, NB = 16, 64, 4
HG, H = S.NUM_K_HEADS, S.NUM_V_HEADS
NITER = 3
DEV = "cuda"


def _mma_acc(mma, thr, ldsm, tcA, tcB, lane, sA, sBt, acc):
    tCrA = thr.make_fragment_A(thr.partition_A(sA))
    tCrB = thr.make_fragment_B(thr.partition_B(sBt))
    cute.copy(ldsm, tcA.get_slice(lane).partition_S(sA), tcA.get_slice(lane).retile(tCrA))
    cute.copy(ldsm, tcB.get_slice(lane).partition_S(sBt), tcB.get_slice(lane).retile(tCrB))
    cute.gemm(mma, acc, tCrA, tCrB, acc)


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
def _blk_inv_grid(gA: cute.Tensor, gAi: cute.Tensor):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    wid = tid // 32
    lane = tid % 32
    op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    mma = cute.make_tiled_mma(op, (1, 1, 1), permutation_mnk=(16, 16, 16))
    thr = mma.get_slice(lane)
    ldsm = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16)
    tcA = cute.make_tiled_copy_A(ldsm, mma)
    tcB = cute.make_tiled_copy_B(ldsm, mma)
    sm = cutlass.utils.SmemAllocator()
    LB = cute.make_layout((NB, NB, B, B), stride=(NB * B * B, B * B, B, 1))
    L1 = cute.make_layout((NB, B, B), stride=(B * B, B, 1))
    sA = sm.allocate_tensor(cutlass.BFloat16, LB, byte_alignment=128)
    sX = sm.allocate_tensor(cutlass.BFloat16, LB, byte_alignment=128)
    sM = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sXw = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sT1 = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sT2 = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sZ = sm.allocate_tensor(cutlass.Float32, L1, byte_alignment=128)
    sXf = sm.allocate_tensor(cutlass.Float32, L1, byte_alignment=128)

    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        r = idx // BT
        c = idx % BT
        sA[r // B, c // B, r % B, c % B] = gA[bid, r, c].to(cutlass.BFloat16)
    cute.arch.barrier()

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
        accY = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
        accY.fill(0.0)
        _mma_acc(mma, thr, ldsm, tcA, tcB, lane, mM, mT1, accY)
        cute.autovec_copy(accY, thr.partition_C(mZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mT2[idx // B, idx % B] = mZ[idx // B, idx % B].to(cutlass.BFloat16)
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mT1[idx % B, idx // B] = mT2[idx // B, idx % B]
        cute.arch.sync_warp()
        accZ = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
        accZ.fill(0.0)
        _mma_acc(mma, thr, ldsm, tcA, tcB, lane, mXw, mT1, accZ)
        cute.autovec_copy(accZ, thr.partition_C(mZ))
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
        _offdiag(mma, thr, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, wid, wid - 1, 1)
    cute.arch.barrier()
    if wid >= 2:
        _offdiag(mma, thr, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, wid, wid - 2, 2)
    cute.arch.barrier()
    if wid >= 3:
        _offdiag(mma, thr, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, wid, wid - 3, 3)
    cute.arch.barrier()

    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        r = idx // BT
        c = idx % BT
        if r // B >= c // B:
            gAi[bid, r, c] = sX[r // B, c // B, r % B, c % B].to(cutlass.Float32)
        else:
            gAi[bid, r, c] = cutlass.Float32(0.0)


@cute.jit
def _run(mA, mAi):
    nb = cute.size(mA, mode=[0])
    _blk_inv_grid(mA, mAi).launch(grid=(nb, 1, 1), block=(128, 1, 1))


_c = {}

def blocked_inv_grid(A):   # A: [NBT, 64, 64] f32
    Ai = torch.zeros_like(A)
    args = [from_dlpack(A.contiguous(), assumed_align=16), from_dlpack(Ai, assumed_align=16)]
    if "b" not in _c:
        _c["b"] = cute.compile(_run, *args)
    _c["b"](*args)
    torch.cuda.synchronize()
    return Ai


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
    print("Device:", torch.cuda.get_device_name(0), "| blocked inverse vs FLA solve_tril, NITER=%d" % NITER)
    print("%7s | %11s %11s %8s" % ("T", "blocked(ms)", "solve_tril", "ratio"))
    results = []
    for T in S.SEQLENS:
        NT = T // BT
        k = torch.randn(1, T, HG, B * 8, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
        beta = torch.rand(1, T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
        g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)
        A_fla = chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT)         # [1,T,H,64] strictly-lower
        # reshape to per-(chunk,head) 64x64 blocks: bid = chunk*H + head
        A_blocks = A_fla[0].reshape(NT, BT, H, BT).permute(0, 2, 1, 3).reshape(NT * H, BT, BT).contiguous()

        # correctness vs solve_tril on block 0
        Ai_mine = blocked_inv_grid(A_blocks)
        Ai_fla = solve_tril(A_fla)
        Ai_fla_b0 = Ai_fla[0, 0:BT, 0, :]
        c0 = cos(Ai_mine[0], Ai_fla_b0)

        t_mine = time_fn(lambda: blocked_inv_grid(A_blocks))
        t_fla = time_fn(lambda: solve_tril(A_fla))
        sp = t_fla / t_mine
        results.append((T, t_mine, t_fla, sp, c0))
        print("%7d | %11.4f %11.4f %7.2fx   (block0 cosine vs solve_tril=%.4f)" % (T, t_mine, t_fla, sp, c0))

    allgo = all(sp > 1.0 for _, _, _, sp, _ in results)
    print(f"\n=== INVERSE VERDICT: blocked-inverse {'>=' if allgo else '<'} FLA solve_tril : {'GO' if allgo else 'still slower'} ===")
    print("(prior global-NS inverse was 0.31-0.34x)")


if __name__ == "__main__":
    main()
