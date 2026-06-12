"""Honing-3b-2 CUDA: full 64x64 blocked inverse = 4-warp parallel 16x16-NS diagonal + block
forward-substitution off-diagonals. Validates vs torch inv(I+A). All matmuls warp-mma (m16n8k16).
warpmma(P,Q)=P@Q^T, so P@Q = _matmul(P, transpose(Q)) with the transpose materialized in smem."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import warp

B, BT, NB = 16, 64, 4
NITER = 3
DEV = "cuda"


def _mma_acc(tiled_mma, thr, ldsm, tcA, tcB, lane, sA, sBt, acc):
    """acc += sA @ (sBt)^T  (sBt already holds the transpose of the desired right operand)."""
    tCrA = thr.make_fragment_A(thr.partition_A(sA))
    tCrB = thr.make_fragment_B(thr.partition_B(sBt))
    cute.copy(ldsm, tcA.get_slice(lane).partition_S(sA), tcA.get_slice(lane).retile(tCrA))
    cute.copy(ldsm, tcB.get_slice(lane).partition_S(sBt), tcB.get_slice(lane).retile(tCrB))
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)


def _offdiag(mma, thr, ldsm, tcA, tcB, lane, sA, sX, mT1, mT2, mSf, i, j, nk):
    """X_ij = -D_i @ (sum_{k=j..i-1} A_ik @ X_kj). Top-level (no closure capture)."""
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
def _blk_inv(gA: cute.Tensor, gAi: cute.Tensor):
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
    # per-warp scratch
    sM = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sXw = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sT1 = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)   # transpose scratch
    sT2 = sm.allocate_tensor(cutlass.BFloat16, L1, byte_alignment=128)
    sZ = sm.allocate_tensor(cutlass.Float32, L1, byte_alignment=128)
    sXf = sm.allocate_tensor(cutlass.Float32, L1, byte_alignment=128)

    # load A (block-major)
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        r = idx // BT
        c = idx % BT
        sA[r // B, c // B, r % B, c % B] = gA[r, c].to(cutlass.BFloat16)
    cute.arch.barrier()

    # ---- diagonal: warp wid inverts block (wid,wid) via 16x16 NS -> sX[wid,wid] ----
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
            mT1[idx % B, idx // B] = mXw[idx // B, idx % B]      # X^T
        cute.arch.sync_warp()
        accY = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
        accY.fill(0.0)
        _mma_acc(mma, thr, ldsm, tcA, tcB, lane, mM, mT1, accY)   # M@X
        cute.autovec_copy(accY, thr.partition_C(mZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mT2[idx // B, idx % B] = mZ[idx // B, idx % B].to(cutlass.BFloat16)   # Y
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mT1[idx % B, idx // B] = mT2[idx // B, idx % B]      # Y^T
        cute.arch.sync_warp()
        accZ = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
        accZ.fill(0.0)
        _mma_acc(mma, thr, ldsm, tcA, tcB, lane, mXw, mT1, accZ)  # X@Y
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

    # ---- block forward-substitution: X_ij = -D_i @ (sum_{k=j..i-1} A_ik @ X_kj) ----
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
            gAi[r, c] = sX[r // B, c // B, r % B, c % B].to(cutlass.Float32)
        else:
            gAi[r, c] = cutlass.Float32(0.0)


@cute.jit
def _run(mA, mAi):
    _blk_inv(mA, mAi).launch(grid=(1, 1, 1), block=(128, 1, 1))


_c = {}

def blocked_inv(A):
    Ai = torch.zeros(BT, BT, device=DEV, dtype=torch.float32)
    args = [from_dlpack(A.contiguous(), assumed_align=16), from_dlpack(Ai, assumed_align=16)]
    if "b" not in _c:
        _c["b"] = cute.compile(_run, *args)
    _c["b"](*args)
    torch.cuda.synchronize()
    return Ai


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    torch.manual_seed(0)
    worst = 1.0
    for t in range(3):
        k = torch.randn(BT, 128, device=DEV); k = k / k.norm(dim=-1, keepdim=True)
        beta = torch.rand(BT, device=DEV).clamp_min(0.1)
        g = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(BT, device=DEV)), 0)
        A = (k @ k.T * torch.exp(g[:, None] - g[None, :]) * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)
        Ai = blocked_inv(A)
        Ai_ref = torch.linalg.inv(torch.eye(BT, device=DEV) + A)
        c = cos(Ai, Ai_ref)
        resid = float(((torch.eye(BT, device=DEV) + A) @ Ai - torch.eye(BT, device=DEV)).abs().max())
        worst = min(worst, c)
        print(f"  trial {t}: cosine={c:.6f} resid={resid:.3e}")
    ok = worst >= 0.999
    print(f"[64x64 blocked inverse] worst cosine={worst:.6f} -> {'PASS' if ok else 'FAIL'}")
    print("BLOCKINV OK" if ok else "BLOCKINV FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
