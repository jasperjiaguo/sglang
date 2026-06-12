"""Honing-3b-1: four 16x16 diagonal-block NS inverses in PARALLEL across 4 warps.
Confirms the multi-warp structure (warp w inverts diagonal block w). Each warp does the proven
16x16 NS on its own smem region. Validates the 4 diagonal blocks of Ai vs torch inv(I+A)."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import warp

B = 16
BT = 64
NW = 4          # 4 diagonal blocks / 4 warps
NITER = 3
DEV = "cuda"


def _matmul(tiled_mma, thr, ldsm, tcA, tcB, lane, sA, sB):
    tCsA = thr.partition_A(sA)
    tCsB = thr.partition_B(sB)
    tCrA = thr.make_fragment_A(tCsA)
    tCrB = thr.make_fragment_B(tCsB)
    acc = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
    cute.copy(ldsm, tcA.get_slice(lane).partition_S(sA), tcA.get_slice(lane).retile(tCrA))
    cute.copy(ldsm, tcB.get_slice(lane).partition_S(sB), tcB.get_slice(lane).retile(tCrB))
    acc.fill(0.0)
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)
    return acc


@cute.kernel
def _ns16x4(gA: cute.Tensor, gAi: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    wid = tid // 32
    lane = tid % 32
    o = wid * B    # this warp's diagonal block offset along both axes

    op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, (1, 1, 1), permutation_mnk=(16, 16, 16))
    thr = tiled_mma.get_slice(lane)
    ldsm = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16)
    tcA = cute.make_tiled_copy_A(ldsm, tiled_mma)
    tcB = cute.make_tiled_copy_B(ldsm, tiled_mma)

    sm = cutlass.utils.SmemAllocator()
    L3 = cute.make_layout((NW, B, B), stride=(B * B, B, 1))
    sM = sm.allocate_tensor(cutlass.BFloat16, L3, byte_alignment=128)
    sX = sm.allocate_tensor(cutlass.BFloat16, L3, byte_alignment=128)
    sXt = sm.allocate_tensor(cutlass.BFloat16, L3, byte_alignment=128)
    sY = sm.allocate_tensor(cutlass.BFloat16, L3, byte_alignment=128)
    sYt = sm.allocate_tensor(cutlass.BFloat16, L3, byte_alignment=128)
    sZ = sm.allocate_tensor(cutlass.Float32, L3, byte_alignment=128)
    sXf = sm.allocate_tensor(cutlass.Float32, L3, byte_alignment=128)

    mM, mX, mXt = sM[wid, None, None], sX[wid, None, None], sXt[wid, None, None]
    mY, mYt = sY[wid, None, None], sYt[wid, None, None]
    mZ, mXf = sZ[wid, None, None], sXf[wid, None, None]

    for it in cutlass.range_constexpr(B * B // 32):
        idx = it * 32 + lane
        i = idx // B
        j = idx % B
        a = gA[o + i, o + j]
        m = a
        x = cutlass.Float32(0.0) - a
        if i == j:
            m = m + cutlass.Float32(1.0)
            x = x + cutlass.Float32(1.0)
        mM[i, j] = m.to(cutlass.BFloat16)
        mX[i, j] = x.to(cutlass.BFloat16)
        mXf[i, j] = x
    cute.arch.sync_warp()

    for _ in cutlass.range_constexpr(NITER):
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mXt[idx % B, idx // B] = mX[idx // B, idx % B]
        cute.arch.sync_warp()
        accY = _matmul(tiled_mma, thr, ldsm, tcA, tcB, lane, mM, mXt)
        cute.autovec_copy(accY, thr.partition_C(mZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mY[idx // B, idx % B] = mZ[idx // B, idx % B].to(cutlass.BFloat16)
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            mYt[idx % B, idx // B] = mY[idx // B, idx % B]
        cute.arch.sync_warp()
        accZ = _matmul(tiled_mma, thr, ldsm, tcA, tcB, lane, mX, mYt)
        cute.autovec_copy(accZ, thr.partition_C(mZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + lane
            i = idx // B
            j = idx % B
            xn = cutlass.Float32(2.0) * mXf[i, j] - mZ[i, j]
            mXf[i, j] = xn
            mX[i, j] = xn.to(cutlass.BFloat16)
        cute.arch.sync_warp()

    # write this warp's diagonal block to gAi (off-diagonal left zero by host init)
    for it in cutlass.range_constexpr(B * B // 32):
        idx = it * 32 + lane
        gAi[o + idx // B, o + idx % B] = mXf[idx // B, idx % B]


@cute.jit
def _run(mA, mAi):
    _ns16x4(mA, mAi).launch(grid=(1, 1, 1), block=(128, 1, 1))


_c = {}

def ns16x4(A):
    Ai = torch.zeros(BT, BT, device=DEV, dtype=torch.float32)
    args = [from_dlpack(A.contiguous(), assumed_align=16), from_dlpack(Ai, assumed_align=16)]
    if "n" not in _c:
        _c["n"] = cute.compile(_run, *args)
    _c["n"](*args)
    torch.cuda.synchronize()
    return Ai


def main():
    torch.manual_seed(0)
    k = torch.randn(BT, 128, device=DEV); k = k / k.norm(dim=-1, keepdim=True)
    beta = torch.rand(BT, device=DEV).clamp_min(0.1)
    A = (k @ k.T * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)
    Ai = ns16x4(A)
    Ai_ref = torch.linalg.inv(torch.eye(BT, device=DEV) + A)
    # validate the 4 diagonal 16x16 blocks (= diagonal blocks of the full inverse)
    worst = 1.0
    for w in range(NW):
        s = slice(w * B, (w + 1) * B)
        g, r = Ai[s, s].reshape(-1), Ai_ref[s, s].reshape(-1)
        c = float(torch.dot(g, r) / (g.norm() * r.norm() + 1e-12))
        worst = min(worst, c)
        print(f"  block {w}: cosine={c:.6f}")
    ok = worst >= 0.999
    print(f"[4-warp diagonal NS] worst block cosine={worst:.6f} -> {'PASS' if ok else 'FAIL'}")
    print("NS16x4 OK" if ok else "NS16x4 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
