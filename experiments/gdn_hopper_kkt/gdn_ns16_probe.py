"""Honing-3: 16x16 Newton-Schulz inverse block via warp-mma (single warp, explicit transpose).
Ai=(I+A)^-1 for a 16x16 strictly-lower A. warpmma(P,Q)=P@Q^T, so M@X = warpmma(M, X^T) with X^T
explicitly transposed in smem (normal ldmatrix). 3 NS rounds (2^4=16 exact for 16x16). Validate vs torch."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import warp

B = 16
NITER = 3
DEV = "cuda"


def _matmul(tiled_mma, thr, ldsm, tcA, tcB, tid, sA, sB):
    """acc = sA @ sB^T (16x16), via warp-mma + normal ldmatrix."""
    tCsA = thr.partition_A(sA)
    tCsB = thr.partition_B(sB)
    tCrA = thr.make_fragment_A(tCsA)
    tCrB = thr.make_fragment_B(tCsB)
    acc = cute.make_rmem_tensor(thr.partition_shape_C((B, B)), cutlass.Float32)
    cute.copy(ldsm, tcA.get_slice(tid).partition_S(sA), tcA.get_slice(tid).retile(tCrA))
    cute.copy(ldsm, tcB.get_slice(tid).partition_S(sB), tcB.get_slice(tid).retile(tCrB))
    acc.fill(0.0)
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)
    return acc


@cute.kernel
def _ns16(gA: cute.Tensor, gAi: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, (1, 1, 1), permutation_mnk=(16, 16, 16))
    thr = tiled_mma.get_slice(tid)
    ldsm = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16)
    tcA = cute.make_tiled_copy_A(ldsm, tiled_mma)
    tcB = cute.make_tiled_copy_B(ldsm, tiled_mma)

    sm = cutlass.utils.SmemAllocator()
    L = cute.make_layout((B, B), stride=(B, 1))
    sM = sm.allocate_tensor(cutlass.BFloat16, L, byte_alignment=128)
    sX = sm.allocate_tensor(cutlass.BFloat16, L, byte_alignment=128)
    sXt = sm.allocate_tensor(cutlass.BFloat16, L, byte_alignment=128)
    sY = sm.allocate_tensor(cutlass.BFloat16, L, byte_alignment=128)
    sYt = sm.allocate_tensor(cutlass.BFloat16, L, byte_alignment=128)
    sZ = sm.allocate_tensor(cutlass.Float32, L, byte_alignment=128)
    sXf = sm.allocate_tensor(cutlass.Float32, L, byte_alignment=128)

    for it in cutlass.range_constexpr(B * B // 32):
        idx = it * 32 + tid
        i = idx // B
        j = idx % B
        a = gA[i, j]
        m = a
        x = cutlass.Float32(0.0) - a
        if i == j:
            m = m + cutlass.Float32(1.0)
            x = x + cutlass.Float32(1.0)
        sM[i, j] = m.to(cutlass.BFloat16)
        sX[i, j] = x.to(cutlass.BFloat16)
        sXf[i, j] = x
    cute.arch.sync_warp()

    for _ in cutlass.range_constexpr(NITER):
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + tid
            sXt[idx % B, idx // B] = sX[idx // B, idx % B]
        cute.arch.sync_warp()
        accY = _matmul(tiled_mma, thr, ldsm, tcA, tcB, tid, sM, sXt)   # M @ (X^T)^T = M@X
        cute.autovec_copy(accY, thr.partition_C(sZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + tid
            sY[idx // B, idx % B] = sZ[idx // B, idx % B].to(cutlass.BFloat16)
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + tid
            sYt[idx % B, idx // B] = sY[idx // B, idx % B]
        cute.arch.sync_warp()
        accZ = _matmul(tiled_mma, thr, ldsm, tcA, tcB, tid, sX, sYt)   # X @ (Y^T)^T = X@Y
        cute.autovec_copy(accZ, thr.partition_C(sZ))
        cute.arch.sync_warp()
        for it in cutlass.range_constexpr(B * B // 32):
            idx = it * 32 + tid
            i = idx // B
            j = idx % B
            xn = cutlass.Float32(2.0) * sXf[i, j] - sZ[i, j]
            sXf[i, j] = xn
            sX[i, j] = xn.to(cutlass.BFloat16)
        cute.arch.sync_warp()

    for it in cutlass.range_constexpr(B * B // 32):
        idx = it * 32 + tid
        gAi[idx // B, idx % B] = sXf[idx // B, idx % B]


@cute.jit
def _run(mA, mAi):
    _ns16(mA, mAi).launch(grid=(1, 1, 1), block=(32, 1, 1))


_c = {}

def ns16(A):
    Ai = torch.empty(B, B, device=DEV, dtype=torch.float32)
    args = [from_dlpack(A.contiguous(), assumed_align=16), from_dlpack(Ai, assumed_align=16)]
    if "n" not in _c:
        _c["n"] = cute.compile(_run, *args)
    _c["n"](*args)
    torch.cuda.synchronize()
    return Ai


def main():
    torch.manual_seed(0)
    # 16x16 strictly-lower A (realistic scale)
    M = torch.randn(B, 64, device=DEV); M = M / M.norm(dim=-1, keepdim=True)
    beta = torch.rand(B, device=DEV).clamp_min(0.1)
    A = (M @ M.T * beta[:, None]) * torch.tril(torch.ones(B, B, device=DEV), -1)
    Ai = ns16(A)
    Ai_ref = torch.linalg.inv(torch.eye(B, device=DEV) + A)
    c = float(torch.dot(Ai.reshape(-1), Ai_ref.reshape(-1)) / (Ai.norm() * Ai_ref.norm() + 1e-12))
    resid = float(((torch.eye(B, device=DEV) + A) @ Ai - torch.eye(B, device=DEV)).abs().max())
    ok = c >= 0.999
    print(f"[16x16 NS inverse] cosine={c:.6f} max|(I+A)Ai-I|={resid:.3e} -> {'PASS' if ok else 'FAIL'}")
    print("NS16 OK" if ok else "NS16 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
