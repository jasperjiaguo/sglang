"""P1c-ii: FUSED in-smem Newton-Schulz inverse (one kernel, smem-resident).

Ai = (I+A)^-1 via X <- 2X - X@M@X, all in smem. Each round = 2 wgmma matmuls with logical
smem transposes between (gemm computes P@Q^T, so P@Q needs Q^T in smem). f32 accumulator is
round-tripped through smem + cast to bf16 to feed the next matmul. Validated vs torch exact.
"""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hh
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.nvgpu.warpgroup import OperandSource
import quack.sm90_utils as smu

BT = 64
NITER = 5
DEV = "cuda"


@cute.kernel
def _fused_inv(gA: cute.Tensor, gAi: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    tiled_mma = hh.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float32, (1, 1, 1), (BT, BT), OperandSource.SMEM,
    )
    lay = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT))
    plain = cute.make_layout((BT, BT), stride=(BT, 1))
    smem = cutlass.utils.SmemAllocator()
    sM = smem.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sX = smem.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sXt = smem.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sY = smem.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sYt = smem.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sZ = smem.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    sXf = smem.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)

    # init: M = I + A, X = I - A
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        i = idx // BT
        j = idx % BT
        a = gA[i, j]
        m = a
        x = cutlass.Float32(0.0) - a
        if i == j:
            m = m + cutlass.Float32(1.0)
            x = x + cutlass.Float32(1.0)
        sM[i, j] = m.to(cutlass.BFloat16)
        sX[i, j] = x.to(cutlass.BFloat16)
        sXf[i, j] = x
    cute.arch.barrier()

    thr_mma = tiled_mma.get_slice(tidx)
    for _ in cutlass.range_constexpr(NITER):
        # Xt = X^T
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            sXt[idx % BT, idx // BT] = sX[idx // BT, idx % BT]
        cute.arch.barrier()
        # Y = M @ X  = gemm(M, Xt)
        accY, tM, tXt = smu.partition_fragment_ABC(thr_mma, (BT, BT, BT), sM, sXt)
        smu.gemm(tiled_mma, accY, tM, tXt, zero_init=True)
        cute.autovec_copy(accY, thr_mma.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            sY[idx // BT, idx % BT] = sZ[idx // BT, idx % BT].to(cutlass.BFloat16)
        cute.arch.barrier()
        # Yt = Y^T
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            sYt[idx % BT, idx // BT] = sY[idx // BT, idx % BT]
        cute.arch.barrier()
        # Z = X @ Y = gemm(X, Yt)
        accZ, tX, tYt = smu.partition_fragment_ABC(thr_mma, (BT, BT, BT), sX, sYt)
        smu.gemm(tiled_mma, accZ, tX, tYt, zero_init=True)
        cute.autovec_copy(accZ, thr_mma.partition_C(sZ))
        cute.arch.barrier()
        # X = 2X - Z
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            i = idx // BT
            j = idx % BT
            xnew = cutlass.Float32(2.0) * sXf[i, j] - sZ[i, j]
            sXf[i, j] = xnew
            sX[i, j] = xnew.to(cutlass.BFloat16)
        cute.arch.barrier()

    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        gAi[idx // BT, idx % BT] = sXf[idx // BT, idx % BT]


@cute.jit
def _run(mA: cute.Tensor, mAi: cute.Tensor):
    _fused_inv(mA, mAi).launch(grid=(1, 1, 1), block=(128, 1, 1))


_c = {}

def fused_inv(A):
    Ai = torch.empty(BT, BT, device=DEV, dtype=torch.float32)
    mA, mAi = from_dlpack(A.contiguous(), assumed_align=16), from_dlpack(Ai, assumed_align=16)
    if "i" not in _c:
        _c["i"] = cute.compile(_run, mA, mAi)
    _c["i"](mA, mAi)
    torch.cuda.synchronize()
    return Ai


def main():
    torch.manual_seed(0)
    k = torch.randn(BT, 128, device=DEV); k = k / k.norm(dim=-1, keepdim=True)
    beta = torch.rand(BT, device=DEV).clamp_min(0.1)
    g = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(BT, device=DEV)), 0)
    A = (k @ k.T * torch.exp(g[:, None] - g[None, :]) * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)

    Ai = fused_inv(A)
    Ai_ref = torch.linalg.inv(torch.eye(BT, device=DEV) + A)
    c = float(torch.dot(Ai.reshape(-1), Ai_ref.reshape(-1)) / (Ai.norm() * Ai_ref.norm() + 1e-12))
    resid = float(((torch.eye(BT, device=DEV) + A) @ Ai - torch.eye(BT, device=DEV)).abs().max())
    ok = c >= 0.999
    print(f"[P1c-ii fused NS inverse] cosine={c:.6f} max|(I+A)Ai-I|={resid:.3e} -> {'PASS' if ok else 'FAIL'}")
    print(f"=== P1c-ii {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
