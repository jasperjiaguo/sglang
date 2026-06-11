"""P1c-iii: FULL fused kkt_inv_uw in ONE kernel (single chunk/head correctness gate).

Combines KKT->A (P1c-i) + in-smem NS inverse (P1c-ii) + U/W, all smem-resident, one launch.
U = Ai@(beta·V), W = Ai@(beta·exp(g)·K), each N=128 done as two 64-wide halves via the 64x64x64
wgmma. Validated vs torch reference (exact inverse). Gridding over chunks/heads + bench is next.
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

BT, DK, DV = 64, 128, 128
NITER = 5
DEV = "cuda"


@cute.kernel
def _fused_full(gK: cute.Tensor, gV: cute.Tensor, gBeta: cute.Tensor, gG: cute.Tensor,
                gU: cute.Tensor, gW: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    mma = hh.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float32, (1, 1, 1), (BT, BT), OperandSource.SMEM)
    lay = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT))
    layK = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, DK))
    plain = cute.make_layout((BT, BT), stride=(BT, 1))
    sm = cutlass.utils.SmemAllocator()
    sK = sm.allocate_tensor(cutlass.BFloat16, layK.outer, byte_alignment=1024, swizzle=layK.inner)
    sV = sm.allocate_tensor(cutlass.BFloat16, layK.outer, byte_alignment=1024, swizzle=layK.inner)
    sM = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sX = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sXt = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sY = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sBt = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sZ = sm.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    sA = sm.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    sBeta = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)
    sG = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)

    # load K, V
    for it in cutlass.range_constexpr(BT * DK // 128):
        idx = it * 128 + tidx
        sK[idx // DK, idx % DK] = gK[idx // DK, idx % DK]
        sV[idx // DV, idx % DV] = gV[idx // DV, idx % DV]
    if tidx < BT:
        sBeta[tidx] = gBeta[tidx]
        sG[tidx] = gG[tidx]
    cute.arch.barrier()

    thr = mma.get_slice(tidx)
    # ---- KKT -> A (in sA f32) ----
    accA, tK1, tK2 = smu.partition_fragment_ABC(thr, (BT, BT, DK), sK, sK)
    smu.gemm(mma, accA, tK1, tK2, zero_init=True)
    cute.autovec_copy(accA, thr.partition_C(sA))
    cute.arch.barrier()
    # ---- A = strictLower(beta*KKt*decay); M = I+A, X = I-A ----
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        i = idx // BT
        j = idx % BT
        a = sA[i, j] * sBeta[i] * cute.math.exp(sG[i] - sG[j])
        if i <= j:
            a = cutlass.Float32(0.0)
        m = a
        x = cutlass.Float32(0.0) - a
        if i == j:
            m = m + cutlass.Float32(1.0)
            x = x + cutlass.Float32(1.0)
        sM[i, j] = m.to(cutlass.BFloat16)
        sX[i, j] = x.to(cutlass.BFloat16)
        sA[i, j] = x   # reuse sA as Xf (f32 running X)
    cute.arch.barrier()
    # ---- NS inverse: X <- 2X - X@M@X ----
    for _ in cutlass.range_constexpr(NITER):
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            sXt[idx % BT, idx // BT] = sX[idx // BT, idx % BT]
        cute.arch.barrier()
        aY, tM, tXt = smu.partition_fragment_ABC(thr, (BT, BT, BT), sM, sXt)
        smu.gemm(mma, aY, tM, tXt, zero_init=True)
        cute.autovec_copy(aY, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            sY[idx // BT, idx % BT] = sZ[idx // BT, idx % BT].to(cutlass.BFloat16)
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            sXt[idx % BT, idx // BT] = sY[idx // BT, idx % BT]   # sXt := Y^T
        cute.arch.barrier()
        aZ, tX, tYt = smu.partition_fragment_ABC(thr, (BT, BT, BT), sX, sXt)
        smu.gemm(mma, aZ, tX, tYt, zero_init=True)
        cute.autovec_copy(aZ, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            i = idx // BT
            j = idx % BT
            xn = cutlass.Float32(2.0) * sA[i, j] - sZ[i, j]
            sA[i, j] = xn
            sX[i, j] = xn.to(cutlass.BFloat16)
        cute.arch.barrier()
    # sX = Ai (bf16). ---- U = Ai@(beta*V), W = Ai@(beta*exp(g)*K) ----
    for half in cutlass.range_constexpr(2):
        # U half: sBt[n,k] = beta[k]*V[k, half*64+n]
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            n = idx // BT
            kk = idx % BT
            sBt[n, kk] = (sBeta[kk] * sV[kk, half * BT + n].to(cutlass.Float32)).to(cutlass.BFloat16)
        cute.arch.barrier()
        aU, tAi, tB = smu.partition_fragment_ABC(thr, (BT, BT, BT), sX, sBt)
        smu.gemm(mma, aU, tAi, tB, zero_init=True)
        cute.autovec_copy(aU, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            gU[idx // BT, half * BT + idx % BT] = sZ[idx // BT, idx % BT]
        cute.arch.barrier()
        # W half: sBt[n,k] = beta[k]*exp(g[k])*K[k, half*64+n]
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            n = idx // BT
            kk = idx % BT
            sBt[n, kk] = (sBeta[kk] * cute.math.exp(sG[kk]) * sK[kk, half * BT + n].to(cutlass.Float32)).to(cutlass.BFloat16)
        cute.arch.barrier()
        aW, tAi2, tB2 = smu.partition_fragment_ABC(thr, (BT, BT, BT), sX, sBt)
        smu.gemm(mma, aW, tAi2, tB2, zero_init=True)
        cute.autovec_copy(aW, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            gW[idx // BT, half * BT + idx % BT] = sZ[idx // BT, idx % BT]
        cute.arch.barrier()


@cute.jit
def _run(mK, mV, mB, mG, mU, mW):
    _fused_full(mK, mV, mB, mG, mU, mW).launch(grid=(1, 1, 1), block=(128, 1, 1))


_c = {}

def fused_uw(k, v, beta, g):
    U = torch.empty(BT, DV, device=DEV, dtype=torch.float32)
    W = torch.empty(BT, DK, device=DEV, dtype=torch.float32)
    args = [from_dlpack(k.to(torch.bfloat16).contiguous(), assumed_align=16),
            from_dlpack(v.to(torch.bfloat16).contiguous(), assumed_align=16),
            from_dlpack(beta.contiguous(), assumed_align=16),
            from_dlpack(g.contiguous(), assumed_align=16),
            from_dlpack(U, assumed_align=16), from_dlpack(W, assumed_align=16)]
    if "f" not in _c:
        _c["f"] = cute.compile(_run, *args)
    _c["f"](*args)
    torch.cuda.synchronize()
    return U, W


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    torch.manual_seed(0)
    k = torch.randn(BT, DK, device=DEV); k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(BT, DV, device=DEV)
    beta = torch.rand(BT, device=DEV).clamp_min(0.1)
    g = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(BT, device=DEV)), 0)

    U, W = fused_uw(k, v, beta, g)
    A = (k @ k.T * torch.exp(g[:, None] - g[None, :]) * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)
    Ai = torch.linalg.inv(torch.eye(BT, device=DEV) + A)
    U_ref = Ai @ (beta[:, None] * v)
    W_ref = Ai @ (beta[:, None] * torch.exp(g)[:, None] * k)
    cu, cw = cos(U, U_ref), cos(W, W_ref)
    print(f"[P1c-iii fused full] U cosine={cu:.6f}  W cosine={cw:.6f}")
    ok = cu >= 0.999 and cw >= 0.999
    print(f"=== P1c-iii {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
