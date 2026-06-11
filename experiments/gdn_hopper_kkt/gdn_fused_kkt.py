"""P1c-i: FUSED KKT->A in a single smem-resident kernel (one chunk/head per CTA).

K@K^T computed as a single wgmma (M=N=64, K=128; the tiled_mma k-loop handles the 128
contraction), result written to smem, then decay*beta*strict-tril applied in-place and stored.
Foundation for fusing the NS inverse + U/W next. Validated vs the A reference.
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

BT, DK = 64, 128
DEV = "cuda"


@cute.kernel
def _fused_kkt(gK: cute.Tensor, gBeta: cute.Tensor, gG: cute.Tensor, gA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    tiled_mma = hh.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float32, (1, 1, 1), (BT, BT), OperandSource.SMEM,
    )
    sK_layout = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, DK))
    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout.outer, byte_alignment=1024, swizzle=sK_layout.inner)
    sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT, 1)), byte_alignment=128)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)

    for it in cutlass.range_constexpr(BT * DK // 128):
        idx = it * 128 + tidx
        sK[idx // DK, idx % DK] = gK[idx // DK, idx % DK]
    if tidx < BT:
        sBeta[tidx] = gBeta[tidx]
        sG[tidx] = gG[tidx]
    cute.arch.barrier()

    thr_mma = tiled_mma.get_slice(tidx)
    acc, tCrA, tCrB = smu.partition_fragment_ABC(thr_mma, (BT, BT, DK), sK, sK)
    smu.gemm(tiled_mma, acc, tCrA, tCrB, zero_init=True)        # acc = K @ K^T
    cute.autovec_copy(acc, thr_mma.partition_C(sA))
    cute.arch.barrier()

    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        i = idx // BT
        j = idx % BT
        if i > j:
            gA[i, j] = sA[i, j] * sBeta[i] * cute.math.exp(sG[i] - sG[j])
        else:
            gA[i, j] = cutlass.Float32(0.0)


@cute.jit
def _run(mK: cute.Tensor, mBeta: cute.Tensor, mG: cute.Tensor, mA: cute.Tensor):
    _fused_kkt(mK, mBeta, mG, mA).launch(grid=(1, 1, 1), block=(128, 1, 1))


_c = {}

def fused_A(k, beta, g):
    A = torch.empty(BT, BT, device=DEV, dtype=torch.float32)
    mK, mB, mG, mA = (from_dlpack(x, assumed_align=16) for x in
                      (k.to(torch.bfloat16).contiguous(), beta, g, A))
    if "k" not in _c:
        _c["k"] = cute.compile(_run, mK, mB, mG, mA)
    _c["k"](mK, mB, mG, mA)
    torch.cuda.synchronize()
    return A


def main():
    torch.manual_seed(0)
    k = torch.randn(BT, DK, device=DEV); k = k / k.norm(dim=-1, keepdim=True)
    beta = torch.rand(BT, device=DEV).clamp_min(0.1)
    g = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(BT, device=DEV)), 0)

    A_got = fused_A(k, beta, g)
    KKt = k @ k.T
    decay = torch.exp(g[:, None] - g[None, :])
    A_ref = (KKt * decay * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)

    c = float(torch.dot(A_got.reshape(-1), A_ref.reshape(-1)) / (A_got.norm() * A_ref.norm() + 1e-12))
    m = float((A_got - A_ref).abs().max())
    ok = c >= 0.999
    print(f"[P1c-i fused KKT->A] cosine={c:.6f} maxabs={m:.3e} -> {'PASS' if ok else 'FAIL'}")
    print(f"=== P1c-i {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
