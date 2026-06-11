"""2c-0: minimal standalone wgmma GEMM probe — prove I can drive Hopper tensor cores
in CuteDSL before building the Newton-Schulz inverse on top.

Computes C[64,64] = A[64,64] @ B[64,64]^T (the cute.gemm K-contraction convention)
via a single warpgroup wgmma, validates vs torch. Uses the quack/cutlass SM90 helpers.
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

MM = NN = KK = 64


@cute.kernel
def _gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    tiled_mma = hh.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float32, (1, 1, 1), (MM, NN), OperandSource.SMEM,
    )
    sA_layout = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (MM, KK))
    sB_layout = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (NN, KK))
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout.outer, byte_alignment=1024, swizzle=sA_layout.inner)
    sB = smem.allocate_tensor(cutlass.BFloat16, sB_layout.outer, byte_alignment=1024, swizzle=sB_layout.inner)

    for it in cutlass.range_constexpr(MM * KK // 128):
        idx = it * 128 + tidx
        sA[idx // KK, idx % KK] = gA[idx // KK, idx % KK]
    for it in cutlass.range_constexpr(NN * KK // 128):
        idx = it * 128 + tidx
        sB[idx // KK, idx % KK] = gB[idx // KK, idx % KK]
    cute.arch.barrier()

    thr_mma = tiled_mma.get_slice(tidx)
    acc, tCrA, tCrB = smu.partition_fragment_ABC(thr_mma, (MM, NN, KK), sA, sB)
    smu.gemm(tiled_mma, acc, tCrA, tCrB, zero_init=True)

    tCgC = thr_mma.partition_C(gC)
    cute.autovec_copy(acc, tCgC)


@cute.jit
def _run(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    _gemm_kernel(mA, mB, mC).launch(grid=(1, 1, 1), block=(128, 1, 1))


_compiled = {}

def wgmma_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A,B: [64,64] bf16 -> C = A @ B^T, f32."""
    C = torch.empty(MM, NN, device=A.device, dtype=torch.float32)
    mA, mB, mC = (from_dlpack(x, assumed_align=16) for x in (A, B, C))
    if "g" not in _compiled:
        _compiled["g"] = cute.compile(_run, mA, mB, mC)
    _compiled["g"](mA, mB, mC)
    torch.cuda.synchronize()
    return C


def main():
    torch.manual_seed(0)
    A = torch.randn(MM, KK, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(NN, KK, device="cuda", dtype=torch.bfloat16)
    C = wgmma_gemm(A, B)
    ref = A.float() @ B.float().T
    c = float(torch.dot(C.reshape(-1), ref.reshape(-1)) / (C.norm() * ref.norm() + 1e-12))
    m = float((C - ref).abs().max())
    ok = c >= 0.999
    print(f"[2c-0 wgmma GEMM] cosine={c:.6f} maxabs={m:.3e} -> {'PASS' if ok else 'FAIL'}")
    print("WGMMA OK" if ok else "WGMMA FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
