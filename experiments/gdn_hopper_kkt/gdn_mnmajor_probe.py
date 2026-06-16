"""Crack the MN-major B wgmma: U = Ai @ V with V fed DIRECTLY (no transpose-build).
A (Ai) K-major; B (V) MN-major built the quack way: shape (N,K), order (1,0). Validate vs torch."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hh
from cutlass.cute.nvgpu import OperandMajorMode, warpgroup
from cutlass.cute.nvgpu.warpgroup import OperandSource
import quack.sm90_utils as smu

M, N, K = 64, 128, 64   # U[64,128] = Ai[64,64] @ V[64,128]
DEV = "cuda"


def mn_major_smem_layout(dtype, n, k):
    """B operand, MN-major (N contiguous): quack recipe — shape (N,K), atom sized by N, order (1,0)."""
    atom = warpgroup.make_smem_layout_atom(
        smu.sm90_utils_og.get_smem_layout_atom(LayoutEnum.COL_MAJOR, dtype, n), dtype)
    return cute.tile_to_shape(atom, (n, k), order=(1, 0))


@cute.kernel
def _k(gAi: cute.Tensor, gV: cute.Tensor, gU: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    mma = hh.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16,
        OperandMajorMode.K, OperandMajorMode.MN, cutlass.Float32, (1, 1, 1), (M, N), OperandSource.SMEM)
    layA = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (M, K))
    layB = mn_major_smem_layout(cutlass.BFloat16, N, K)
    sm = cutlass.utils.SmemAllocator()
    sAi = sm.allocate_tensor(cutlass.BFloat16, layA.outer, byte_alignment=1024, swizzle=layA.inner)
    sV = sm.allocate_tensor(cutlass.BFloat16, layB.outer, byte_alignment=1024, swizzle=layB.inner)

    for it in cutlass.range_constexpr(M * K // 128):
        idx = it * 128 + tid
        sAi[idx // K, idx % K] = gAi[idx // K, idx % K]
    # V fed directly: sV is [N,K]; sV[n,k] = V[k,n]. n inner -> coalesced gmem + smem.
    for it in cutlass.range_constexpr(N * K // 128):
        idx = it * 128 + tid
        k = idx // N
        n = idx % N
        sV[n, k] = gV[k, n]
    cute.arch.barrier()

    thr = mma.get_slice(tid)
    accU, tA, tB = smu.partition_fragment_ABC(thr, (M, N, K), sAi, sV)
    smu.gemm(mma, accU, tA, tB, zero_init=True)
    cute.autovec_copy(accU, thr.partition_C(gU))


@cute.jit
def _run(mAi, mV, mU):
    _k(mAi, mV, mU).launch(grid=(1, 1, 1), block=(128, 1, 1))


_c = {}

def mnmm(Ai, V):
    U = torch.empty(M, N, device=DEV, dtype=torch.float32)
    args = [from_dlpack(Ai, assumed_align=16), from_dlpack(V, assumed_align=16), from_dlpack(U, assumed_align=16)]
    if "k" not in _c:
        _c["k"] = cute.compile(_run, *args)
    _c["k"](*args)
    torch.cuda.synchronize()
    return U


def main():
    torch.manual_seed(0)
    Ai = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    V = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    U = mnmm(Ai, V)
    ref = Ai.float() @ V.float()
    c = float(torch.dot(U.reshape(-1), ref.reshape(-1)) / (U.norm() * ref.norm() + 1e-12))
    m = float((U - ref).abs().max())
    ok = c >= 0.999
    print(f"[MN-major Ai@V direct] cosine={c:.6f} maxabs={m:.3e} -> {'PASS' if ok else 'FAIL'}")
    print("MNMAJOR OK" if ok else "MNMAJOR FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
