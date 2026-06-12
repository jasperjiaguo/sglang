"""Honing-2: warp-level mma.sync m16n8k16 probe (the primitive for the blocked-16x16 inverse).
Hopper-native warp MMA (NOT SM80-specific). C[16,16]=A@B^T via one warp + ldmatrix s2r, vs torch."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import warp

MM, NN, KK = 16, 8, 16   # one m16n8k16 atom (no N-tiling yet)
DEV = "cuda"


@cute.kernel
def _wmma(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((MM, KK), stride=(KK, 1)), byte_alignment=128)
    sB = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((NN, KK), stride=(KK, 1)), byte_alignment=128)
    for it in cutlass.range_constexpr(MM * KK // 32):
        idx = it * 32 + tidx
        sA[idx // KK, idx % KK] = gA[idx // KK, idx % KK]
    for it in cutlass.range_constexpr(NN * KK // 32):
        idx = it * 32 + tidx
        sB[idx // KK, idx % KK] = gB[idx // KK, idx % KK]
    cute.arch.sync_warp()

    op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (MM, 8, KK))
    tiled_mma = cute.make_tiled_mma(op, (1, 1, 1))
    thr_mma = tiled_mma.get_slice(tidx)

    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCrA = thr_mma.make_fragment_A(tCsA)
    tCrB = thr_mma.make_fragment_B(tCsB)
    acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((MM, NN)), cutlass.Float32)

    # ldmatrix s2r via tiled-copy (builds the ldmatrix-aware partition); A frag=8 bf16/thr (x4), B frag=4 (x2)
    ldsm_A = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16)
    ldsm_B = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=2), cutlass.BFloat16)
    tcA = cute.make_tiled_copy_A(ldsm_A, tiled_mma)
    tcB = cute.make_tiled_copy_B(ldsm_B, tiled_mma)
    thrcA = tcA.get_slice(tidx)
    thrcB = tcB.get_slice(tidx)
    cute.copy(ldsm_A, thrcA.partition_S(sA), thrcA.retile(tCrA))
    cute.copy(ldsm_B, thrcB.partition_S(sB), thrcB.retile(tCrB))

    acc.fill(0.0)
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)

    cute.autovec_copy(acc, thr_mma.partition_C(gC))


@cute.jit
def _run(mA, mB, mC):
    _wmma(mA, mB, mC).launch(grid=(1, 1, 1), block=(32, 1, 1))


_c = {}

def wmma(A, B):
    C = torch.empty(MM, NN, device=DEV, dtype=torch.float32)
    args = [from_dlpack(A, assumed_align=16), from_dlpack(B, assumed_align=16), from_dlpack(C, assumed_align=16)]
    if "w" not in _c:
        _c["w"] = cute.compile(_run, *args)
    _c["w"](*args)
    torch.cuda.synchronize()
    return C


def main():
    torch.manual_seed(0)
    A = torch.randn(MM, KK, device=DEV, dtype=torch.bfloat16)
    B = torch.randn(NN, KK, device=DEV, dtype=torch.bfloat16)
    C = wmma(A, B)
    ref = A.float() @ B.float().T
    c = float(torch.dot(C.reshape(-1), ref.reshape(-1)) / (C.norm() * ref.norm() + 1e-12))
    m = float((C - ref).abs().max())
    ok = c >= 0.999
    print(f"[warp-mma 16x16] cosine={c:.6f} maxabs={m:.3e} -> {'PASS' if ok else 'FAIL'}")
    print("WARPMMA OK" if ok else "WARPMMA FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
