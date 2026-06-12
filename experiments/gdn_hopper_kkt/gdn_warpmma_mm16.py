"""Honing-3 step 1: 16x16 STANDARD matmul C=A@B via warp-mma, with B loaded by
ldmatrix-TRANSPOSE (no smem round-trip) and N=16 (two n=8 atom steps via permutation).
This is the exact primitive the 16x16 NS inverse needs (M@X). Validate vs torch A@B."""
import os, sys
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import warp

N16 = 16
DEV = "cuda"


@cute.kernel
def _mm(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((N16, N16), stride=(N16, 1)), byte_alignment=128)
    sB = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((N16, N16), stride=(N16, 1)), byte_alignment=128)
    for it in cutlass.range_constexpr(N16 * N16 // 32):
        idx = it * 32 + tidx
        sA[idx // N16, idx % N16] = gA[idx // N16, idx % N16]
        sB[idx // N16, idx % N16] = gB[idx // N16, idx % N16]
    cute.arch.sync_warp()

    op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, (1, 1, 1), permutation_mnk=(16, 16, 16))
    thr = tiled_mma.get_slice(tidx)

    tCsA = thr.partition_A(sA)
    tCsB = thr.partition_B(sB)
    tCrA = thr.make_fragment_A(tCsA)
    tCrB = thr.make_fragment_B(tCsB)
    acc = cute.make_rmem_tensor(thr.partition_shape_C((N16, N16)), cutlass.Float32)

    ldsm_A = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16)
    # B via TRANSPOSE load: gives B^T in the B-fragment -> acc = A @ (B^T)^T = A@B
    ldsm_Bt = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True), cutlass.BFloat16)
    tcA = cute.make_tiled_copy_A(ldsm_A, tiled_mma)
    tcB = cute.make_tiled_copy_B(ldsm_Bt, tiled_mma)
    thrcA = tcA.get_slice(tidx)
    thrcB = tcB.get_slice(tidx)
    cute.copy(ldsm_A, thrcA.partition_S(sA), thrcA.retile(tCrA))
    cute.copy(ldsm_Bt, thrcB.partition_S(sB), thrcB.retile(tCrB))

    acc.fill(0.0)
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)
    cute.autovec_copy(acc, thr.partition_C(gC))


@cute.jit
def _run(mA, mB, mC):
    _mm(mA, mB, mC).launch(grid=(1, 1, 1), block=(32, 1, 1))


_c = {}

def mm16(A, B):
    C = torch.empty(N16, N16, device=DEV, dtype=torch.float32)
    args = [from_dlpack(A, assumed_align=16), from_dlpack(B, assumed_align=16), from_dlpack(C, assumed_align=16)]
    if "m" not in _c:
        _c["m"] = cute.compile(_run, *args)
    _c["m"](*args)
    torch.cuda.synchronize()
    return C


def main():
    torch.manual_seed(0)
    A = torch.randn(N16, N16, device=DEV, dtype=torch.bfloat16)
    B = torch.randn(N16, N16, device=DEV, dtype=torch.bfloat16)
    C = mm16(A, B)
    ref = A.float() @ B.float()   # standard matmul
    c = float(torch.dot(C.reshape(-1), ref.reshape(-1)) / (C.norm() * ref.norm() + 1e-12))
    m = float((C - ref).abs().max())
    ok = c >= 0.999
    print(f"[16x16 A@B (ldmatrix-transpose, N=16)] cosine={c:.6f} maxabs={m:.3e} -> {'PASS' if ok else 'FAIL'}")
    print("MM16 OK" if ok else "MM16 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
