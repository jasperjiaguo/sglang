"""CuteDSL toolchain smoke test — proves @cute.kernel/@cute.jit compile + launch + dlpack
work STANDALONE on this pod (no `import sglang`). Foundation for P1b milestone 2a.

Kernel: copy a [M,N] f32 tensor through smem, element per thread-tile. Validates:
  - cute.compile + .launch
  - from_dlpack on torch tensors
  - SmemAllocator + smem round-trip
Asserts output == input exactly.
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

M, N = 64, 64
THREADS = 64  # one thread per row; each copies N elems


@cute.kernel
def copy_kernel(gIn: cute.Tensor, gOut: cute.Tensor, smem_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sBuf = smem.allocate_tensor(cutlass.Float32, smem_layout, 128)

    # each thread = one row; copy row tidx through smem
    if tidx < M:
        for j in cutlass.range_constexpr(N):
            sBuf[tidx, j] = gIn[tidx, j]
    cute.arch.barrier()
    if tidx < M:
        for j in cutlass.range_constexpr(N):
            gOut[tidx, j] = sBuf[tidx, j]


@cute.jit
def run(mIn: cute.Tensor, mOut: cute.Tensor):
    smem_layout = cute.make_layout((M, N), stride=(N, 1))
    copy_kernel(mIn, mOut, smem_layout).launch(
        grid=(1, 1, 1),
        block=(THREADS, 1, 1),
    )


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    mIn = from_dlpack(x, assumed_align=16)
    mOut = from_dlpack(y, assumed_align=16)

    compiled = cute.compile(run, mIn, mOut)
    compiled(mIn, mOut)
    torch.cuda.synchronize()

    ok = torch.equal(x, y)
    print(f"CuteDSL smoke: max|diff|={ (x-y).abs().max().item():.3e}  exact_match={ok}")
    print("TOOLCHAIN OK" if ok else "TOOLCHAIN FAIL")
    import sys; sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
