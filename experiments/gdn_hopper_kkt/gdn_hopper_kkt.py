"""Standalone Hopper (SM90) kkt_inv_uw kernel — built incrementally via TDD milestones.

Milestone status:
  2a: load K tile [BT,DK] bf16 through smem, store back (identity plumbing)   <-- current
  2b: K @ K^T -> A (+ decay*beta*strict-tril)
  2c: blocked Newton-Schulz inverse (4x 16x16 diag blocks, 3 rounds + back-sub)
  2d: U = Ai@(beta*V),  W = Ai@(beta*exp(g)*K)
  2e: full multi-chunk / multi-head

Each public entry returns torch tensors so test_kkt.py can compare against gdn_torch_ref.
Single chunk (BT=64), single head for 2a-2d; multi extended at 2e.
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

BT = 64    # chunk
DK = 128   # head_k_dim
DV = 128   # head_v_dim


# ----------------------------------------------------------------------------
# Milestone 2a: K-tile plumbing — load [BT,DK] bf16 through smem, store back.
# ----------------------------------------------------------------------------
@cute.kernel
def _k_identity_kernel(gK: cute.Tensor, gOut: cute.Tensor, smem_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, smem_layout, 128)

    # 128 threads, tile is [64,128]=8192 elems -> 64 elems/thread, row-major flat
    n = BT * DK
    nthreads = 128
    for it in cutlass.range_constexpr(n // nthreads):
        idx = it * nthreads + tidx
        r = idx // DK
        c = idx % DK
        sK[r, c] = gK[r, c]
    cute.arch.barrier()
    for it in cutlass.range_constexpr(n // nthreads):
        idx = it * nthreads + tidx
        r = idx // DK
        c = idx % DK
        gOut[r, c] = sK[r, c]


@cute.jit
def _run_k_identity(mK: cute.Tensor, mOut: cute.Tensor):
    smem_layout = cute.make_layout((BT, DK), stride=(DK, 1))
    _k_identity_kernel(mK, mOut, smem_layout).launch(grid=(1, 1, 1), block=(128, 1, 1))


_compiled = {}

def load_k_identity(k_tile: torch.Tensor) -> torch.Tensor:
    """k_tile: [BT, DK] bf16 cuda -> identity copy through smem."""
    assert k_tile.shape == (BT, DK) and k_tile.dtype == torch.bfloat16
    out = torch.empty_like(k_tile)
    mK = from_dlpack(k_tile, assumed_align=16)
    mOut = from_dlpack(out, assumed_align=16)
    key = "2a"
    if key not in _compiled:
        _compiled[key] = cute.compile(_run_k_identity, mK, mOut)
    _compiled[key](mK, mOut)
    torch.cuda.synchronize()
    return out


# ----------------------------------------------------------------------------
# Milestone 2b: A = strictLower( beta_i * <k_i,k_j> * exp(g_i - g_j) )
# Correctness-first: naive per-element dot (tensor-core swap is a honing step).
# ----------------------------------------------------------------------------
@cute.kernel
def _kkt_A_kernel(gK: cute.Tensor, gBeta: cute.Tensor, gG: cute.Tensor,
                  gA: cute.Tensor, smem_k: cute.Layout, smem_v: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.Float32, smem_k, 128)     # [BT,DK] f32
    sBeta = smem.allocate_tensor(cutlass.Float32, smem_v, 128)  # [BT]
    sG = smem.allocate_tensor(cutlass.Float32, smem_v, 128)     # [BT]

    # load K -> f32 smem
    for it in cutlass.range_constexpr(BT * DK // 128):
        idx = it * 128 + tidx
        sK[idx // DK, idx % DK] = gK[idx // DK, idx % DK].to(cutlass.Float32)
    # load beta, g (BT=64 <= 128 threads)
    if tidx < BT:
        sBeta[tidx] = gBeta[tidx]
        sG[tidx] = gG[tidx]
    cute.arch.barrier()

    # A[i,j], 64*64=4096 elems / 128 threads = 32 each
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        i = idx // BT
        j = idx % BT
        if i > j:
            acc = cutlass.Float32(0.0)
            for d in cutlass.range_constexpr(DK):
                acc += sK[i, d] * sK[j, d]
            gA[i, j] = sBeta[i] * acc * cute.math.exp(sG[i] - sG[j])
        else:
            gA[i, j] = cutlass.Float32(0.0)


@cute.jit
def _run_kkt_A(mK: cute.Tensor, mBeta: cute.Tensor, mG: cute.Tensor, mA: cute.Tensor):
    smem_k = cute.make_layout((BT, DK), stride=(DK, 1))
    smem_v = cute.make_layout((BT,), stride=(1,))
    _kkt_A_kernel(mK, mBeta, mG, mA, smem_k, smem_v).launch(grid=(1, 1, 1), block=(128, 1, 1))


def compute_A(k_tile, beta, g_cs):
    """k:[BT,DK] bf16, beta:[BT] f32, g_cs:[BT] f32 -> A:[BT,BT] f32 (strict lower)."""
    A = torch.empty(BT, BT, device=k_tile.device, dtype=torch.float32)
    mK = from_dlpack(k_tile, assumed_align=16)
    mB = from_dlpack(beta, assumed_align=16)
    mG = from_dlpack(g_cs, assumed_align=16)
    mA = from_dlpack(A, assumed_align=16)
    key = "2b"
    if key not in _compiled:
        _compiled[key] = cute.compile(_run_kkt_A, mK, mB, mG, mA)
    _compiled[key](mK, mB, mG, mA)
    torch.cuda.synchronize()
    return A
