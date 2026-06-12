"""Honing-5d: U/W with MN-major B + pre-scaled Ai — eliminate the transposed sBt build.
U = (Ai*diag(beta)) @ V = Ais @ V; feed V DIRECTLY as B (b_leading_mode=MN, no transpose).
Build only the cheap 64x64 scaled-Ai. vs FLA recompute."""
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

import gdn_shapes as S
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.fla.solve_tril import solve_tril
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd

BT, DK, DV = 64, 128, 128
HG, H = S.NUM_K_HEADS, S.NUM_V_HEADS
REP = H // HG
DEV = "cuda"


@cute.kernel
def _grid_uw4(gAi, gK, gV, gBeta, gG, gU, gW):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    chunk = bid // H
    head = bid % H
    kh = head // REP
    r0 = chunk * BT
    # A (Ais) K-major, B (V/K) MN-major -> acc = A @ B (standard, no transpose)
    mma = hh.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K,
        OperandMajorMode.MN, cutlass.Float32, (1, 1, 1), (BT, DV), OperandSource.SMEM)
    layA = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT))
    layB = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, DK))
    sm = cutlass.utils.SmemAllocator()
    sAi = sm.allocate_tensor(cutlass.BFloat16, layA.outer, byte_alignment=1024, swizzle=layA.inner)
    sAs = sm.allocate_tensor(cutlass.BFloat16, layA.outer, byte_alignment=1024, swizzle=layA.inner)
    sK = sm.allocate_tensor(cutlass.BFloat16, layB.outer, byte_alignment=1024, swizzle=layB.inner)
    sV = sm.allocate_tensor(cutlass.BFloat16, layB.outer, byte_alignment=1024, swizzle=layB.inner)
    sBeta = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)
    sG = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)

    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        sAi[idx // BT, idx % BT] = gAi[bid, idx // BT, idx % BT].to(cutlass.BFloat16)
    for it in cutlass.range_constexpr(BT * DK // 128):
        idx = it * 128 + tid
        sK[idx // DK, idx % DK] = gK[r0 + idx // DK, kh, idx % DK]
        sV[idx // DV, idx % DV] = gV[r0 + idx // DV, head, idx % DV]
    if tid < BT:
        sBeta[tid] = gBeta[r0 + tid, head]
        sG[tid] = gG[r0 + tid, head]
    cute.arch.barrier()
    thr = mma.get_slice(tid)

    # U = Ais @ V,  Ais[i,kk] = Ai[i,kk]*beta[kk]  (cheap 64x64 build, V fed directly)
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        sAs[idx // BT, idx % BT] = (sAi[idx // BT, idx % BT].to(cutlass.Float32) * sBeta[idx % BT]).to(cutlass.BFloat16)
    cute.arch.barrier()
    accU, tA, tV = smu.partition_fragment_ABC(thr, (BT, DV, BT), sAs, sV)
    smu.gemm(mma, accU, tA, tV, zero_init=True)
    cute.autovec_copy(accU, thr.partition_C(gU[chunk, None, head, None]))
    cute.arch.barrier()

    # W = Aig @ K,  Aig[i,kk] = Ai[i,kk]*beta[kk]*exp(g[kk])
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tid
        sAs[idx // BT, idx % BT] = (sAi[idx // BT, idx % BT].to(cutlass.Float32) * sBeta[idx % BT] * cute.math.exp(sG[idx % BT])).to(cutlass.BFloat16)
    cute.arch.barrier()
    accW, tA2, tK = smu.partition_fragment_ABC(thr, (BT, DK, BT), sAs, sK)
    smu.gemm(mma, accW, tA2, tK, zero_init=True)
    cute.autovec_copy(accW, thr.partition_C(gW[chunk, None, head, None]))


@cute.jit
def _run(mAi, mK, mV, mB, mG, mU, mW):
    nb = cute.size(mAi, mode=[0])
    _grid_uw4(mAi, mK, mV, mB, mG, mU, mW).launch(grid=(nb, 1, 1), block=(128, 1, 1))


_c = {}

def uw4(Ai, k, v, beta, g, T):
    NT = T // BT
    U = torch.empty(T, H, DV, device=DEV, dtype=torch.float32)
    W = torch.empty(T, H, DK, device=DEV, dtype=torch.float32)
    args = [from_dlpack(Ai, assumed_align=16), from_dlpack(k, assumed_align=16), from_dlpack(v, assumed_align=16),
            from_dlpack(beta, assumed_align=16), from_dlpack(g, assumed_align=16),
            from_dlpack(U.view(NT, BT, H, DV), assumed_align=16), from_dlpack(W.view(NT, BT, H, DK), assumed_align=16)]
    if T not in _c:
        _c[T] = cute.compile(_run, *args)
    _c[T](*args)
    torch.cuda.synchronize()
    return U, W


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    print("Device:", torch.cuda.get_device_name(0), "| U/W MN-major+prescaled-Ai vs FLA recompute")
    print("%7s | %9s %11s %8s" % ("T", "myUW(ms)", "flaRecompUW", "ratio"))
    for T in S.SEQLENS:
        NT = T // BT
        k = torch.randn(1, T, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
        v = torch.randn(1, T, H, DV, device=DEV, dtype=torch.bfloat16)
        beta = torch.rand(1, T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
        g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)
        A_raw = chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT)
        Ai_fla = solve_tril(A_raw)
        Ai_blocks = Ai_fla[0].reshape(NT, BT, H, BT).permute(0, 2, 1, 3).reshape(NT * H, BT, BT).contiguous()
        kc, vc, bc, gc = k[0].contiguous(), v[0].contiguous(), beta[0].contiguous(), g[0].contiguous()
        U, W = uw4(Ai_blocks, kc, vc, bc, gc, T)
        Ai0 = Ai_blocks[0].float()
        Uref0 = Ai0 @ (bc[:BT, 0, None].float() * vc[:BT, 0, :].float())
        Wref0 = Ai0 @ (bc[:BT, 0, None].float() * torch.exp(gc[:BT, 0, None].float()) * kc[:BT, 0, :].float())
        cU, cW = cos(U[:BT, 0, :], Uref0), cos(W[:BT, 0, :], Wref0)
        t_mine = time_fn(lambda: uw4(Ai_blocks, kc, vc, bc, gc, T))
        t_fla = time_fn(lambda: recompute_w_u_fwd(k, v, beta, g, Ai_fla.to(torch.bfloat16), None))
        print("%7d | %9.4f %11.4f %7.2fx   (U cos=%.4f W cos=%.4f)" % (T, t_mine, t_fla, t_fla / t_mine, cU, cW))


if __name__ == "__main__":
    main()
