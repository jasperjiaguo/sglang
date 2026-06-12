"""Honing-4 last stage: gridded U/W (Ai@(beta·V), Ai@(beta·exp(g)·K)) via wgmma vs FLA
recompute_w_u_fwd, at 35B shapes. Completes the 3-stage end-to-end comparison vs FLA intra."""
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
def _grid_uw(gAi, gK, gV, gBeta, gG, gU, gW):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    chunk = bid // H
    head = bid % H
    kh = head // REP
    r0 = chunk * BT
    mma = hh.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K,
        OperandMajorMode.K, cutlass.Float32, (1, 1, 1), (BT, BT), OperandSource.SMEM)
    lay = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT))
    layK = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, DK))
    sm = cutlass.utils.SmemAllocator()
    sAi = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sBt = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sK = sm.allocate_tensor(cutlass.BFloat16, layK.outer, byte_alignment=1024, swizzle=layK.inner)
    sV = sm.allocate_tensor(cutlass.BFloat16, layK.outer, byte_alignment=1024, swizzle=layK.inner)
    sZ = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT, 1)), byte_alignment=128)
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

    for half in cutlass.range_constexpr(2):
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tid
            n = idx // BT
            kk = idx % BT
            sBt[n, kk] = (sBeta[kk] * sV[kk, half * BT + n].to(cutlass.Float32)).to(cutlass.BFloat16)
        cute.arch.barrier()
        accU, tAi, tB = smu.partition_fragment_ABC(thr, (BT, BT, BT), sAi, sBt)
        smu.gemm(mma, accU, tAi, tB, zero_init=True)
        cute.autovec_copy(accU, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tid
            gU[r0 + idx // BT, head, half * BT + idx % BT] = sZ[idx // BT, idx % BT]
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tid
            n = idx // BT
            kk = idx % BT
            sBt[n, kk] = (sBeta[kk] * cute.math.exp(sG[kk]) * sK[kk, half * BT + n].to(cutlass.Float32)).to(cutlass.BFloat16)
        cute.arch.barrier()
        accW, tAi2, tB2 = smu.partition_fragment_ABC(thr, (BT, BT, BT), sAi, sBt)
        smu.gemm(mma, accW, tAi2, tB2, zero_init=True)
        cute.autovec_copy(accW, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tid
            gW[r0 + idx // BT, head, half * BT + idx % BT] = sZ[idx // BT, idx % BT]
        cute.arch.barrier()


@cute.jit
def _run(mAi, mK, mV, mB, mG, mU, mW):
    nb = cute.size(mAi, mode=[0])
    _grid_uw(mAi, mK, mV, mB, mG, mU, mW).launch(grid=(nb, 1, 1), block=(128, 1, 1))


_c = {}

def uw_grid(Ai, k, v, beta, g, T):
    U = torch.empty(T, H, DV, device=DEV, dtype=torch.float32)
    W = torch.empty(T, H, DK, device=DEV, dtype=torch.float32)
    args = [from_dlpack(Ai, assumed_align=16), from_dlpack(k, assumed_align=16), from_dlpack(v, assumed_align=16),
            from_dlpack(beta, assumed_align=16), from_dlpack(g, assumed_align=16),
            from_dlpack(U, assumed_align=16), from_dlpack(W, assumed_align=16)]
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
    print("Device:", torch.cuda.get_device_name(0), "| U/W vs FLA recompute_w_u")
    print("%7s | %9s %11s %8s" % ("T", "myUW(ms)", "flaRecompUW", "ratio"))
    for T in S.SEQLENS:
        NT = T // BT
        k = torch.randn(1, T, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
        v = torch.randn(1, T, H, DV, device=DEV, dtype=torch.bfloat16)
        beta = torch.rand(1, T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
        g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)
        A_raw = chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT)
        Ai_fla = solve_tril(A_raw)                       # [1,T,H,64]
        Ai_blocks = Ai_fla[0].reshape(NT, BT, H, BT).permute(0, 2, 1, 3).reshape(NT * H, BT, BT).contiguous()
        kc = k[0].contiguous(); vc = v[0].contiguous(); bc = beta[0].contiguous(); gc = g[0].contiguous()

        Ai_bf = Ai_fla.to(torch.bfloat16)
        U, W = uw_grid(Ai_blocks, kc, vc, bc, gc, T)
        w_fla, u_fla = recompute_w_u_fwd(k, v, beta, g, Ai_bf, None)
        cu, cw = cos(U, u_fla[0]), cos(W, w_fla[0])
        t_mine = time_fn(lambda: uw_grid(Ai_blocks, kc, vc, bc, gc, T))
        t_fla = time_fn(lambda: recompute_w_u_fwd(k, v, beta, g, Ai_bf, None))
        print("%7d | %9.4f %11.4f %7.2fx   (U cos=%.4f W cos=%.4f)" % (T, t_mine, t_fla, t_fla / t_mine, cu, cw))


if __name__ == "__main__":
    main()
