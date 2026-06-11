"""Per-stage breakdown: where does the 3x live? Time my gridded KKT->A and inverse
stages vs FLA's chunk_scaled_dot_kkt_fwd and solve_tril, at 35B shapes."""
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

BT, DK = 64, 128
HG, H = S.NUM_K_HEADS, S.NUM_V_HEADS
REP = H // HG
NITER = 3
DEV = "cuda"


@cute.kernel
def _grid_kkt(gK, gBeta, gG, gA):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    chunk = bid // H
    head = bid % H
    kh = head // REP
    r0 = chunk * BT
    mma = hh.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K,
        OperandMajorMode.K, cutlass.Float32, (1, 1, 1), (BT, BT), OperandSource.SMEM)
    layK = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, DK))
    plain = cute.make_layout((BT, BT), stride=(BT, 1))
    sm = cutlass.utils.SmemAllocator()
    sK = sm.allocate_tensor(cutlass.BFloat16, layK.outer, byte_alignment=1024, swizzle=layK.inner)
    sA = sm.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    sBeta = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)
    sG = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)
    for it in cutlass.range_constexpr(BT * DK // 128):
        idx = it * 128 + tidx
        sK[idx // DK, idx % DK] = gK[r0 + idx // DK, kh, idx % DK]
    if tidx < BT:
        sBeta[tidx] = gBeta[r0 + tidx, head]
        sG[tidx] = gG[r0 + tidx, head]
    cute.arch.barrier()
    thr = mma.get_slice(tidx)
    accA, t1, t2 = smu.partition_fragment_ABC(thr, (BT, BT, DK), sK, sK)
    smu.gemm(mma, accA, t1, t2, zero_init=True)
    cute.autovec_copy(accA, thr.partition_C(sA))
    cute.arch.barrier()
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        i = idx // BT
        j = idx % BT
        a = sA[i, j] * sBeta[i] * cute.math.exp(sG[i] - sG[j])
        if i <= j:
            a = cutlass.Float32(0.0)
        gA[chunk, head, i, j] = a


@cute.kernel
def _grid_inv(gA, gAi):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    chunk = bid // H
    head = bid % H
    mma = hh.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K,
        OperandMajorMode.K, cutlass.Float32, (1, 1, 1), (BT, BT), OperandSource.SMEM)
    lay = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT))
    plain = cute.make_layout((BT, BT), stride=(BT, 1))
    sm = cutlass.utils.SmemAllocator()
    sM = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sX = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sXt = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sY = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sZ = sm.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    sXf = sm.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        i = idx // BT
        j = idx % BT
        a = gA[chunk, head, i, j]
        m = a
        x = cutlass.Float32(0.0) - a
        if i == j:
            m = m + cutlass.Float32(1.0)
            x = x + cutlass.Float32(1.0)
        sM[i, j] = m.to(cutlass.BFloat16)
        sX[i, j] = x.to(cutlass.BFloat16)
        sXf[i, j] = x
    cute.arch.barrier()
    thr = mma.get_slice(tidx)
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
            sXt[idx % BT, idx // BT] = sY[idx // BT, idx % BT]
        cute.arch.barrier()
        aZ, tX, tYt = smu.partition_fragment_ABC(thr, (BT, BT, BT), sX, sXt)
        smu.gemm(mma, aZ, tX, tYt, zero_init=True)
        cute.autovec_copy(aZ, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            i = idx // BT
            j = idx % BT
            xn = cutlass.Float32(2.0) * sXf[i, j] - sZ[i, j]
            sXf[i, j] = xn
            sX[i, j] = xn.to(cutlass.BFloat16)
        cute.arch.barrier()
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        gAi[chunk, head, idx // BT, idx % BT] = sXf[idx // BT, idx % BT]


@cute.jit
def _run_kkt(mK, mB, mG, mA):
    nb = cute.size(mA, mode=[0]) * H
    _grid_kkt(mK, mB, mG, mA).launch(grid=(nb, 1, 1), block=(128, 1, 1))

@cute.jit
def _run_inv(mA, mAi):
    nb = cute.size(mA, mode=[0]) * H
    _grid_inv(mA, mAi).launch(grid=(nb, 1, 1), block=(128, 1, 1))


_ck, _ci = {}, {}

def my_kkt(k, beta, g, NT):
    A = torch.empty(NT, H, BT, BT, device=DEV, dtype=torch.float32)
    args = [from_dlpack(k, assumed_align=16), from_dlpack(beta, assumed_align=16),
            from_dlpack(g, assumed_align=16), from_dlpack(A, assumed_align=16)]
    if NT not in _ck: _ck[NT] = cute.compile(_run_kkt, *args)
    _ck[NT](*args); torch.cuda.synchronize(); return A

def my_inv(A, NT):
    Ai = torch.empty_like(A)
    args = [from_dlpack(A, assumed_align=16), from_dlpack(Ai, assumed_align=16)]
    if NT not in _ci: _ci[NT] = cute.compile(_run_inv, *args)
    _ci[NT](*args); torch.cuda.synchronize(); return Ai


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    print("Device:", torch.cuda.get_device_name(0), "| per-stage breakdown, NITER=%d" % NITER)
    print("%7s | %9s %9s %7s | %9s %9s %7s" % ("T", "myKKT", "flaKKT", "ratio", "myINV", "flaSolve", "ratio"))
    for T in S.SEQLENS:
        NT = T // BT
        k = torch.randn(T, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
        beta = torch.rand(T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
        g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)[0]
        kc, bc, gc = k.contiguous(), beta.contiguous(), g.contiguous()
        # FLA stage inputs (B=1)
        kb, bb, gb = k[None], beta[None], g[None]

        t_my_kkt = time_fn(lambda: my_kkt(kc, bc, gc, NT))
        t_fla_kkt = time_fn(lambda: chunk_scaled_dot_kkt_fwd(kb, bb, gb, chunk_size=BT))
        A_raw = chunk_scaled_dot_kkt_fwd(kb, bb, gb, chunk_size=BT)
        myA = my_kkt(kc, bc, gc, NT)
        t_my_inv = time_fn(lambda: my_inv(myA, NT))
        t_fla_inv = time_fn(lambda: solve_tril(A_raw))

        print("%7d | %9.4f %9.4f %6.2fx | %9.4f %9.4f %6.2fx" % (
            T, t_my_kkt, t_fla_kkt, t_fla_kkt / t_my_kkt,
            t_my_inv, t_fla_inv, t_fla_inv / t_my_inv))


if __name__ == "__main__":
    main()
