"""P1c-iv: GO/NO-GO benchmark — gridded fused kkt_inv_uw kernel vs FLA intra @ 35B shapes.

Grids the fused kernel over (chunk, head): each CTA does the full KKT->A->NS-inverse->U/W for its
64-token chunk / head, smem-resident. Validates vs FLA once, then times both at {2K,8K,32K}.
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

import gdn_shapes as S
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra

BT, DK, DV = 64, 128, 128
HG, H = S.NUM_K_HEADS, S.NUM_V_HEADS    # 16, 32
REP = H // HG                            # 2
NITER = 3                                # realistic conditioning -> 3 exact
DEV = "cuda"


@cute.kernel
def _grid(gK, gV, gBeta, gG, gU, gW):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    chunk = bid // H
    head = bid % H
    kh = head // REP
    r0 = chunk * BT

    mma = hh.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float32, (1, 1, 1), (BT, BT), OperandSource.SMEM)
    lay = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT))
    layK = smu.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (BT, DK))
    plain = cute.make_layout((BT, BT), stride=(BT, 1))
    sm = cutlass.utils.SmemAllocator()
    sK = sm.allocate_tensor(cutlass.BFloat16, layK.outer, byte_alignment=1024, swizzle=layK.inner)
    sV = sm.allocate_tensor(cutlass.BFloat16, layK.outer, byte_alignment=1024, swizzle=layK.inner)
    sM = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sX = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sXt = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sY = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sBt = sm.allocate_tensor(cutlass.BFloat16, lay.outer, byte_alignment=1024, swizzle=lay.inner)
    sZ = sm.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    sA = sm.allocate_tensor(cutlass.Float32, plain, byte_alignment=128)
    sBeta = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)
    sG = sm.allocate_tensor(cutlass.Float32, cute.make_layout((BT,), stride=(1,)), byte_alignment=128)

    for it in cutlass.range_constexpr(BT * DK // 128):
        idx = it * 128 + tidx
        rr = idx // DK
        dd = idx % DK
        sK[rr, dd] = gK[r0 + rr, kh, dd]
        sV[rr, dd] = gV[r0 + rr, head, dd]
    if tidx < BT:
        sBeta[tidx] = gBeta[r0 + tidx, head]
        sG[tidx] = gG[r0 + tidx, head]
    cute.arch.barrier()

    thr = mma.get_slice(tidx)
    accA, tK1, tK2 = smu.partition_fragment_ABC(thr, (BT, BT, DK), sK, sK)
    smu.gemm(mma, accA, tK1, tK2, zero_init=True)
    cute.autovec_copy(accA, thr.partition_C(sA))
    cute.arch.barrier()
    for it in cutlass.range_constexpr(BT * BT // 128):
        idx = it * 128 + tidx
        i = idx // BT
        j = idx % BT
        a = sA[i, j] * sBeta[i] * cute.math.exp(sG[i] - sG[j])
        if i <= j:
            a = cutlass.Float32(0.0)
        m = a
        x = cutlass.Float32(0.0) - a
        if i == j:
            m = m + cutlass.Float32(1.0)
            x = x + cutlass.Float32(1.0)
        sM[i, j] = m.to(cutlass.BFloat16)
        sX[i, j] = x.to(cutlass.BFloat16)
        sA[i, j] = x
    cute.arch.barrier()
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
            xn = cutlass.Float32(2.0) * sA[i, j] - sZ[i, j]
            sA[i, j] = xn
            sX[i, j] = xn.to(cutlass.BFloat16)
        cute.arch.barrier()
    for half in cutlass.range_constexpr(2):
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            n = idx // BT
            kk = idx % BT
            sBt[n, kk] = (sBeta[kk] * sV[kk, half * BT + n].to(cutlass.Float32)).to(cutlass.BFloat16)
        cute.arch.barrier()
        aU, tAi, tB = smu.partition_fragment_ABC(thr, (BT, BT, BT), sX, sBt)
        smu.gemm(mma, aU, tAi, tB, zero_init=True)
        cute.autovec_copy(aU, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            gU[r0 + idx // BT, head, half * BT + idx % BT] = sZ[idx // BT, idx % BT]
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            n = idx // BT
            kk = idx % BT
            sBt[n, kk] = (sBeta[kk] * cute.math.exp(sG[kk]) * sK[kk, half * BT + n].to(cutlass.Float32)).to(cutlass.BFloat16)
        cute.arch.barrier()
        aW, tAi2, tB2 = smu.partition_fragment_ABC(thr, (BT, BT, BT), sX, sBt)
        smu.gemm(mma, aW, tAi2, tB2, zero_init=True)
        cute.autovec_copy(aW, thr.partition_C(sZ))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(BT * BT // 128):
            idx = it * 128 + tidx
            gW[r0 + idx // BT, head, half * BT + idx % BT] = sZ[idx // BT, idx % BT]
        cute.arch.barrier()


@cute.jit
def _run(mK, mV, mB, mG, mU, mW):
    nblk = cute.size(mU, mode=[0]) // BT * H
    _grid(mK, mV, mB, mG, mU, mW).launch(grid=(nblk, 1, 1), block=(128, 1, 1))


_c = {}

def fused(k, v, beta, g, T):
    U = torch.empty(T, H, DV, device=DEV, dtype=torch.float32)
    W = torch.empty(T, H, DK, device=DEV, dtype=torch.float32)
    args = [from_dlpack(k, assumed_align=16), from_dlpack(v, assumed_align=16),
            from_dlpack(beta, assumed_align=16), from_dlpack(g, assumed_align=16),
            from_dlpack(U, assumed_align=16), from_dlpack(W, assumed_align=16)]
    key = T
    if key not in _c:
        _c[key] = cute.compile(_run, *args)
    _c[key](*args)
    torch.cuda.synchronize()
    return U, W


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    print("Device:", torch.cuda.get_device_name(0), "| Hg=%d H=%d d=%d NITER=%d" % (HG, H, DK, NITER))
    # correctness at small T
    torch.manual_seed(0)
    Tc = 128
    k = torch.randn(Tc, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
    v = torch.randn(Tc, H, DV, device=DEV, dtype=torch.bfloat16)
    beta = torch.rand(Tc, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
    g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, Tc, H, device=DEV, dtype=torch.float32)), chunk_size=BT)[0]
    U, W = fused(k.contiguous(), v.contiguous(), beta.contiguous(), g.contiguous(), Tc)
    w_fla, u_fla, _ = chunk_gated_delta_rule_fwd_intra(k=k[None], v=v[None], g=g[None], beta=beta[None], chunk_size=BT)
    cu, cw = cos(U, u_fla[0]), cos(W, w_fla[0])
    print(f"correctness (gridded, T={Tc}): U cosine={cu:.6f} W cosine={cw:.6f} -> {'OK' if cu>=0.999 and cw>=0.999 else 'FAIL'}")
    if not (cu >= 0.999 and cw >= 0.999):
        print("=== P1c-iv: correctness FAIL, aborting bench ==="); sys.exit(1)

    print("\n%8s %14s %14s %10s" % ("T", "fused(ms)", "FLA intra(ms)", "speedup"))
    results = []
    for T in S.SEQLENS:
        k = torch.randn(T, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
        v = torch.randn(T, H, DV, device=DEV, dtype=torch.bfloat16)
        beta = torch.rand(T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
        g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)[0]
        kc, vc, bc, gc = k.contiguous(), v.contiguous(), beta.contiguous(), g.contiguous()
        kb, vb, gb, bb = k[None], v[None], g[None], beta[None]
        t_fused = time_fn(lambda: fused(kc, vc, bc, gc, T))
        t_fla = time_fn(lambda: chunk_gated_delta_rule_fwd_intra(k=kb, v=vb, g=gb, beta=bb, chunk_size=BT))
        sp = t_fla / t_fused
        results.append((T, t_fused, t_fla, sp))
        print("%8d %14.4f %14.4f %9.2fx" % (T, t_fused, t_fla, sp))

    allgo = all(sp > 1.0 for _, _, _, sp in results)
    print(f"\n=== VERDICT: {'GO' if allgo else 'NO-GO'} (>1.0x across all seqlens: {allgo}) ===")


if __name__ == "__main__":
    main()
