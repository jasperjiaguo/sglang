"""Honing-6: definitive END-TO-END kkt_inv_uw — sum my 3 best kernels (KKT + blocked inverse +
occupancy-opt U/W) vs FLA's 3 stages (kkt + solve_tril + recompute_w_u), one run, {2K,8K,32K}."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_shapes as S
from gdn_bench_stages import my_kkt, BT, HG, H
from gdn_bench_inv import blocked_inv_grid
from gdn_bench_uw3 import uw3
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.fla.solve_tril import solve_tril
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra

DK = DV = 128
DEV = "cuda"


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    print("Device:", torch.cuda.get_device_name(0), "| END-TO-END kkt_inv_uw: mine (3 kernels) vs FLA (3 + fused intra)")
    print("%7s | %7s %7s %7s %8s | %7s %8s | %7s" % ("T","myKKT","myINV","myUW","myTOT","flaTOT","intra","ratio"))
    for T in S.SEQLENS:
        NT = T // BT
        k = torch.randn(1, T, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
        v = torch.randn(1, T, H, DV, device=DEV, dtype=torch.bfloat16)
        beta = torch.rand(1, T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
        g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)
        kc, vc, bc, gc = k[0].contiguous(), v[0].contiguous(), beta[0].contiguous(), g[0].contiguous()
        A_raw = chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT)
        Ai_fla = solve_tril(A_raw)
        Ai_blocks = Ai_fla[0].reshape(NT, BT, H, BT).permute(0, 2, 1, 3).reshape(NT * H, BT, BT).contiguous()
        A_blocks = A_raw[0].reshape(NT, BT, H, BT).permute(0, 2, 1, 3).reshape(NT * H, BT, BT).contiguous()
        Ai_bf = Ai_fla.to(torch.bfloat16)

        # mine
        t_kkt = time_fn(lambda: my_kkt(kc, bc, gc, NT))
        t_inv = time_fn(lambda: blocked_inv_grid(A_blocks))
        t_uw = time_fn(lambda: uw3(Ai_blocks, kc, vc, bc, gc, T))
        mine = t_kkt + t_inv + t_uw
        # FLA (3 separate kernels)
        t_fk = time_fn(lambda: chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT))
        t_fs = time_fn(lambda: solve_tril(A_raw))
        t_fu = time_fn(lambda: recompute_w_u_fwd(k, v, beta, g, Ai_bf, None))
        fla = t_fk + t_fs + t_fu
        # FLA fused intra (the single 3-kernel pipeline call)
        t_intra = time_fn(lambda: chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, chunk_size=BT))

        print("%7d | %7.3f %7.3f %7.3f %8.3f | %7.3f %8.3f | %6.2fx" % (
            T, t_kkt, t_inv, t_uw, mine, fla, t_intra, fla / mine))


if __name__ == "__main__":
    main()
