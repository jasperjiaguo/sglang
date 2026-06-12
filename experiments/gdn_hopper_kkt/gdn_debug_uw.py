"""Localize the gridded U/W bug: compare uw_grid output to a TORCH reference computed from the
SAME Ai_blocks + inputs. If they match -> kernel OK (FLA-call was the issue). If not -> kernel bug."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_shapes as S
from gdn_bench_uw import uw_grid, BT, DK, DV, H, HG, REP
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.fla.solve_tril import solve_tril

DEV = "cuda"


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    torch.manual_seed(0)
    T = 128
    NT = T // BT
    k = torch.randn(1, T, HG, DK, device=DEV, dtype=torch.bfloat16); k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
    v = torch.randn(1, T, H, DV, device=DEV, dtype=torch.bfloat16)
    beta = torch.rand(1, T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
    g = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32)), chunk_size=BT)
    A_raw = chunk_scaled_dot_kkt_fwd(k, beta, g, chunk_size=BT)
    Ai_fla = solve_tril(A_raw)
    Ai_blocks = Ai_fla[0].reshape(NT, BT, H, BT).permute(0, 2, 1, 3).reshape(NT * H, BT, BT).contiguous()
    kc, vc, bc, gc = k[0].contiguous(), v[0].contiguous(), beta[0].contiguous(), g[0].contiguous()

    U, W = uw_grid(Ai_blocks, kc, vc, bc, gc, T)

    # torch reference from the SAME Ai_blocks
    U_ref = torch.zeros(T, H, DV, device=DEV)
    W_ref = torch.zeros(T, H, DK, device=DEV)
    for c in range(NT):
        for h in range(H):
            Ai = Ai_blocks[c * H + h].float()                      # [64,64]
            kh = h // REP
            bV = bc[c * BT:(c + 1) * BT, h, None].float() * vc[c * BT:(c + 1) * BT, h, :].float()
            bKg = (bc[c * BT:(c + 1) * BT, h, None].float()
                   * torch.exp(gc[c * BT:(c + 1) * BT, h, None].float())
                   * kc[c * BT:(c + 1) * BT, kh, :].float())
            U_ref[c * BT:(c + 1) * BT, h, :] = Ai @ bV
            W_ref[c * BT:(c + 1) * BT, h, :] = Ai @ bKg

    print(f"my U/W vs TORCH ref (same Ai): U cosine={cos(U, U_ref):.6f}  W cosine={cos(W, W_ref):.6f}")
    print(f"  U[0,0,:4] mine ={U[0,0,:4].tolist()}")
    print(f"  U[0,0,:4] torch={U_ref[0,0,:4].tolist()}")
    print(f"  any nan in mine U? {bool(torch.isnan(U).any())}  W? {bool(torch.isnan(W).any())}")


if __name__ == "__main__":
    main()
