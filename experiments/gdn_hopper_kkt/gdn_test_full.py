"""2e: full kkt_inv_uw pipeline on tensor cores, multi-chunk/multi-head, vs FLA intra.

Replaces the naive-dot A (2b) with in-kernel-style wgmma KKT (K@K^T via split-K, contraction
dim 128 = two 64-wide accumulations through the 64x64x64 wgmma primitive). Then NS inverse (2c)
+ U/W (2d), per (chunk, head) with GVA k-head mapping. Assembles U,W [B,T,H,D] and cross-checks
against FLA chunk_gated_delta_rule_fwd_intra (the kkt_inv_uw counterpart).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_wgmma_probe import wgmma_gemm
from gdn_test_ns_inverse import ns_inverse_wgmma, mm

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra

DEV = "cuda"
BT = 64


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def kkt_wgmma(k):
    """k:[64,128] -> K@K^T [64,64] on tensor cores, split-K (contraction 128 = 2x64)."""
    kb = k.to(torch.bfloat16)
    return (wgmma_gemm(kb[:, :64].contiguous(), kb[:, :64].contiguous())
            + wgmma_gemm(kb[:, 64:].contiguous(), kb[:, 64:].contiguous()))


def kkt_inv_uw_one(k, v, beta, g_cs):
    """single chunk/head; g_cs = chunk-local-cumsum gate. Returns U,W [64,Dv],[64,Dk]."""
    KKt = kkt_wgmma(k)
    decay = torch.exp(g_cs[:, None] - g_cs[None, :])
    A = (KKt * decay * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)
    Ai = ns_inverse_wgmma(A, n_iter=5)
    betaV = beta[:, None] * v
    betaKg = beta[:, None] * torch.exp(g_cs)[:, None] * k
    U = torch.cat([mm(Ai, betaV[:, :64]), mm(Ai, betaV[:, 64:])], dim=1)
    W = torch.cat([mm(Ai, betaKg[:, :64]), mm(Ai, betaKg[:, 64:])], dim=1)
    return U, W


def main():
    torch.manual_seed(0)
    B, T, Hg, H, D = 1, 128, 4, 8, 128       # 2 chunks, GVA 2:1, head dim 128
    NT, rep = T // BT, H // Hg

    k = torch.randn(B, T, Hg, D, device=DEV, dtype=torch.bfloat16)
    k = k / k.float().norm(dim=-1, keepdim=True).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, device=DEV, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=DEV, dtype=torch.float32).clamp_min(0.1)
    g_raw = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=DEV, dtype=torch.float32))
    g_cs = chunk_local_cumsum(g_raw, chunk_size=BT)

    # FLA reference
    w_fla, u_fla, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cs, beta=beta, chunk_size=BT)

    # tensor-core pipeline, per (chunk, head)
    U = torch.zeros(B, T, H, D, device=DEV, dtype=torch.float32)
    W = torch.zeros(B, T, H, D, device=DEV, dtype=torch.float32)
    for c in range(NT):
        sl = slice(c * BT, (c + 1) * BT)
        for h in range(H):
            kh = k[0, sl, h // rep, :].float()            # GVA: k-head = h//rep
            vh = v[0, sl, h, :].float()
            bh = beta[0, sl, h]
            gh = g_cs[0, sl, h]
            Uh, Wh = kkt_inv_uw_one(kh, vh, bh, gh)
            U[0, sl, h, :] = Uh
            W[0, sl, h, :] = Wh

    cu, cw = cos(U, u_fla), cos(W, w_fla)
    print(f"[2e] full pipeline vs FLA intra (B={B},T={T},Hg={Hg},H={H}): U cosine={cu:.6f}  W cosine={cw:.6f}")
    ok = cu >= 0.999 and cw >= 0.999
    print(f"=== 2e {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
