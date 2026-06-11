"""2d: U/W via wgmma tensor cores.

U = Ai @ (beta·V),  W = Ai @ (beta·exp(g)·K),  with Ai from the tensor-core NS inverse (2c).
N=128 (V/K width) handled as two 64-wide halves through the proven 64x64x64 wgmma primitive.
Full chain (wgmma inverse -> wgmma U/W) validated vs torch oracle (exact inverse).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_wgmma_probe import wgmma_gemm
from gdn_test_ns_inverse import ns_inverse_wgmma, mm

BT, DK, DV = 64, 128, 64 + 64
DEV = "cuda"


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def compute_UW_wgmma(A, V, k, beta, g):
    Ai = ns_inverse_wgmma(A, n_iter=5)                       # [64,64] f32, tensor-core
    betaV = beta[:, None] * V                                # [64,128]
    betaKg = beta[:, None] * torch.exp(g)[:, None] * k       # [64,128]
    U = torch.cat([mm(Ai, betaV[:, :64]), mm(Ai, betaV[:, 64:])], dim=1)
    W = torch.cat([mm(Ai, betaKg[:, :64]), mm(Ai, betaKg[:, 64:])], dim=1)
    return U, W


def main():
    torch.manual_seed(0)
    k = torch.randn(BT, DK, device=DEV); k = k / k.norm(dim=-1, keepdim=True)
    V = torch.randn(BT, DV, device=DEV)
    beta = torch.rand(BT, device=DEV).clamp_min(0.1)
    g = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(BT, device=DEV)), 0)
    KKt = k @ k.T
    decay = torch.exp(g[:, None] - g[None, :])
    A = (KKt * decay * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)

    U_got, W_got = compute_UW_wgmma(A, V, k, beta, g)

    Ai_ref = torch.linalg.inv(torch.eye(BT, device=DEV) + A)
    U_ref = Ai_ref @ (beta[:, None] * V)
    W_ref = Ai_ref @ (beta[:, None] * torch.exp(g)[:, None] * k)

    cu, cw = cos(U_got, U_ref), cos(W_got, W_ref)
    print(f"[2d] U: cosine={cu:.6f}  W: cosine={cw:.6f}")
    ok = cu >= 0.999 and cw >= 0.999
    print(f"=== 2d {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
