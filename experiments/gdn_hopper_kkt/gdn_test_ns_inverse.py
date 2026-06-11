"""2c correctness: tensor-core Newton-Schulz inverse using the proven wgmma GEMM primitive.

Drives the NS iteration X <- X(2I - MX) = 2X - X@M@X with EVERY matmul on Hopper tensor
cores (wgmma_gemm). Validates Ai = (I+A)^-1 vs torch. Proves the TC inverse is numerically
correct before fusing into one kernel (the honing step).

wgmma_gemm(P, Q) computes P @ Q^T (K-contraction), so a normal matmul P@Q = wgmma_gemm(P, Q^T).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_wgmma_probe import wgmma_gemm

BT = 64
DEV = "cuda"


def mm(P, Q):
    """real matmul P@Q via the wgmma primitive (which does A@B^T)."""
    return wgmma_gemm(P.to(torch.bfloat16).contiguous(),
                      Q.t().to(torch.bfloat16).contiguous())  # P @ (Q^T)^T = P@Q


def ns_inverse_wgmma(A, n_iter=5):
    """A: [BT,BT] f32 strictly-lower -> Ai = (I+A)^-1 via tensor-core Newton-Schulz."""
    I = torch.eye(BT, device=A.device, dtype=torch.float32)
    M = I + A
    X = I - A                      # init: order-2 approximation
    for _ in range(n_iter):
        Y = mm(M, X)               # M @ X     (f32 acc)
        Z = mm(X, Y)               # X @ M @ X
        X = 2.0 * X - Z
    return X


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    torch.manual_seed(0)
    # realistic strictly-lower A from L2-normalized K (well-conditioned, like real GDN)
    k = torch.randn(BT, 128, device=DEV)
    k = k / k.norm(dim=-1, keepdim=True)
    beta = torch.rand(BT, device=DEV).clamp_min(0.1)
    g = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(BT, device=DEV)), 0)
    KKt = k @ k.T
    decay = torch.exp(g[:, None] - g[None, :])
    A = (KKt * decay * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)

    Ai_ref = torch.linalg.inv(torch.eye(BT, device=DEV) + A)
    for nit in [3, 4, 5, 6]:
        Ai = ns_inverse_wgmma(A, n_iter=nit)
        c = cos(Ai, Ai_ref)
        # also check it's a real inverse: (I+A)@Ai ~ I
        resid = float(((torch.eye(BT, device=DEV) + A) @ Ai - torch.eye(BT, device=DEV)).abs().max())
        print(f"  n_iter={nit}: cosine(Ai, exact)={c:.6f}  max|(I+A)Ai - I|={resid:.3e}")

    Ai5 = ns_inverse_wgmma(A, 5)
    c5 = cos(Ai5, Ai_ref)
    ok = c5 >= 0.999
    print(f"[2c] tensor-core NS inverse (5 iters): cosine={c5:.6f} -> {'PASS' if ok else 'FAIL'}")
    print(f"=== 2c {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
