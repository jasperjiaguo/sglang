"""Honing-3b-2 de-risk: torch sim of the bf16 BLOCKED inverse algorithm (the CUDA back-sub oracle).
Confirms 16x16-block NS diagonals + bf16 block forward-substitution reaches cosine>=0.999 vs exact,
and that bf16 back-sub accumulation is accurate enough. If green, the CUDA back-sub is pure engineering."""
import torch

B, BT, NB = 16, 64, 4
DEV = "cuda"


def ns_inv_bf16(M, n_iter=3):
    """(I+A)^-1 for a 16x16 block; M = the full block I+A. Newton-Schulz in bf16-rounded matmuls."""
    I = torch.eye(B, device=DEV, dtype=torch.float32)
    X = 2 * I - M
    for _ in range(n_iter):
        # emulate bf16 matmul rounding (inputs cast to bf16, accumulate f32)
        MX = (M.bfloat16().float() @ X.bfloat16().float())
        X = X @ (2 * I - MX).bfloat16().float()
        X = X.bfloat16().float()
    return X


def blocked_inverse(A):
    """A: [64,64] strictly-lower f32. Returns Ai=(I+A)^-1 via blocked 16x16 + back-sub, bf16 matmuls."""
    I16 = torch.eye(B, device=DEV, dtype=torch.float32)

    def blk(T, i, j):
        return T[i * B:(i + 1) * B, j * B:(j + 1) * B]

    M = torch.eye(BT, device=DEV) + A
    X = torch.zeros(BT, BT, device=DEV)
    D = [None] * NB
    # diagonal blocks
    for i in range(NB):
        D[i] = ns_inv_bf16(I16 + blk(A, i, i))
        blk(X, i, i).copy_(D[i])
    # block forward-substitution (lower-tri): X_ij = -D_ii @ sum_{k=j..i-1} A_ik @ X_kj
    for j in range(NB):
        for i in range(j + 1, NB):
            S = torch.zeros(B, B, device=DEV)
            for k in range(j, i):
                S = S + (blk(A, i, k).bfloat16().float() @ blk(X, k, j).bfloat16().float())
            Xij = -(D[i].bfloat16().float() @ S.bfloat16().float())
            blk(X, i, j).copy_(Xij)
    return X


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    torch.manual_seed(0)
    worst = 1.0
    for trial in range(5):
        k = torch.randn(BT, 128, device=DEV); k = k / k.norm(dim=-1, keepdim=True)
        beta = torch.rand(BT, device=DEV).clamp_min(0.1)
        g = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(BT, device=DEV)), 0)
        A = (k @ k.T * torch.exp(g[:, None] - g[None, :]) * beta[:, None]) * torch.tril(torch.ones(BT, BT, device=DEV), -1)
        Ai = blocked_inverse(A)
        Ai_ref = torch.linalg.inv(torch.eye(BT, device=DEV) + A)
        c = cos(Ai, Ai_ref)
        resid = float(((torch.eye(BT, device=DEV) + A) @ Ai - torch.eye(BT, device=DEV)).abs().max())
        worst = min(worst, c)
        print(f"  trial {trial}: cosine={c:.6f} resid={resid:.3e}")
    ok = worst >= 0.999
    print(f"[blocked bf16 inverse sim] worst cosine={worst:.6f} -> {'PASS' if ok else 'FAIL'}")
    print("BLOCKED-NUMERICS OK" if ok else "BLOCKED-NUMERICS FAIL")
    import sys; sys.exit(0 if ok else 1)


main()
