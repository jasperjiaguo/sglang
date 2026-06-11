"""P1a — torch reference oracle for the kkt_inv_uw kernel.

Mirrors FLA's intra stage EXACTLY (pinned from chunk_scaled_dot_kkt + solve_tril
+ wy_fast.recompute_w_u), per 64-token chunk, per v-head (k-head = h//(H//Hg)):

    A[i,j] = beta_i * <k_i,k_j> * exp(g_i - g_j)   for i>j, else 0   (g = chunk-local cumsum)
    Ai     = (I + A)^-1                              (exact, or Newton-Schulz with n_iter)
    U      = Ai @ (beta * V)
    W      = Ai @ (beta * exp(g) * K)

`n_iter=None` -> exact inverse (cross-checks the oracle against FLA's exact solve_tril).
`n_iter=int`  -> Newton-Schulz iterative inverse (matches what the Hopper kernel will do).
"""
import torch


def newton_schulz_inv(M, n_iter):
    """Iterative inverse of unit-lower-triangular M = I + A (A strictly lower, nilpotent).
    X_{k+1} = X_k (2I - M X_k).  Init X0 = 2I - M = I - A (first Neumann term).
    Quadratic convergence: order doubles each iter; ~ceil(log2(BT)) iters = exact for BT=64.
    """
    *batch, n, _ = M.shape
    I = torch.eye(n, device=M.device, dtype=M.dtype).expand(*batch, n, n)
    X = 2 * I - M
    for _ in range(n_iter):
        X = X @ (2 * I - M @ X)
    return X


def ref_kkt_inv_uw(k, v, beta, g_cs, chunk_size=64, n_iter=None):
    """
    Args (already chunk-local-cumsum'd g):
        k    : [B, T, Hg, Dk]  bf16/fp32
        v    : [B, T, H,  Dv]
        beta : [B, T, H]
        g_cs : [B, T, H]   (output of chunk_local_cumsum)
    Returns (all fp32):
        Ai : [B, NT, H, BT, BT]   inverse (I+A)^-1
        U  : [B, T, H, Dv]
        W  : [B, T, H, Dk]
    """
    B, T, Hg, Dk = k.shape
    H, Dv = v.shape[2], v.shape[3]
    BT = chunk_size
    assert T % BT == 0, f"T={T} not divisible by BT={BT}"
    NT = T // BT
    rep = H // Hg

    kf, vf, bf, gf = k.float(), v.float(), beta.float(), g_cs.float()

    # -> chunks, head-major: [B, NT, H, BT, D]
    kc = kf.reshape(B, NT, BT, Hg, Dk).repeat_interleave(rep, dim=3).permute(0, 1, 3, 2, 4)
    vc = vf.reshape(B, NT, BT, H, Dv).permute(0, 1, 3, 2, 4)
    bc = bf.reshape(B, NT, BT, H).permute(0, 1, 3, 2)            # [B,NT,H,BT]
    gc = gf.reshape(B, NT, BT, H).permute(0, 1, 3, 2)            # [B,NT,H,BT]

    KKt = kc @ kc.transpose(-1, -2)                              # [B,NT,H,BT,BT]
    decay = torch.exp(gc[..., :, None] - gc[..., None, :])       # exp(g_i - g_j)
    A = KKt * decay * bc[..., :, None]                           # * beta_i
    tril = torch.tril(torch.ones(BT, BT, device=k.device, dtype=torch.bool), -1)
    A = A * tril                                                 # strictly lower

    I = torch.eye(BT, device=k.device, dtype=torch.float32)
    M = I + A
    Ai = torch.linalg.inv(M) if n_iter is None else newton_schulz_inv(M, n_iter)

    Vb = vc * bc[..., :, None]                                   # beta * V
    Kb = kc * bc[..., :, None] * torch.exp(gc)[..., :, None]     # beta * exp(g) * K
    U = (Ai @ Vb).permute(0, 1, 3, 2, 4).reshape(B, T, H, Dv)
    W = (Ai @ Kb).permute(0, 1, 3, 2, 4).reshape(B, T, H, Dk)
    return Ai, U, W
