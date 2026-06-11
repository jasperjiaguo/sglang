"""P1a gate test: validate the torch reference oracle against FLA's intra stage.

If the EXACT-inverse reference reproduces FLA's (w, u) within bf16 tolerance, the
oracle is trustworthy and becomes the test fixture for every Hopper-kernel milestone.
Also reports how close Newton-Schulz (n_iter sweep) gets to the exact inverse — this
sets the iteration count the Hopper kernel will need.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_shapes as S
from gdn_torch_ref import ref_kkt_inv_uw, newton_schulz_inv

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra

DEV = "cuda"
DT = torch.bfloat16


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    torch.manual_seed(0)
    B, T = 1, 256          # 4 chunks — enough to exercise multi-chunk, fast for a correctness ref
    Hk, Hv, Dk, Dv, BT = S.NUM_K_HEADS, S.NUM_V_HEADS, S.HEAD_K_DIM, S.HEAD_V_DIM, S.CHUNK_SIZE

    k = torch.randn(B, T, Hk, Dk, device=DEV, dtype=DT)
    v = torch.randn(B, T, Hv, Dv, device=DEV, dtype=DT)
    beta = torch.rand(B, T, Hv, device=DEV, dtype=torch.float32).clamp_min(0.1)
    g_raw = torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, device=DEV, dtype=torch.float32))

    # cumsum gate ONCE, feed identical g to both ref and FLA intra
    g_cs = chunk_local_cumsum(g_raw, chunk_size=BT)

    # FLA intra (the kkt_inv_uw counterpart): returns (w, u, A)
    w_fla, u_fla, A_fla = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cs, beta=beta, chunk_size=BT)

    # torch ref, exact inverse
    Ai, U, W = ref_kkt_inv_uw(k, v, beta, g_cs, chunk_size=BT, n_iter=None)

    print(f"shapes: U {tuple(U.shape)} vs u_fla {tuple(u_fla.shape)} | W {tuple(W.shape)} vs w_fla {tuple(w_fla.shape)}")
    cu, cw = cos(U, u_fla), cos(W, w_fla)
    print(f"\n=== oracle (exact inv) vs FLA intra ===")
    print(f"  U: cosine={cu:.6f}  maxabs={maxabs(U, u_fla):.4e}")
    print(f"  W: cosine={cw:.6f}  maxabs={maxabs(W, w_fla):.4e}")
    ok = cu >= 0.999 and cw >= 0.999
    print(f"  ORACLE {'VALID' if ok else 'MISMATCH'} (threshold cosine>=0.999)")

    # Newton-Schulz iteration sweep: how many iters to match the exact inverse?
    print(f"\n=== Newton-Schulz inverse convergence (vs exact, on U) ===")
    for nit in [2, 3, 4, 5, 6, 8]:
        _, U_ns, W_ns = ref_kkt_inv_uw(k, v, beta, g_cs, chunk_size=BT, n_iter=nit)
        print(f"  n_iter={nit}: U cosine(vs FLA)={cos(U_ns, u_fla):.6f}  U cosine(vs exact)={cos(U_ns, U):.6f}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
