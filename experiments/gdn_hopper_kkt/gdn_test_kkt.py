"""Milestone test runner for the Hopper kkt_inv_uw kernel — the TDD gate.

Usage: MILESTONE=2a python gdn_test_kkt.py
Each milestone asserts the kernel output against gdn_torch_ref (the validated oracle),
cosine >= 0.999 (identity milestones require exact match). Exits 0 on PASS, 1 on FAIL.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_shapes as S
from gdn_torch_ref import ref_kkt_inv_uw
import gdn_hopper_kkt as K

DEV = "cuda"
M = os.environ.get("MILESTONE", "2a")


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max())


def l2norm_k(k):
    return k / (k.float().norm(dim=-1, keepdim=True) + 1e-6).to(k.dtype)


def make_single_chunk():
    """One chunk, one head, L2-normalized K (realistic conditioning)."""
    torch.manual_seed(0)
    k = l2norm_k(torch.randn(K.BT, K.DK, device=DEV, dtype=torch.bfloat16))
    v = torch.randn(K.BT, K.DV, device=DEV, dtype=torch.bfloat16)
    beta = torch.rand(K.BT, device=DEV, dtype=torch.float32).clamp_min(0.1)
    g_raw = torch.nn.functional.logsigmoid(torch.randn(K.BT, device=DEV, dtype=torch.float32))
    g_cs = torch.cumsum(g_raw, dim=0)  # single chunk => local cumsum == global cumsum
    return k, v, beta, g_cs


def report(name, got, ref, exact=False):
    c, m = cos(got, ref), maxabs(got, ref)
    ok = (torch.equal(got, ref) if exact else c >= 0.999)
    print(f"[{M}] {name}: cosine={c:.6f} maxabs={m:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def ref_A_single(k, beta, g_cs):
    kf = k.float()
    KKt = kf @ kf.T
    decay = torch.exp(g_cs[:, None] - g_cs[None, :])
    A = KKt * decay * beta[:, None]
    tril = torch.tril(torch.ones(K.BT, K.BT, device=k.device, dtype=torch.bool), -1)
    return A * tril


def test_2a():
    k, _, _, _ = make_single_chunk()
    out = K.load_k_identity(k)
    return report("K-tile identity through smem", out, k, exact=True)


def test_2b():
    k, v, beta, g_cs = make_single_chunk()
    A_got = K.compute_A(k, beta, g_cs)
    A_ref = ref_A_single(k, beta, g_cs)
    return report("A = strictLower(beta*KKt*decay)", A_got, A_ref)


TESTS = {"2a": test_2a, "2b": test_2b}


def main():
    if M not in TESTS:
        print(f"milestone {M} not implemented yet; available: {sorted(TESTS)}")
        sys.exit(2)
    ok = TESTS[M]()
    print(f"=== MILESTONE {M}: {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
