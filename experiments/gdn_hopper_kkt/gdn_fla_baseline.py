"""P0 — FLA GDN chunked-prefill per-stage baseline on H200, at Qwen3.5-35B shapes.

Times the four forward stages separately to (a) confirm the intra stage
(= kkt_inv_uw counterpart: "fused kkt + solve_tril + recompute_w_u") is a
meaningful fraction, and (b) set the bar the Hopper kernel must beat.

Stages:
  cumsum : chunk_local_cumsum(g)                      (gate prep)
  intra  : chunk_gated_delta_rule_fwd_intra -> w,u,A  <== kkt_inv_uw target
  h      : chunk_gated_delta_rule_fwd_h     -> h,v_new
  o      : chunk_fwd_o                       -> o
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_shapes as S

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o

DEV = "cuda"
DT = torch.bfloat16
WARMUP = int(os.environ.get("WARMUP", "10"))
ITERS = int(os.environ.get("ITERS", "50"))


def make_inputs(B, T):
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(B, T, S.NUM_K_HEADS, S.HEAD_K_DIM, device=DEV, dtype=DT, generator=g)
    k = torch.randn(B, T, S.NUM_K_HEADS, S.HEAD_K_DIM, device=DEV, dtype=DT, generator=g)
    v = torch.randn(B, T, S.NUM_V_HEADS, S.HEAD_V_DIM, device=DEV, dtype=DT, generator=g)
    # gate log-decay (negative, small) in fp32; beta in (0,1)
    gate = torch.nn.functional.logsigmoid(
        torch.randn(B, T, S.NUM_V_HEADS, device=DEV, dtype=torch.float32, generator=g)
    )
    beta = torch.rand(B, T, S.NUM_V_HEADS, device=DEV, dtype=torch.float32, generator=g).clamp_min(0.1)
    return q, k, v, gate, beta


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def run(B, T):
    q, k, v, gate, beta = make_inputs(B, T)
    scale = S.HEAD_K_DIM ** -0.5

    # gate cumsum (must precede intra/h/o; matches chunk_gated_delta_rule_fwd)
    g_cs = chunk_local_cumsum(gate, chunk_size=S.CHUNK_SIZE)

    def f_cumsum():
        chunk_local_cumsum(gate, chunk_size=S.CHUNK_SIZE)

    def f_intra():
        return chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cs, beta=beta, chunk_size=S.CHUNK_SIZE)

    w, u, A = f_intra()

    # h kernel always reads initial_state_indices; real prefill supplies a zeros state.
    N = B
    init_state = torch.zeros(N, S.NUM_V_HEADS, S.HEAD_V_DIM, S.HEAD_K_DIM, device=DEV, dtype=DT)
    init_idx = torch.arange(N, device=DEV, dtype=torch.long)

    def f_h():
        return chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g_cs,
            initial_state=init_state, initial_state_indices=init_idx,
        )

    t_cumsum = time_fn(f_cumsum)
    t_intra = time_fn(lambda: f_intra())
    t_h, t_o = float("nan"), float("nan")
    try:
        _r = f_h()
        h, v_new = _r[0], _r[1]
        t_h = time_fn(lambda: f_h())

        def f_o():
            return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cs, scale=scale)

        f_o()
        t_o = time_fn(lambda: f_o())
    except Exception as e:
        print(f"  [h/o timing skipped: {type(e).__name__}: {e}]")

    import math
    parts = [t_intra, t_h, t_o]
    three = sum(p for p in parts if not math.isnan(p))
    pct = lambda x: f"{100*x/three:5.1f}%" if three and not math.isnan(x) else "  n/a"
    print(f"\n=== T={T} (B={B}, k_heads={S.NUM_K_HEADS}, v_heads={S.NUM_V_HEADS}, d={S.HEAD_K_DIM}) ===")
    print(f"  cumsum : {t_cumsum:8.4f} ms")
    print(f"  intra  : {t_intra:8.4f} ms   <== kkt_inv_uw target   ({pct(t_intra)} of 3-kernel)")
    print(f"  h      : {t_h:8.4f} ms                               ({pct(t_h)})")
    print(f"  o      : {t_o:8.4f} ms                               ({pct(t_o)})")
    print(f"  ---- intra+h+o = {three:8.4f} ms")
    return dict(T=T, cumsum=t_cumsum, intra=t_intra, h=t_h, o=t_o)


def main():
    print("Device:", torch.cuda.get_device_name(0))
    print(S.summary())
    print(f"warmup={WARMUP} iters={ITERS}")
    rows = []
    for T in S.SEQLENS:
        rows.append(run(S.BATCH, T))
    import math
    print("\n=== summary (ms) ===")
    print(f"{'T':>8} {'intra(kkt)':>12} {'h':>10} {'o':>10} {'intra%':>8}")
    for r in rows:
        three = sum(r[k] for k in ('intra', 'h', 'o') if not math.isnan(r[k]))
        ip = f"{100*r['intra']/three:>6.1f}%" if three else "   n/a"
        print(f"{r['T']:>8} {r['intra']:>12.4f} {r['h']:>10.4f} {r['o']:>10.4f} {ip:>8}")


if __name__ == "__main__":
    main()
