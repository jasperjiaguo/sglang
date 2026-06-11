"""GDN chunked-prefill shapes for Qwen3.5-35B-A3B (text_config), pinned from
/shared/public/elr-models/Qwen/Qwen3.5-35B-A3B/config.json (2026-06-11).

These are the ONLY model-derived facts the kernel work needs — no weights.
"""

# --- Qwen3.5-35B-A3B GDN (linear-attention) layer config ---
HEAD_K_DIM = 128          # linear_key_head_dim
HEAD_V_DIM = 128          # linear_value_head_dim
NUM_K_HEADS = 16          # linear_num_key_heads
NUM_V_HEADS = 32          # linear_num_value_heads  (GVA ratio v/k = 2)
CONV_KERNEL = 4           # linear_conv_kernel_dim  (not used by kkt_inv_uw)
CHUNK_SIZE = 64           # BT — hardcoded in both FLA (CHUNK_SIZE) and the Blackwell kernel

# Prefill sweep (single sequence, batch=1)
SEQLENS = [2048, 8192, 32768]
BATCH = 1

# dtypes: GDN state is native BF16; gate log-decay accumulated in fp32
DTYPE = "bfloat16"

def summary():
    return (
        f"Qwen3.5-35B-A3B GDN: k_heads={NUM_K_HEADS} v_heads={NUM_V_HEADS} "
        f"head_k={HEAD_K_DIM} head_v={HEAD_V_DIM} BT={CHUNK_SIZE} "
        f"seqlens={SEQLENS}"
    )

if __name__ == "__main__":
    print(summary())
