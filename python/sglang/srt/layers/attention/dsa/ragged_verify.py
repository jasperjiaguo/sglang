"""Packed verification geometry for the one-query-per-row DSA kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


def update_ragged_dsa_query_rows(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    layout: RaggedVerifyLayout,
    query_seq_lens: torch.Tensor,
    query_req_pool_indices: torch.Tensor,
) -> None:
    """Fill stable buffers with each packed query's causal KV length and owner.

    ``seq_lens`` excludes the verification block. A query at local offset j
    sees exactly ``seq_lens[request] + j + 1`` keys, including itself. Capped
    graph layouts can leave tokens after ``qo_indptr[-1]``; those ghost rows
    have length zero and must never select a real KV entry.

    CUDA uses a kernel that writes the supplied buffers without allocations or
    device-to-host reads. The CPU path is a reference for geometry tests.
    """
    if seq_lens.ndim != 1 or req_pool_indices.shape != seq_lens.shape:
        raise ValueError(
            "DSA request lengths and pool indices must be matching vectors"
        )
    num_requests = layout.qo_indptr_device.numel() - 1
    if layout.qo_indptr_device.ndim != 1 or not 0 <= num_requests <= seq_lens.numel():
        raise ValueError("DSA query boundaries exceed the available request slots")
    if (
        query_seq_lens.ndim != 1
        or query_req_pool_indices.shape != query_seq_lens.shape
        or query_seq_lens.numel() != layout.graph_num_tokens
    ):
        raise ValueError("DSA query buffers must match the verification token tier")
    if query_seq_lens.dtype != torch.int32:
        raise ValueError("DSA query lengths must use int32")
    if query_req_pool_indices.dtype != req_pool_indices.dtype:
        raise ValueError("DSA query and request pool indices must have the same dtype")
    if not query_seq_lens.is_contiguous() or not query_req_pool_indices.is_contiguous():
        raise ValueError("DSA query output buffers must be contiguous")
    if any(
        tensor.device != seq_lens.device
        for tensor in (
            req_pool_indices,
            layout.qo_indptr_device,
            query_seq_lens,
            query_req_pool_indices,
        )
    ):
        raise ValueError("DSA verification geometry must use a single device")
    if seq_lens.is_cuda:
        from sglang.kernels.ops.attention.dsa.metadata import (
            fused_dsa_ragged_query_rows,
        )

        fused_dsa_ragged_query_rows(
            seq_lens,
            req_pool_indices,
            layout.qo_indptr_device,
            query_seq_lens,
            query_req_pool_indices,
        )
        return

    # This branch is deliberately CPU-only: searchsorted/gather intermediates
    # must not introduce allocations or hidden synchronization on replay.
    if num_requests == 0:
        query_seq_lens.zero_()
        query_req_pool_indices.zero_()
        return
    tokens = torch.arange(query_seq_lens.numel(), dtype=torch.int32)
    rows = torch.searchsorted(layout.qo_indptr_device[1:], tokens, right=True)
    live = rows < num_requests
    safe_rows = rows.clamp_max(num_requests - 1).long()
    lengths = seq_lens[safe_rows] + tokens - layout.qo_indptr_device[safe_rows] + 1
    query_seq_lens.copy_(torch.where(live, lengths, 0))
    query_req_pool_indices.copy_(torch.where(live, req_pool_indices[safe_rows], 0))
