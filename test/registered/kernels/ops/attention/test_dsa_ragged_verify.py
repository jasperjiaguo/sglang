"""Real packed DSA kernels, including changing layouts under CUDA graph replay."""

import unittest
from dataclasses import fields
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_split,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.ragged_verify import update_ragged_dsa_query_rows
from sglang.srt.layers.attention.dsa.utils import (
    resolve_fp8_paged_mqa_logits_fn,
    resolve_num_sms,
    resolve_paged_mqa_logits_metadata_fn,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SLOTS, TIER, WIDTH, TOPK = 8, 16, 256, 64


def _backend():
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device = torch.device("cuda")
    backend.device_sm_major = torch.cuda.get_device_capability()[0]
    backend.attn_dcp_size, backend.attn_dcp_rank = 1, 0
    backend.dsa_index_kpool, backend.dsa_index_topk = 1, TOPK
    backend.hisparse_coordinator = backend._kv_shard_pool = None
    backend.dsa_decode_impl = "flashmla_sparse"
    backend.physical_page_size = backend.real_page_span = 64
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.speculative_num_draft_tokens = 6
    backend.set_dsa_prefill_impl = lambda batch: None
    backend._arange_buf = torch.arange(TIER + 1, device="cuda", dtype=torch.int32)
    backend.req_to_token = torch.arange(
        SLOTS * WIDTH, device="cuda", dtype=torch.int32
    ).view(SLOTS, WIDTH)
    backend.decode_cuda_graph_metadata = {
        "page_table": torch.zeros(TIER, WIDTH, device="cuda", dtype=torch.int32)
    }
    backend.ragged_verify_cuda_graph_metadata = {}
    backend._sink_pad_cache = {}
    backend._init_kpool_metadata_fusion()
    return backend


def _batch(prefix, requests, verify):
    # Poison padded request slots: an accidental last-request continuation or
    # use of captured slots instead of live indptr would read out of bounds.
    seq = torch.full((SLOTS,), 1000000, device="cuda", dtype=torch.int64)
    req = torch.full((SLOTS,), 1000000, device="cuda", dtype=torch.int64)
    seq[: len(prefix)] = torch.tensor(prefix, device="cuda")
    req[: len(requests)] = torch.tensor(requests, device="cuda")
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify, device=torch.device("cuda"), grid=[TIER]
    )
    return SimpleNamespace(
        batch_size=SLOTS,
        seq_lens=seq,
        req_pool_indices=req,
        forward_mode=ForwardMode.TARGET_VERIFY,
        ragged_verify_layout=layout,
    )


def _addresses(metadata):
    return {
        f.name: value.data_ptr()
        for f in fields(metadata)
        if isinstance(value := getattr(metadata, f.name), torch.Tensor)
    }


class TestDSARaggedVerifyCapability(unittest.TestCase):
    def test_unsupported_configs_cannot_enter_ragged_metadata(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        supported = dict(
            dsa_decode_impl="flashmla_sparse",
            attn_dcp_size=1,
            dsa_index_kpool=1,
            hisparse_coordinator=None,
            _kv_shard_pool=None,
        )
        for name, value in supported.items():
            setattr(backend, name, value)
        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=True
        ):
            self.assertTrue(backend.supports_ragged_verify_graph)
            for name, value in (
                ("dsa_decode_impl", "flashmla_kv"),
                ("dsa_decode_impl", "trtllm"),
                ("attn_dcp_size", 2),
                ("dsa_index_kpool", 4),
                ("hisparse_coordinator", object()),
                ("_kv_shard_pool", object()),
            ):
                with self.subTest(name=name, value=value):
                    setattr(backend, name, value)
                    self.assertFalse(backend.supports_ragged_verify_graph)
                    with self.assertRaisesRegex(NotImplementedError, "flashmla_sparse"):
                        backend._init_ragged_verify_metadata(None, cuda_graph=False)
                    setattr(backend, name, supported[name])
        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=False
        ):
            self.assertFalse(backend.supports_ragged_verify_graph)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDSARaggedVerifyCUDA(unittest.TestCase):
    def test_empty_request_slots_skip_poisoned_owners(self):
        prefix = torch.tensor([62, 1000000, 127, 1000000], device="cuda")
        requests = torch.tensor([7, 1000000, 3, 1000000], device="cuda")
        layout = SimpleNamespace(
            graph_num_tokens=8,
            qo_indptr_device=torch.tensor(
                [0, 1, 1, 4, 4], device="cuda", dtype=torch.int32
            ),
        )
        lengths = torch.empty(8, device="cuda", dtype=torch.int32)
        owners = torch.empty(8, device="cuda", dtype=torch.int64)
        update_ragged_dsa_query_rows(prefix, requests, layout, lengths, owners)
        self.assertEqual(lengths.tolist(), [63, 128, 129, 130, 0, 0, 0, 0])
        self.assertEqual(owners.tolist(), [7, 3, 3, 3, 0, 0, 0, 0])

    def test_native_indexer_sparse_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("native FP8 DSA requires Hopper or newer")
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        ):
            self._check_native_graph("deepgemm")

    def test_cutedsl_indexer_sparse_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("CuTe DSL paged DSA requires Blackwell")
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        ):
            self._check_native_graph("cutedsl")

    def _check_native_graph(self, logits_backend):
        torch.manual_seed(19)
        backend = _backend()
        first = _batch([0, 62, 127], [5, 2, 7], [1, 3, 6])
        backend._init_ragged_verify_metadata(first, cuda_graph=True)
        metadata = backend.forward_metadata
        pointers = _addresses(metadata)
        num_pages = SLOTS * WIDTH // 64
        index_k = torch.randn(num_pages, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(num_pages, 64, device="cuda", dtype=torch.float32)
        packed = torch.empty(num_pages, 64 * 132, device="cuda", dtype=torch.uint8)
        packed[:, : 64 * 128] = index_k.view(torch.uint8).reshape(num_pages, -1)
        packed[:, 64 * 128 :] = scales.view(torch.uint8).reshape(num_pages, -1)
        index_cache = packed.view(torch.float8_e4m3fn).view(num_pages, 64, 1, 132)
        index_q = torch.randn(TIER, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        weights = torch.rand(TIER, 64, device="cuda")
        q = torch.randn(TIER, 16, 576, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(SLOTS * WIDTH, 1, 576, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(16, device="cuda", dtype=torch.float32)
        indexer_metadata = DSAIndexerMetadata(
            metadata,
            TopkTransformMethod.PAGED,
            topk_backend=DSATopKBackend.SGL_KERNEL,
        )

        def forward():
            if logits_backend == "cutedsl":
                logits = cutedsl_paged_mqa_logits(
                    index_q,
                    index_cache,
                    weights,
                    metadata.cache_seqlens_int32,
                    metadata.real_page_table,
                    metadata.paged_mqa_schedule_metadata,
                    WIDTH,
                    q_offset=TIER,
                    B=TIER,
                    next_n=1,
                    is_target_verify=True,
                    dsl_expand_factor=1,
                    dsl_atom=1,
                    blocksize=64,
                    sm_count=resolve_num_sms(),
                    get_paged_mqa_logits_metadata_fn=resolve_paged_mqa_logits_metadata_fn(),
                )
            else:
                logits = deepgemm_paged_mqa_logits_split(
                    resolve_fp8_paged_mqa_logits_fn(),
                    index_q,
                    index_cache,
                    weights,
                    metadata.paged_mqa_ctx_lens_2d,
                    metadata.real_page_table,
                    metadata.paged_mqa_schedule_metadata,
                    WIDTH,
                    q_offset=TIER,
                )
            selected = indexer_metadata.topk_transform(logits, TOPK)
            output = backend._forward_flashmla_sparse(
                q,
                kv,
                512,
                selected,
                576**-0.5,
                topk_length=metadata.dsa_cache_seqlens_int32,
                attn_sink=sink,
            )
            return selected, output

        # Warm the actual native kernels before capture; no replacements or
        # synthetic logits are used in this graph.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_topk, graph_output = forward()
        torch.cuda.current_stream().wait_stream(stream)

        for prefix, requests, verify in (
            ([0, 62, 127], [5, 2, 7], [1, 3, 6]),
            ([127], [3], [1]),
            ([63, 126], [7, 2], [6, 6]),
            ([0, 62, 127], [2, 7, 5], [6, 1, 3]),
        ):
            batch = _batch(prefix, requests, verify)
            backend._init_ragged_verify_metadata(batch, cuda_graph=True)
            self.assertIs(backend.forward_metadata, metadata)
            self.assertEqual(_addresses(metadata), pointers)
            expected_lengths = [
                prefix[r] + j + 1 for r, n in enumerate(verify) for j in range(n)
            ]
            live_tokens = len(expected_lengths)
            self.assertEqual(
                metadata.cache_seqlens_int32.tolist(),
                expected_lengths + [0] * (TIER - live_tokens),
            )
            padded_lengths = expected_lengths + [0] * (TIER - live_tokens)
            self.assertEqual(
                metadata.cu_seqlens_k.tolist(), [0, *accumulate(padded_lengths)]
            )
            self.assertEqual(
                metadata.dsa_cu_seqlens_k.tolist(),
                [0, *accumulate(min(n, TOPK) for n in padded_lengths)],
            )
            self.assertEqual(metadata.cu_seqlens_q.tolist(), list(range(TIER + 1)))
            eager_topk, eager_output = forward()
            graph.replay()
            torch.testing.assert_close(graph_topk, eager_topk, rtol=0, atol=0)
            torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(graph_output).all())
            self.assertTrue((graph_topk[live_tokens:] == -1).all())
            self.assertTrue((graph_output[live_tokens:] == 0).all())

            # Independent causal reference validates page ownership as well as
            # the selected sparse attention, including the learned sink term.
            for token, length in enumerate(expected_lengths):
                row = next(
                    r for r in range(len(verify)) if token < sum(verify[: r + 1])
                )
                chosen = graph_topk[token]
                chosen = chosen[chosen >= 0].long()
                logical = chosen - requests[row] * WIDTH
                self.assertTrue(((logical >= 0) & (logical < length)).all())
                self.assertEqual(chosen.numel(), min(length, TOPK))
                keys = kv[chosen, 0].float()
                scores = (q[token].float() @ keys.T) * 576**-0.5
                probs = torch.softmax(torch.cat([scores, sink[:, None]], dim=1), dim=1)
                reference = probs[:, :-1] @ keys[:, :512]
                torch.testing.assert_close(
                    graph_output[token].float(), reference, atol=0.025, rtol=0.025
                )


if __name__ == "__main__":
    unittest.main()
