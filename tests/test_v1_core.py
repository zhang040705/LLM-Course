"""V1 核心能力单元测试（T08 Query 改写、T09 >32k 上下文预算、T10 引用真实性与拒答守卫）。"""

import unittest
from unittest.mock import MagicMock

from src.llm.client import LLMResponse
from src.rag.citation import normalize_citation_markers, verify_and_extract_citations
from src.rag.context_manager import ContextBudgetManager, normalize_history
from src.rag.rewriter import QueryRewriter
from src.rag.service import RAGService
from src.retrieval.vectorstore import RetrievedChunk


class TestQueryRewriter(unittest.TestCase):
    def test_single_turn_skips_llm(self) -> None:
        llm = MagicMock()
        rewriter = QueryRewriter(llm_client=llm)
        out = rewriter.rewrite("感冒发烧怎么办？", history=None)
        self.assertEqual(out, "感冒发烧怎么办？")
        llm.chat.assert_not_called()

    def test_multiturn_pronoun_rewrite(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="怀孕25周腹泻在饮食上需要注意什么？",
            model="deepseek-chat",
        )
        rewriter = QueryRewriter(llm_client=llm)
        history = [
            {"role": "user", "content": "怀孕25周拉肚子怎么办？"},
            {"role": "assistant", "content": "注意腹部保暖并及时就医。"},
        ]
        out = rewriter.rewrite("那饮食上注意什么？", history=history)
        self.assertIn("怀孕25周", out)
        llm.chat.assert_called_once()

    def test_topic_switch_and_error_fallback(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("network error")
        rewriter = QueryRewriter(llm_client=llm)
        history = [{"role": "user", "content": "感冒怎么办"}]
        out = rewriter.rewrite("Python asyncio 怎么用？", history=history)
        self.assertEqual(out, "Python asyncio 怎么用？")


class TestContextBudgetManager(unittest.TestCase):
    def test_gt_32k_passes_without_truncation(self) -> None:
        manager = ContextBudgetManager(max_context_tokens=65536, max_output_tokens=1536)
        chunks = [
            RetrievedChunk("c1", "d1", "孕期腹泻饮食清淡，避免自行用药。", 0.88, {}),
        ]
        # 构造 ~35k 估算 token 的对话历史
        long_msg = "这是病历记录。" * 500
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": long_msg} for i in range(10)]
        ctx, kept, stats = manager.build(chunks, history, "饮食要注意什么？", max_history_turns=0)
        self.assertGreater(stats.estimated_prompt_tokens, 32768)
        self.assertFalse(stats.truncated)
        self.assertEqual(stats.included_chunks, 1)
        self.assertEqual(len(kept), 10)
        self.assertIn("孕期腹泻", ctx)

    def test_over_budget_truncates_gracefully_and_preserves_top1(self) -> None:
        manager = ContextBudgetManager(max_context_tokens=4096, max_output_tokens=512)
        huge_top1 = "核心禁忌信息MED-NEEDLE。" + ("补充医学描述。" * 800)
        chunks = [
            RetrievedChunk("c1", "d1", huge_top1, 0.90, {}),
            RetrievedChunk("c2", "d2", "第二条证据" * 400, 0.70, {}),
        ]
        history = [{"role": "user", "content": "历史消息" * 300} for _ in range(6)]
        ctx, _, stats = manager.build(chunks, history, "禁忌是什么？", max_history_turns=0)
        self.assertTrue(stats.truncated)
        self.assertGreaterEqual(stats.included_chunks, 1)
        self.assertIn("核心禁忌信息MED-NEEDLE", ctx)

    def test_normalize_history_strips_disclaimer_and_blocks(self) -> None:
        disc = "【医疗免责声明】仅供参考。"
        raw = [
            {"role": "user", "content": [{"text": "你好"}]},
            {"role": "assistant", "content": f"多喝水。\n\n{disc}"},
        ]
        norm = normalize_history(raw, disclaimer=disc)
        self.assertEqual(norm[0]["content"], "你好")
        self.assertEqual(norm[1]["content"], "多喝水。")


class TestCitationVerificationAndServiceV1(unittest.TestCase):
    def test_normalize_and_verify_citations(self) -> None:
        raw = "建议注意休息【1】，多喝温水[1, 2]，避免盲目用药[99][0]。"
        norm = normalize_citation_markers(raw)
        self.assertIn("[1][2]", norm)

        chunks = [
            RetrievedChunk("c1", "d1", "建议注意休息，多喝温水。", 0.85, {}),
            RetrievedChunk("c2", "d2", "多喝温水有助于缓解症状。", 0.78, {}),
        ]
        res = verify_and_extract_citations(raw, chunks, min_overlap_chars=3)
        self.assertEqual([c.index for c in res.citations], [1, 2])
        self.assertEqual(res.invalid_indices, [99, 0])
        self.assertNotIn("[99]", res.cleaned_answer)
        self.assertNotIn("[0]", res.cleaned_answer)

    def test_require_citation_triggers_refusal_when_uncited(self) -> None:
        class _StaticRetriever:
            def search(self, query: str, top_k: int = 5):
                return [RetrievedChunk("c1", "d1", "相关医疗建议。", 0.72, {})]

        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="这是一个没有任何引用编号的回答。",
            model="deepseek-chat",
        )
        svc = RAGService(
            retriever=_StaticRetriever(),
            llm_client=llm,
            min_similarity_score=0.50,
            require_citation=True,
        )
        resp = svc.answer("头疼怎么办？")
        self.assertTrue(resp.refused)
        self.assertEqual(resp.refusal_reason, "no_valid_citation")

    def test_service_v1_rewrite_and_noise_filtering(self) -> None:
        seen_queries = []

        class _SpyRetriever:
            def search(self, query: str, top_k: int = 5):
                seen_queries.append(query)
                return [
                    RetrievedChunk("c1", "d1", "怀孕期腹泻需清淡饮食，勿自行服药。", 0.75, {}),
                    RetrievedChunk("c2", "d2", "低相关性噪声片段。", 0.35, {}),
                ]

        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse(content="怀孕25周腹泻饮食注意事项", model="deepseek-chat"),
            LLMResponse(content="怀孕期腹泻建议清淡饮食，切勿自行服药[1]。", model="deepseek-chat", total_tokens=150),
        ]
        svc = RAGService(
            retriever=_SpyRetriever(),
            llm_client=llm,
            min_similarity_score=0.52,
            min_chunk_score=0.50,
            query_rewrite_enabled=True,
            require_citation=True,
        )
        resp = svc.answer(
            "那饮食要注意什么？",
            history=[{"role": "user", "content": "怀孕25周拉肚子怎么办？"}],
        )
        self.assertFalse(resp.refused)
        self.assertEqual(resp.rewritten_query, "怀孕25周腹泻饮食注意事项")
        self.assertEqual(seen_queries[0], "怀孕25周腹泻饮食注意事项")
        self.assertEqual(len(resp.retrieved_chunks), 1)  # 噪声片段 0.35 已被过滤
        self.assertIsNotNone(resp.context_stats)


if __name__ == "__main__":
    unittest.main()
