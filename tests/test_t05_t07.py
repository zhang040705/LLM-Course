"""T05-T07 离线单元测试（RAGService、评测指标计算与 Gradio 界面构建）。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.app.gradio_app import build_gradio_app, format_citations_markdown
from src.llm.client import LLMResponse
from src.rag.evaluator import compute_char_f1, compute_citation_metrics, evaluate_rag
from src.rag.service import RAGService
from src.retrieval.vectorstore import RetrievedChunk


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query: str, top_k: int = 5):
        return self._chunks[:top_k]


class TestRAGService(unittest.TestCase):
    def test_low_similarity_triggers_refusal_without_llm(self) -> None:
        retriever = _FakeRetriever(
            [
                RetrievedChunk(
                    chunk_id="c1",
                    doc_id="d1",
                    text="不相关文本",
                    score=0.30,
                    metadata={"question": "q"},
                )
            ]
        )
        llm = MagicMock()
        service = RAGService(retriever=retriever, llm_client=llm, min_similarity_score=0.45)

        resp = service.answer("Python 怎么写协程？")
        self.assertTrue(resp.refused)
        self.assertIn("low_similarity", resp.refusal_reason)
        self.assertIn("医疗免责声明", resp.answer)
        llm.chat.assert_not_called()

    def test_citation_filtering_and_disclaimer(self) -> None:
        retriever = _FakeRetriever(
            [
                RetrievedChunk(
                    chunk_id="d1_c0",
                    doc_id="d1",
                    text="问题：孕期腹泻怎么办\n解答：建议多喝温开水，注意腹部保暖，必要时就医检查。",
                    score=0.72,
                    metadata={"question": "孕期腹泻怎么办", "source": "cMedQA2"},
                )
            ]
        )
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="建议多喝温开水并注意腹部保暖[1]，避免生冷食物[99]。",
            model="deepseek-chat",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            latency_ms=350.0,
        )
        service = RAGService(retriever=retriever, llm_client=llm, min_similarity_score=0.45)

        resp = service.answer("孕期腹泻怎么办？", history=[{"role": "user", "content": "你好"}])
        self.assertFalse(resp.refused)
        self.assertIn("[1]", resp.answer)
        self.assertNotIn("[99]", resp.answer)
        self.assertEqual(len(resp.citations), 1)
        self.assertEqual(resp.citations[0].doc_id, "d1")
        self.assertIn("医疗免责声明", resp.answer)

    def test_llm_refusal_prefix_handled(self) -> None:
        retriever = _FakeRetriever(
            [
                RetrievedChunk(
                    chunk_id="d1_c0",
                    doc_id="d1",
                    text="仅提及轻微头痛。",
                    score=0.50,
                    metadata={"question": "头痛"},
                )
            ]
        )
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="[REFUSE] 资料不足以给出明确诊断，请及时就医。",
            model="deepseek-chat",
        )
        service = RAGService(retriever=retriever, llm_client=llm, min_similarity_score=0.45)
        resp = service.answer("能帮我确诊吗？")
        self.assertTrue(resp.refused)
        self.assertEqual(resp.refusal_reason, "llm_refusal")
        self.assertEqual(resp.citations, [])
        self.assertNotIn("[REFUSE]", resp.answer)


class TestEvaluatorAndGradio(unittest.TestCase):
    def test_char_f1_and_evaluate_rag(self) -> None:
        self.assertGreater(compute_char_f1("多喝温水注意休息", "建议多喝温水，注意休息"), 0.7)

        retriever = _FakeRetriever(
            [
                RetrievedChunk(
                    chunk_id="d1_c0",
                    doc_id="d1",
                    text="建议多喝温开水，注意休息，观察体温变化。",
                    score=0.75,
                    metadata={"question": "发烧怎么办"},
                )
            ]
        )
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="建议多喝温开水，注意休息，观察体温变化[1]。",
            model="deepseek-chat",
            total_tokens=90,
        )
        service = RAGService(retriever=retriever, llm_client=llm, min_similarity_score=0.45)

        # 对 OOD 查询返回低相似度，模拟真实拦截
        orig_search = retriever.search

        def dynamic_search(query: str, top_k: int = 5):
            if "asyncio" in query:
                return [RetrievedChunk("ood_c0", "d_ood", "无关文本", 0.25, {})]
            return orig_search(query, top_k)

        service.retriever.search = dynamic_search

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            in_path = tmp_dir / "in.jsonl"
            ood_path = tmp_dir / "ood.jsonl"
            out_path = tmp_dir / "out.json"

            in_path.write_text(
                json.dumps(
                    {
                        "eval_id": "med_0001",
                        "question": "发烧怎么办？",
                        "reference_answer": "建议多喝温开水，注意休息。",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            ood_path.write_text(
                json.dumps(
                    {
                        "eval_id": "ood_0001",
                        "question": "Python asyncio 怎么用？",
                        "expected_refusal": True,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report = evaluate_rag(
                rag_service=service,
                in_domain_path=in_path,
                ood_path=ood_path,
                in_domain_limit=None,
                ood_limit=None,
                output_path=out_path,
            )
            self.assertTrue(out_path.exists())
            self.assertEqual(report["metrics"]["ood_refusal_accuracy"], 1.0)
            self.assertGreater(report["metrics"]["citation_f1"], 0.5)

    def test_gradio_blocks_build_and_citation_format(self) -> None:
        retriever = _FakeRetriever([])
        service = RAGService(retriever=retriever, llm_client=None, min_similarity_score=0.45)
        resp = service.answer("测试空检索")
        md = format_citations_markdown(resp)
        self.assertIn("触发拒答", md)

        demo = build_gradio_app(rag_service=service)
        self.assertIsNotNone(demo)


if __name__ == "__main__":
    unittest.main()
