"""T02-T04 核心模块单元测试（数据切分、向量检索配置与 LLM 客户端）。"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import load_config
from src.ingestion.chunker import chunk_kb_records
from src.ingestion.cmedqa import build_ood_eval_set, read_jsonl
from src.llm.client import LLMClient


class TestChunkerAndData(unittest.TestCase):
    def test_chunk_kb_records_metadata(self) -> None:
        records = [
            {
                "doc_id": "cmedqa_1001",
                "question_id": "1001",
                "question": "感冒发烧怎么办？",
                "answer": "多喝温水，注意休息。" * 30,
                "source": "cMedQA2",
            }
        ]
        chunks = chunk_kb_records(records, chunk_size=120, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertEqual(c["doc_id"], "cmedqa_1001")
            self.assertEqual(c["chunk_id"], f"cmedqa_1001_c{idx}")
            self.assertEqual(c["chunk_index"], idx)
            self.assertEqual(c["source"], "cMedQA2")

    def test_ood_eval_set_count(self) -> None:
        items = build_ood_eval_set(ood_size=30, seed=42)
        self.assertEqual(len(items), 30)
        self.assertTrue(all(x["expected_refusal"] is True for x in items))

    def test_processed_datasets_no_leakage_if_present(self) -> None:
        kb_path = Path("data/processed/kb_corpus.jsonl")
        eval_path = Path("data/processed/eval_in_domain.jsonl")
        if kb_path.exists() and eval_path.exists():
            kb = read_jsonl(kb_path)
            ev = read_jsonl(eval_path)
            self.assertGreaterEqual(len(kb), 5000)
            kb_qids = {r["question_id"] for r in kb}
            ev_qids = {r["question_id"] for r in ev}
            self.assertEqual(len(kb_qids & ev_qids), 0)


class TestLLMClient(unittest.TestCase):
    def test_missing_deepseek_key_raises(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                LLMClient(provider="deepseek", api_key="")

    def test_local_provider_without_key_works(self) -> None:
        client = LLMClient(provider="local", model="qwen2.5-7b-instruct")
        self.assertEqual(client.provider, "local")
        self.assertEqual(client.base_url, "http://localhost:8000/v1")

    def test_chat_mocked_response(self) -> None:
        client = LLMClient(provider="deepseek", api_key="sk-test-mock")
        mock_choice = MagicMock()
        mock_choice.message.content = "建议多喝水并观察体温变化[1]。"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "deepseek-chat"
        mock_resp.usage.prompt_tokens = 120
        mock_resp.usage.completion_tokens = 25
        mock_resp.usage.total_tokens = 145

        client._client.chat.completions.create = MagicMock(return_value=mock_resp)
        res = client.chat([{"role": "user", "content": "发烧怎么办？"}])
        self.assertIn("多喝水", res.content)
        self.assertEqual(res.total_tokens, 145)

    def test_from_config(self) -> None:
        cfg = load_config()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-mock"}, clear=False):
            client = LLMClient.from_config(cfg)
            self.assertEqual(client.model, "deepseek-chat")
            self.assertGreaterEqual(client.max_context_tokens, 32768)


if __name__ == "__main__":
    unittest.main()
