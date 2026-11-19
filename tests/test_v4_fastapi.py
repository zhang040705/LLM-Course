"""T15 FastAPI 服务端与 SSE 流式接口单元测试。"""

import json
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.app.fastapi_app import create_fastapi_app
from src.llm.client import LLMResponse
from src.rag.service import RAGService
from src.retrieval.vectorstore import RetrievedChunk


class _FakeRetriever:
    def search(self, query: str, top_k: int = 5):
        if "asyncio" in query:
            return [RetrievedChunk("ood_c0", "d_ood", "不相关片段", 0.05, {})]
        return [
            RetrievedChunk(
                chunk_id="d1_c0",
                doc_id="d1",
                text="孕期腹泻应清淡饮食，注意保暖，切勿自行服药。",
                score=0.92,
                metadata={"question": "孕期腹泻怎么办", "source": "cMedQA2"},
            )
        ]


class TestFastAPIApp(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = MagicMock()
        self.llm.model = "deepseek-chat"
        self.llm.chat.return_value = LLMResponse(
            content="孕期腹泻建议清淡饮食并注意保暖，切勿自行服药[1]。",
            model="deepseek-chat",
            total_tokens=110,
            latency_ms=200.0,
        )
        self.llm.stream_chat.return_value = iter(
            ["孕期腹泻建议", "清淡饮食并注意保暖，", "切勿自行服药[1]。"]
        )
        self.service = RAGService(
            retriever=_FakeRetriever(),
            llm_client=self.llm,
            min_similarity_score=0.20,
            min_chunk_score=0.15,
            require_citation=True,
        )
        self.app = create_fastapi_app(rag_service=self.service)
        self.client = TestClient(self.app)

    def test_index_and_health(self) -> None:
        r_idx = self.client.get("/")
        self.assertEqual(r_idx.status_code, 200)
        self.assertIn("中文医疗领域特定 RAG 问答系统", r_idx.text)

        for asset in (
            "/static/styles.css",
            "/static/mri_shader.js",
            "/static/app.js",
            "/static/assets/mri_slice_texture.png",
        ):
            r_asset = self.client.get(asset)
            self.assertEqual(r_asset.status_code, 200, f"静态资源加载失败: {asset}")

        r_health = self.client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        body = r_health.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("retrieval_mode", body)
        self.assertIn("max_context_tokens", body)
        self.assertIn("min_similarity_score", body)

    def test_rest_chat_and_ood_refusal(self) -> None:
        r_ok = self.client.post("/api/chat", json={"query": "孕期腹泻怎么办？", "history": []})
        self.assertEqual(r_ok.status_code, 200)
        data = r_ok.json()
        self.assertFalse(data["refused"])
        self.assertEqual(len(data["citations"]), 1)
        self.assertAlmostEqual(data["max_score"], 0.92, places=3)
        self.assertIn("医疗免责声明", data["answer"])

        r_ood = self.client.post("/api/chat", json={"query": "Python asyncio 怎么用？", "history": []})
        self.assertEqual(r_ood.status_code, 200)
        self.assertTrue(r_ood.json()["refused"])

    def test_sse_stream_chat(self) -> None:
        r_stream = self.client.post("/api/chat/stream", json={"query": "孕期腹泻怎么办？", "history": []})
        self.assertEqual(r_stream.status_code, 200)
        self.assertIn("text/event-stream", r_stream.headers["content-type"])

        events = []
        for line in r_stream.text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]
        self.assertIn("meta", types)
        self.assertIn("delta", types)
        self.assertEqual(types[-1], "done")
        done_resp = events[-1]["response"]
        self.assertFalse(done_resp["refused"])
        self.assertEqual(len(done_resp["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
