"""V2 检索与重排模块单元测试（T11-T12：BM25、Hybrid RRF、Reranker）。"""

import tempfile
import unittest
from pathlib import Path
from typing import List

from src.retrieval.bm25 import BM25Retriever, tokenize_zh
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import BaseReranker
from src.retrieval.vectorstore import RetrievedChunk


class _FakeDenseStore:
    def __init__(self, hits: List[RetrievedChunk]) -> None:
        self._hits = hits

    def count(self) -> int:
        return len(self._hits)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        return self._hits[:top_k]


class _FakeReranker(BaseReranker):
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        scored = []
        for c in chunks:
            # 简单根据关键医学词命中模拟 Cross-Encoder 精排
            score = 0.92 if "阿莫西林" in c.text else 0.15
            scored.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    text=c.text,
                    score=score,
                    metadata={**c.metadata, "rerank_score": score},
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class TestV2Retrieval(unittest.TestCase):
    def test_tokenize_zh_handles_medical_terms_and_ascii(self) -> None:
        toks = tokenize_zh("怀孕25周服用阿莫西林")
        self.assertIn("25", toks)
        self.assertIn("阿莫", toks)
        self.assertIn("西林", toks)

    def test_bm25_build_save_load_and_search(self) -> None:
        chunks = [
            {
                "chunk_id": "d1_c0",
                "doc_id": "d1",
                "question_id": "1",
                "question": "阿莫西林过敏怎么办",
                "text": "问题：阿莫西林过敏怎么办\n解答：青霉素或阿莫西林过敏应立即停药并就医。",
                "source": "cMedQA2",
                "chunk_index": 0,
            },
            {
                "chunk_id": "d2_c0",
                "doc_id": "d2",
                "question_id": "2",
                "question": "感冒发烧饮食注意",
                "text": "问题：感冒发烧饮食注意\n解答：多喝温开水，饮食清淡易消化。",
                "source": "cMedQA2",
                "chunk_index": 0,
            },
        ]
        bm25 = BM25Retriever()
        self.assertEqual(bm25.build_from_chunks(chunks), 2)

        with tempfile.TemporaryDirectory() as tmp:
            idx_path = Path(tmp) / "bm25.json"
            bm25.save(idx_path)
            loaded = BM25Retriever.load(idx_path)
            hits = loaded.search("阿莫西林过敏如何处理", top_k=2)
            self.assertGreaterEqual(len(hits), 1)
            self.assertEqual(hits[0].doc_id, "d1")

    def test_hybrid_rrf_and_reranker_pipeline(self) -> None:
        dense_hits = [
            RetrievedChunk("d2_c0", "d2", "感冒发烧多喝温水。", 0.72, {"doc_id": "d2"}),
            RetrievedChunk("d1_c0", "d1", "青霉素与阿莫西林过敏严禁再次服用。", 0.65, {"doc_id": "d1"}),
        ]
        bm25 = BM25Retriever()
        bm25.build_from_chunks(
            [
                {"chunk_id": "d1_c0", "doc_id": "d1", "text": "青霉素与阿莫西林过敏严禁再次服用。"},
                {"chunk_id": "d2_c0", "doc_id": "d2", "text": "感冒发烧多喝温水。"},
            ]
        )

        # 不启用重排时，RRF 融合后仍保持可比性分值
        hybrid_no_rerank = HybridRetriever(
            dense_store=_FakeDenseStore(dense_hits),
            bm25_store=bm25,
            reranker=None,
            candidate_k=5,
        )
        res_rrf = hybrid_no_rerank.search("阿莫西林过敏", top_k=2)
        self.assertEqual(len(res_rrf), 2)
        self.assertIn("rrf_score", res_rrf[0].metadata)

        # 启用重排后，相关度最高的 d1 跃升至 Top-1
        hybrid_with_rerank = HybridRetriever(
            dense_store=_FakeDenseStore(dense_hits),
            bm25_store=bm25,
            reranker=_FakeReranker(),
            candidate_k=5,
        )
        res_rerank = hybrid_with_rerank.search("阿莫西林过敏", top_k=2)
        self.assertEqual(res_rerank[0].doc_id, "d1")
        self.assertEqual(res_rerank[0].score, 0.92)


if __name__ == "__main__":
    unittest.main()
