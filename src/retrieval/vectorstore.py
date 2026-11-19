"""本地持久化向量库与语义检索封装（ChromaDB）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from src.retrieval.embedder import BaseEmbedder


@dataclass
class RetrievedChunk:
    """单条检索召回证据结构。"""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChromaVectorStore:
    """基于 ChromaDB 的持久化本地向量检索器（余弦相似度空间）。"""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedder: BaseEmbedder,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """返回当前集合已入库的分块数量。"""
        return self._collection.count()

    def reset(self) -> None:
        """清空并重建当前集合（用于重新建库）。"""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 128) -> int:
        """批量将分块文本与溯源元数据向量化并写入 Chroma。"""
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [str(item["chunk_id"]) for item in batch]
            docs = [str(item["text"]) for item in batch]
            metadatas = []
            for item in batch:
                meta = {
                    "doc_id": str(item.get("doc_id", "")),
                    "question_id": str(item.get("question_id", "")),
                    "question": str(item.get("question", ""))[:200],
                    "source": str(item.get("source", "cMedQA2")),
                    "chunk_index": int(item.get("chunk_index", 0)),
                }
                metadatas.append(meta)

            embeddings = self.embedder.embed_documents(docs)
            self._collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total += len(batch)
        return total

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """按语义相似度检索 Top-K 医疗证据分块。"""
        if self.count() == 0 or not query.strip():
            return []

        query_vec = self.embedder.embed_query(query)
        res = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        results: List[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma cosine distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            meta_dict = dict(meta or {})
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=str(meta_dict.get("doc_id", "")),
                    text=doc or "",
                    score=round(similarity, 4),
                    metadata=meta_dict,
                )
            )
        return results
