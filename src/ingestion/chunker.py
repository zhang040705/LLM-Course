"""文本分块（Chunking）与溯源元数据绑定模块。"""

from __future__ import annotations

from typing import Any, Dict, List


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """按最大长度与重叠切分文本，优先在中文句末标点处断开。"""
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    start = 0
    n = len(text)
    delimiters = ("。", "！", "？", "\n", "；")

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # 在后半段寻找句末标点，尽量保持语义完整
            window_start = start + chunk_size // 2
            best_cut = -1
            for delim in delimiters:
                pos = text.rfind(delim, window_start, end)
                if pos > best_cut:
                    best_cut = pos + 1
            if best_cut > start:
                end = best_cut

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(start + step, end - chunk_overlap) if end > start else start + step

    return chunks


def chunk_kb_records(
    records: List[Dict[str, Any]],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """将知识库 QA 记录切分为带完整溯源元数据的 Chunk 列表。"""
    all_chunks: List[Dict[str, Any]] = []
    for rec in records:
        doc_id = str(rec["doc_id"])
        question_id = str(rec.get("question_id", ""))
        question = str(rec.get("question", "")).strip()
        answer = str(rec.get("answer", "")).strip()
        source = str(rec.get("source", "cMedQA2"))

        full_text = f"问题：{question}\n解答：{answer}"
        pieces = _split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}_c{idx}",
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "question": question,
                    "source": source,
                    "chunk_index": idx,
                    "text": piece,
                }
            )
    return all_chunks
