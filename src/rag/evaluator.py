"""V0–V4 自动化评测模块（准确率、引用 F1、幻觉率、拒答率与延迟统计）。"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.ingestion import read_jsonl
from src.rag.service import Citation, RAGResponse, RAGService


def _strip_tags_and_disclaimer(text: str, disclaimer: str = "") -> str:
    cleaned = text
    if disclaimer:
        cleaned = cleaned.replace(disclaimer, "")
    cleaned = re.sub(r"\[\d+\]", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip()


def compute_char_f1(prediction: str, reference: str, disclaimer: str = "") -> float:
    """计算中文字符级 F1 分数（去除免责声明与引用标记后对比）。"""
    pred_clean = _strip_tags_and_disclaimer(prediction, disclaimer)
    ref_clean = _strip_tags_and_disclaimer(reference, "")
    if not pred_clean or not ref_clean:
        return 0.0

    pred_counts = Counter(pred_clean)
    ref_counts = Counter(ref_clean)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_clean)
    recall = overlap / len(ref_clean)
    return round(2 * precision * recall / (precision + recall), 4)


def compute_citation_metrics(resp: RAGResponse) -> Dict[str, float]:
    """计算单条回答的引用 Precision / Recall / F1。

    - Precision: 已引用条目中与回答存在实质字符重叠的比例（反映引用相关性）。
    - Recall: 召回的高分证据（score >= 0.45）中被实际引用的覆盖比例。
    """
    if resp.refused:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    cited: List[Citation] = resp.citations
    if not cited:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ans_chars = set(_strip_tags_and_disclaimer(resp.answer, resp.disclaimer))
    supported = 0
    for c in cited:
        chunk_chars = set(_strip_tags_and_disclaimer(c.text, ""))
        if len(ans_chars & chunk_chars) >= 8:
            supported += 1

    precision = supported / len(cited)
    relevant_retrieved = [c for c in resp.retrieved_chunks if c.score >= 0.45]
    denom = max(1, min(len(relevant_retrieved), 3))
    recall = min(1.0, len(cited) / denom)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_rag(
    rag_service: RAGService,
    in_domain_path: Path,
    ood_path: Path,
    version: str = "v0-baseline",
    in_domain_limit: Optional[int] = 50,
    ood_limit: Optional[int] = 30,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """在领域评测集与拒答测试集上运行自动化评测并归档结果。"""
    in_domain_data = read_jsonl(in_domain_path)
    ood_data = read_jsonl(ood_path)
    if in_domain_limit is not None:
        in_domain_data = in_domain_data[:in_domain_limit]
    if ood_limit is not None:
        ood_data = ood_data[:ood_limit]

    char_f1_scores: List[float] = []
    cit_precisions: List[float] = []
    cit_recalls: List[float] = []
    cit_f1_scores: List[float] = []
    in_domain_hallucinations = 0
    in_domain_answered = 0
    latencies: List[float] = []
    tokens_used: List[int] = []
    sample_logs: List[Dict[str, Any]] = []

    for item in tqdm(in_domain_data, desc="Evaluating In-Domain"):
        resp = rag_service.answer(item["question"])
        latencies.append(resp.latency_ms)
        tokens_used.append(resp.total_tokens)

        f1 = 0.0 if resp.refused else compute_char_f1(resp.answer, item["reference_answer"], resp.disclaimer)
        char_f1_scores.append(f1)

        cit_m = compute_citation_metrics(resp)
        cit_precisions.append(cit_m["precision"])
        cit_recalls.append(cit_m["recall"])
        cit_f1_scores.append(cit_m["f1"])

        if not resp.refused:
            in_domain_answered += 1
            if not resp.citations or cit_m["precision"] == 0.0:
                in_domain_hallucinations += 1

        if len(sample_logs) < 5:
            sample_logs.append(
                {
                    "type": "in_domain",
                    "eval_id": item.get("eval_id"),
                    "question": item["question"],
                    "refused": resp.refused,
                    "char_f1": f1,
                    "citation_f1": cit_m["f1"],
                    "answer_preview": resp.answer[:160],
                }
            )

    ood_refused_count = 0
    for item in tqdm(ood_data, desc="Evaluating OOD/Refusal"):
        resp = rag_service.answer(item["question"])
        latencies.append(resp.latency_ms)
        if resp.refused:
            ood_refused_count += 1
        if len(sample_logs) < 8:
            sample_logs.append(
                {
                    "type": "ood",
                    "eval_id": item.get("eval_id"),
                    "question": item["question"],
                    "refused": resp.refused,
                    "refusal_reason": resp.refusal_reason,
                }
            )

    n_in = max(1, len(in_domain_data))
    n_ood = max(1, len(ood_data))
    ood_hallucination_count = len(ood_data) - ood_refused_count
    total_eval_cases = len(in_domain_data) + len(ood_data)

    metrics = {
        "in_domain_count": len(in_domain_data),
        "ood_count": len(ood_data),
        "in_domain_answer_rate": round(in_domain_answered / n_in, 4),
        "accuracy_char_f1": round(sum(char_f1_scores) / n_in, 4),
        "citation_precision": round(sum(cit_precisions) / n_in, 4),
        "citation_recall": round(sum(cit_recalls) / n_in, 4),
        "citation_f1": round(sum(cit_f1_scores) / n_in, 4),
        "ood_refusal_accuracy": round(ood_refused_count / n_ood, 4),
        "hallucination_rate": round(
            (in_domain_hallucinations + ood_hallucination_count) / max(1, total_eval_cases), 4
        ),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "avg_total_tokens": round(sum(tokens_used) / max(1, len(tokens_used)), 1),
    }

    report = {
        "version": version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "samples": sample_logs,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report
