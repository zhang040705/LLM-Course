"""Embedding 模型升级与云端对比实验脚本（对应 T11）。

在同一组可控语料与评测集上对比：
1. 本地轻量 Xenova/bge-small-zh-v1.5 (512-dim ONNX)
2. 本地高精度 Xenova/bge-base-zh-v1.5 (768-dim ONNX)
3. 阿里云百炼 text-embedding-v3 (1024-dim API，若已配置 DASHSCOPE_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.ingestion import read_jsonl
from src.retrieval import BaseEmbedder, DashScopeEmbedder, LocalONNXEmbedder


def _eval_embedder(
    name: str,
    embedder: BaseEmbedder,
    kb_texts: List[str],
    in_queries: List[str],
    ood_queries: List[str],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    doc_vecs = np.asarray(embedder.embed_documents(kb_texts), dtype=np.float32)
    index_sec = time.perf_counter() - t0
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_vecs = doc_vecs / np.clip(doc_norms, 1e-12, None)

    t1 = time.perf_counter()
    in_top1: List[float] = []
    for q in in_queries:
        qv = np.asarray(embedder.embed_query(q), dtype=np.float32)
        qv = qv / max(1e-12, float(np.linalg.norm(qv)))
        sims = doc_vecs @ qv
        in_top1.append(float(np.max(sims)))
    query_ms = ((time.perf_counter() - t1) * 1000.0) / max(1, len(in_queries))

    ood_top1: List[float] = []
    for q in ood_queries:
        qv = np.asarray(embedder.embed_query(q), dtype=np.float32)
        qv = qv / max(1e-12, float(np.linalg.norm(qv)))
        sims = doc_vecs @ qv
        ood_top1.append(float(np.max(sims)))

    in_mean = float(np.mean(in_top1))
    ood_mean = float(np.mean(ood_top1))
    # 使用领域内与 OOD 均值差（separation margin）衡量领域区分度
    return {
        "name": name,
        "dimension": int(doc_vecs.shape[1]),
        "index_docs": len(kb_texts),
        "index_time_sec": round(index_sec, 2),
        "avg_query_latency_ms": round(query_ms, 2),
        "in_domain_top1_mean": round(in_mean, 4),
        "in_domain_top1_min": round(float(np.min(in_top1)), 4),
        "ood_top1_mean": round(ood_mean, 4),
        "ood_top1_max": round(float(np.max(ood_top1)), 4),
        "domain_ood_margin": round(in_mean - ood_mean, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对比本地与云端 Embedding 检索表现（T11）")
    parser.add_argument("--kb-limit", type=int, default=300, help="对比实验抽样知识库条数（兼顾云端免费额度）")
    parser.add_argument("--query-limit", type=int, default=30, help="领域内测试查询条数")
    parser.add_argument("--ood-limit", type=int, default=20, help="域外测试查询条数")
    parser.add_argument("--skip-dashscope", action="store_true", help="跳过阿里云百炼云端对比")
    parser.add_argument("--output", type=str, default="results/v2_embedding_compare.json")
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = PROJECT_ROOT / cfg.get("data", {}).get("processed_dir", "data/processed")
    kb_rows = read_jsonl(processed_dir / "kb_corpus.jsonl")[: args.kb_limit]
    in_rows = read_jsonl(processed_dir / "eval_in_domain.jsonl")[: args.query_limit]
    ood_rows = read_jsonl(processed_dir / "eval_ood.jsonl")[: args.ood_limit]

    kb_texts = [f"问题：{r['question']}\n解答：{r['answer'][:350]}" for r in kb_rows]
    in_queries = [r["question"] for r in in_rows]
    ood_queries = [r["question"] for r in ood_rows]

    candidates: List[tuple[str, BaseEmbedder]] = [
        ("local/bge-small-zh-v1.5", LocalONNXEmbedder(model_id="Xenova/bge-small-zh-v1.5")),
        ("local/bge-base-zh-v1.5", LocalONNXEmbedder(model_id="Xenova/bge-base-zh-v1.5")),
    ]

    dash_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not args.skip_dashscope and dash_key and dash_key != "your_dashscope_api_key_here":
        candidates.append(("cloud/dashscope-text-embedding-v3", DashScopeEmbedder(model="text-embedding-v3")))

    results: List[Dict[str, Any]] = []
    for name, emb in candidates:
        print(f"正在评测 Embedding: {name} ...")
        try:
            res = _eval_embedder(name, emb, kb_texts, in_queries, ood_queries)
            results.append(res)
            print("   ->", json.dumps(res, ensure_ascii=False))
        except Exception as exc:
            print(f"   [跳过] {name} 执行异常: {exc}")
            results.append({"name": name, "error": str(exc)})

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kb_limit": args.kb_limit,
        "query_limit": args.query_limit,
        "ood_limit": args.ood_limit,
        "results": results,
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] Embedding 对比报告已保存至: {out_path}")


if __name__ == "__main__":
    main()
