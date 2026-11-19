"""长上下文（>32k tokens）专项压力测试脚本（对应 T09）。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.ingestion import read_jsonl
from src.llm.client import LLMClient
from src.rag.context_manager import ContextBudgetManager
from src.rag.service import RAGService
from src.retrieval.vectorstore import RetrievedChunk


def _build_medical_history(kb_path: Path, target_est_tokens: int) -> List[Dict[str, str]]:
    """使用真实 CMedQA 医疗问答记录构造指定估算 Token 规模的长对话历史。"""
    records = read_jsonl(kb_path)
    history: List[Dict[str, str]] = []
    est_tokens = 0
    idx = 0
    while est_tokens < target_est_tokens:
        rec = records[idx % len(records)]
        q = f"[病历参考 #{idx + 1}] {rec['question']}"
        a = f"历史诊疗记录：{rec['answer']}"
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": a})
        est_tokens += LLMClient.estimate_tokens(q) + LLMClient.estimate_tokens(a)
        idx += 1
    return history


def run_offline_stress(kb_path: Path) -> Dict[str, Any]:
    """离线验证 >32k 放行与超上限优雅截断（保护 Top-1 证据）。"""
    # 1. >32k 正常放行测试（max_context_tokens=65536）
    manager = ContextBudgetManager(max_context_tokens=65536, max_output_tokens=1536)
    history_35k = _build_medical_history(kb_path, target_est_tokens=36000)
    chunks = [
        RetrievedChunk(
            chunk_id="needle_c0",
            doc_id="needle_doc",
            text="患者青霉素皮试阳性，严禁使用阿莫西林，专项核验编号为 MED-32K-2026。",
            score=0.95,
            metadata={"question": "用药禁忌"},
        ),
        RetrievedChunk(
            chunk_id="sup_c1",
            doc_id="sup_doc",
            text="青霉素类药物过敏者应避免一切β-内酰胺类交叉过敏风险药物。",
            score=0.82,
            metadata={"question": "交叉过敏"},
        ),
    ]
    _, kept_35k, stats_35k = manager.build(
        chunks=chunks,
        history=history_35k,
        query="该患者的用药禁忌和核验编号是什么？",
        max_history_turns=0,
    )
    assert stats_35k.estimated_prompt_tokens > 32768, f"未超过 32k: {stats_35k.estimated_prompt_tokens}"
    assert stats_35k.included_chunks == 2
    assert stats_35k.dropped_chunks == 0
    assert len(kept_35k) > 0

    # 2. 超 64k 上限截断保护测试（>70k tokens 输入）
    history_72k = _build_medical_history(kb_path, target_est_tokens=72000)
    ctx_over, kept_over, stats_over = manager.build(
        chunks=chunks,
        history=history_72k,
        query="该患者的用药禁忌和核验编号是什么？",
        max_history_turns=0,
    )
    assert stats_over.truncated is True
    assert stats_over.estimated_prompt_tokens <= 65536
    assert stats_over.included_chunks >= 1
    assert "MED-32K-2026" in ctx_over
    assert stats_over.dropped_history_msgs > 0

    return {
        "passed": True,
        "case_gt_32k": {
            "estimated_prompt_tokens": stats_35k.estimated_prompt_tokens,
            "included_chunks": stats_35k.included_chunks,
            "included_history_msgs": stats_35k.included_history_msgs,
            "truncated": stats_35k.truncated,
        },
        "case_over_limit_truncation": {
            "estimated_prompt_tokens": stats_over.estimated_prompt_tokens,
            "included_chunks": stats_over.included_chunks,
            "dropped_history_msgs": stats_over.dropped_history_msgs,
            "truncated": stats_over.truncated,
            "top1_preserved": "MED-32K-2026" in ctx_over,
        },
    }


def run_online_needle_test(cfg: Dict[str, Any], kb_path: Path, target_est_tokens: int) -> Dict[str, Any]:
    """真实调用 DeepSeek API 执行 >32k tokens 大海捞针端到端测试。"""
    needle_chunk = RetrievedChunk(
        chunk_id="needle_32k_c0",
        doc_id="med_needle_32k",
        text=(
            "【专项病历摘要】患者青霉素皮试阳性，严禁使用阿莫西林、"
            "氨苄西林等青霉素类药物，替代方案需由主治医师评估；"
            "专项核验编号为 MED-32K-2026。"
        ),
        score=0.95,
        metadata={"question": "青霉素过敏患者用药禁忌与专项核验编号", "source": "clinical_needle"},
    )

    class _NeedleRetriever:
        def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
            return [needle_chunk]

    llm_client = LLMClient.from_config(cfg)
    max_ctx = int(cfg.get("llm", {}).get("max_context_tokens", 65536))
    rag = RAGService(
        retriever=_NeedleRetriever(),
        llm_client=llm_client,
        max_history_turns=0,  # 不按轮数截断，完全按 >32k Token 预算装填
        top_k=1,
        min_similarity_score=0.50,
        max_context_tokens=max_ctx,
        max_output_tokens=512,
        query_rewrite_enabled=False,
        require_citation=True,
    )

    history = _build_medical_history(kb_path, target_est_tokens=target_est_tokens)
    query = "结合病历摘要，请明确指出该患者严禁使用的具体药物、原因以及专项核验编号，并标注引用。"
    resp = rag.answer(query, history=history)

    hit_code = "MED-32K-2026" in resp.answer
    hit_drug = ("青霉素" in resp.answer) or ("阿莫西林" in resp.answer)
    has_citation = len(resp.citations) >= 1
    gt_32k = resp.prompt_tokens > 32768

    passed = (not resp.refused) and gt_32k and hit_code and hit_drug and has_citation
    result = {
        "passed": passed,
        "prompt_tokens_real": resp.prompt_tokens,
        "completion_tokens_real": resp.completion_tokens,
        "total_tokens_real": resp.total_tokens,
        "estimated_prompt_tokens": resp.context_stats.estimated_prompt_tokens if resp.context_stats else 0,
        "exceeds_32k": gt_32k,
        "refused": resp.refused,
        "hit_needle_code": hit_code,
        "hit_drug_name": hit_drug,
        "citations_count": len(resp.citations),
        "latency_ms": resp.latency_ms,
        "answer_preview": resp.answer[:300],
    }
    if not passed:
        raise AssertionError(f"在线 >32k 长上下文测试未通过: {json.dumps(result, ensure_ascii=False)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 >32k 长上下文专项压力测试")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--mode", choices=["offline", "online", "both"], default="both")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=60000,
        help="在线压测构造历史的估算目标 Token 数（确保真实分词后稳定 >32768）",
    )
    parser.add_argument("--output", type=str, default="results/v1_long_context_test.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    processed_dir = PROJECT_ROOT / cfg.get("data", {}).get("processed_dir", "data/processed")
    kb_path = processed_dir / "kb_corpus.jsonl"

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
    }

    if args.mode in ("offline", "both"):
        print("[1/2] 运行离线 >32k 放行与超限截断保护压测...")
        report["offline"] = run_offline_stress(kb_path)
        print("    PASS:", json.dumps(report["offline"], ensure_ascii=False))

    if args.mode in ("online", "both"):
        print("[2/2] 运行 DeepSeek API >32k 大海捞针端到端压测...")
        report["online"] = run_online_needle_test(cfg, kb_path, target_est_tokens=args.target_tokens)
        print("    PASS:", json.dumps(report["online"], ensure_ascii=False))

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 长上下文压测报告已保存至: {out_path}")


if __name__ == "__main__":
    main()
