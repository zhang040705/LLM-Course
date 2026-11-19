"""自动化评测运行脚本（对应 T07 / 各版本迭代对比）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.rag import RAGService
from src.rag.evaluator import evaluate_rag


def main() -> None:
    parser = argparse.ArgumentParser(description="运行医疗 RAG 系统自动化评测")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--version", type=str, default=None, help="实验版本标识（默认读取 config.yaml）")
    parser.add_argument("--in-domain-limit", type=int, default=50, help="领域评测集抽样条数（0 表示全量）")
    parser.add_argument("--ood-limit", type=int, default=30, help="拒答测试集抽样条数（0 表示全量）")
    parser.add_argument("--output", type=str, default="results/v1_core.json", help="结果输出路径")
    parser.add_argument(
        "--with-multiturn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否同时运行多轮 Query 改写专项评测",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    version = args.version or cfg.get("project", {}).get("version", "v1-core")
    processed_dir = PROJECT_ROOT / cfg.get("data", {}).get("processed_dir", "data/processed")
    rag_service = RAGService.from_config(cfg)

    in_limit = None if args.in_domain_limit <= 0 else args.in_domain_limit
    ood_limit = None if args.ood_limit <= 0 else args.ood_limit
    out_path = PROJECT_ROOT / args.output

    report = evaluate_rag(
        rag_service=rag_service,
        in_domain_path=processed_dir / "eval_in_domain.jsonl",
        ood_path=processed_dir / "eval_ood.jsonl",
        version=version,
        in_domain_limit=in_limit,
        ood_limit=ood_limit,
        output_path=out_path,
        with_multiturn=args.with_multiturn,
    )
    print(f"\n[评测完成] 结果已保存至: {out_path}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if "multiturn_rewrite" in report:
        mt = {k: v for k, v in report["multiturn_rewrite"].items() if k != "cases"}
        print("\n[多轮 Query 改写评测]")
        print(json.dumps(mt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
