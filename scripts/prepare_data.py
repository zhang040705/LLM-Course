"""一键数据准备与向量库构建脚本（对应 T02 + T03）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.ingestion import (
    chunk_kb_records,
    download_cmedqa2,
    load_and_clean_cmedqa2,
    read_jsonl,
    split_and_save_datasets,
)
from src.retrieval import ChromaVectorStore, create_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 CMedQA2 数据集并构建本地向量索引")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--kb-size", type=int, default=None, help="覆盖知识库条目数")
    parser.add_argument("--force-download", action="store_true", help="强制重新下载原始数据")
    parser.add_argument("--skip-index", action="store_true", help="仅做清洗划分，跳过向量建库")
    parser.add_argument("--reset-index", action="store_true", help="建库前清空旧向量集合")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    data_cfg = cfg.get("data", {})
    ret_cfg = cfg.get("retrieval", {})

    raw_dir = PROJECT_ROOT / data_cfg.get("raw_dir", "data/raw")
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_dir", "data/processed")
    samples_dir = PROJECT_ROOT / data_cfg.get("samples_dir", "data/samples")

    kb_size = args.kb_size or int(data_cfg.get("kb_size", 5000))
    eval_size = int(data_cfg.get("eval_size", 300))
    ood_size = int(data_cfg.get("ood_size", 80))
    seed = int(cfg.get("project", {}).get("seed", 42))

    print("[1/4] 检查并获取 CMedQA2 原始数据...")
    q_zip, a_zip = download_cmedqa2(raw_dir, force=args.force_download)

    print("[2/4] 清洗数据并执行防泄漏划分...")
    records = load_and_clean_cmedqa2(q_zip, a_zip)
    stats = split_and_save_datasets(
        records=records,
        processed_dir=processed_dir,
        samples_dir=samples_dir,
        kb_size=kb_size,
        eval_size=eval_size,
        ood_size=ood_size,
        seed=seed,
    )
    print(
        f"    清洗去重总计: {stats['total_cleaned']} 条 | "
        f"KB: {stats['kb_count']} | "
        f"In-Domain Eval: {stats['eval_in_domain_count']} | "
        f"OOD Eval: {stats['eval_ood_count']}"
    )

    if args.skip_index:
        print("[完成] 已跳过向量库构建（--skip-index）。")
        return

    print("[3/4] 文本分块与元数据绑定...")
    kb_records = read_jsonl(processed_dir / "kb_corpus.jsonl")
    chunks = chunk_kb_records(
        kb_records,
        chunk_size=int(data_cfg.get("chunk_size", 400)),
        chunk_overlap=int(data_cfg.get("chunk_overlap", 50)),
    )
    print(f"    生成分块总计: {len(chunks)} 个")

    print("[4/4] 初始化本地 Embedding 并写入 Chroma 向量库...")
    embedder = create_embedder(cfg)
    store = ChromaVectorStore(
        persist_dir=PROJECT_ROOT / ret_cfg.get("vectorstore_dir", "data/vectorstore"),
        collection_name=ret_cfg.get("collection_name", "cmedqa_kb"),
        embedder=embedder,
    )
    if args.reset_index:
        store.reset()
    written = store.add_chunks(chunks, batch_size=128)
    print(f"[完成] 本次写入 {written} 个分块，集合当前总分块数: {store.count()}")


if __name__ == "__main__":
    main()
