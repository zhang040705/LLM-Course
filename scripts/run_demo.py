"""Web Demo 一键启动脚本（MVP: Gradio）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.gradio_app import build_gradio_app
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="启动中文医疗 RAG 问答系统 Web Demo")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--host", type=str, default=None, help="监听地址")
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    parser.add_argument("--share", action="store_true", help="是否生成 Gradio 公网分享链接")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    app_cfg = cfg.get("app", {})
    host = args.host or app_cfg.get("host", "127.0.0.1")
    port = args.port or int(app_cfg.get("port", 7860))

    demo = build_gradio_app(config=cfg)
    demo.launch(server_name=host, server_port=port, share=args.share)


if __name__ == "__main__":
    main()
