"""项目配置与环境变量加载工具。"""

from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """加载 YAML 全局配置，并尝试加载项目根目录 .env 环境变量。"""
    try:
        from dotenv import load_dotenv

        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass

    import yaml

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
