"""工程骨架基础导入冒烟测试。"""

import importlib
import unittest


class TestPackageImports(unittest.TestCase):
    """验证核心模块包均可正常导入。"""

    def test_package_imports(self) -> None:
        modules = [
            "src",
            "src.config",
            "src.ingestion",
            "src.retrieval",
            "src.llm",
            "src.rag",
            "src.app",
        ]
        for mod in modules:
            self.assertIsNotNone(importlib.import_module(mod))


if __name__ == "__main__":
    unittest.main()
