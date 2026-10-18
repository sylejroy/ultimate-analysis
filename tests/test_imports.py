"""Every relative import in the package names a module that exists, also the imports
made inside functions, which only run when their code does."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"


class ImportsTest(unittest.TestCase):
    def test_every_relative_import_resolves(self):
        sys.path.insert(0, str(SOURCE))
        self.addCleanup(sys.path.remove, str(SOURCE))
        missing = []
        for path in sorted((SOURCE / "ultimate_analysis").rglob("*.py")):
            parts = path.relative_to(SOURCE).with_suffix("").parts
            package = list(parts if path.name == "__init__.py" else parts[:-1])
            if path.name == "__init__.py":
                package = package[:-1]
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ImportFrom) or not node.level:
                    continue
                base = package[: len(package) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
                try:
                    found = importlib.util.find_spec(module) is not None
                except ModuleNotFoundError:  # Not even the package it would be in
                    found = False
                if not found:
                    missing.append(f"{path.relative_to(SOURCE)}:{node.lineno} {module}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
