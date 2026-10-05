"""The layering of the package: a module only imports from its own layer and the ones below.

See "Layout" in docs/DEVELOPMENT_GUIDELINES.md.
"""

import ast
import unittest
from pathlib import Path
from typing import Iterator, Set

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ultimate_analysis"

BASE = {"config", "constants", "utils"}
# Layer -> layers its modules may import from. The GUI may import everything.
ALLOWED = {
    "config": BASE,
    "constants": BASE,
    "utils": BASE,
    "processing": BASE | {"processing"},
    "optimization": BASE | {"processing", "optimization"},
    "training": BASE | {"training"},
    "rendering": BASE | {"processing", "rendering"},
    "pipeline": BASE | {"processing", "rendering"},
    # The phone labelling page; like the GUI it sits on top, but must work without Qt
    "web": BASE | {"processing", "web"},
}


def layer_of(path: Path) -> str:
    """Top-level package or module a file belongs to, e.g. "processing" or "pipeline"."""
    return path.relative_to(PACKAGE).parts[0].removesuffix(".py")


def imports_of(path: Path) -> Iterator[str]:
    """Dotted names a file imports, with relative imports resolved inside the package."""
    package_parts = ("ultimate_analysis", *path.relative_to(PACKAGE).parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = package_parts[: len(package_parts) - node.level + 1]
                module = ".".join((*parent, *(node.module or "").split("."))).rstrip(".")
                # "from . import x" names the modules themselves
                names = [alias.name for alias in node.names] if not node.module else [""]
                yield from (f"{module}.{name}".rstrip(".") for name in names)
            else:
                yield node.module or ""


class LayeringTests(unittest.TestCase):
    def test_modules_only_import_from_their_own_layer_and_the_ones_below(self):
        violations = []
        for path in sorted(PACKAGE.rglob("*.py")):
            layer = layer_of(path)
            if layer not in ALLOWED:
                continue
            for name in imports_of(path):
                parts = name.split(".")
                if parts[0] == "PyQt5":
                    violations.append(f"{path.relative_to(PACKAGE)} imports Qt ({name})")
                elif parts[0] == "ultimate_analysis" and len(parts) > 1:
                    if parts[1] not in ALLOWED[layer]:
                        violations.append(f"{path.relative_to(PACKAGE)} imports {name}")
        self.assertEqual(violations, [])

    def test_every_layer_is_covered(self):
        layers: Set[str] = {
            layer_of(path) for path in PACKAGE.rglob("*.py") if path.name != "__init__.py"
        }
        self.assertEqual(layers - set(ALLOWED), {"gui"})


if __name__ == "__main__":
    unittest.main()
