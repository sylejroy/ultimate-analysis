"""Repo file size guard.

Scans the repository for text files and checks line counts against:
- Soft limit: MAX_FILE_SIZE_LINES from ultimate_analysis.constants (default 500)
- Hard limit: 600 lines (warning tolerance window for well-structured files)

Exits non-zero if any file exceeds the hard limit.

Notes:
- Excludes heavy/non-source directories (data/, .venv/, caches) and binary formats.
- Intended to be run via VS Code task "Check File Sizes".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def find_project_root(start: Path) -> Path:
    # Assume this script lives in <root>/scripts/
    if start.name == "scripts":
        return start.parent
    return start


def try_get_soft_limit(root: Path) -> int:
    # Default soft limit
    default_limit = 500
    try:
        sys.path.insert(0, str(root / "src"))
        from ultimate_analysis.constants import MAX_FILE_SIZE_LINES  # type: ignore

        return int(MAX_FILE_SIZE_LINES)
    except Exception:
        return default_limit
    finally:
        # Clean up path insertion
        try:
            sys.path.remove(str(root / "src"))
        except ValueError:
            pass


def iter_files(root: Path, include_exts: Iterable[str], exclude_dirs: Iterable[str]) -> Iterable[Path]:
    exclude_set = {d.lower() for d in exclude_dirs}
    include_set = {e.lower() for e in include_exts}

    for p in root.rglob("*"):
        if p.is_dir():
            # Skip excluded directories by name match at any depth
            if p.name.lower() in exclude_set:
                # Prune traversal by skipping children of this directory
                # rglob can't prune directly; rely on name checks for files below
                continue
            else:
                continue
        # Skip files in excluded dir segments
        if any(seg.lower() in exclude_set for seg in p.parts):
            continue
        # Skip obvious binaries and notebooks
        if p.suffix.lower() in {".ipynb", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".ico", ".pdf", ".pt", ".onnx", ".engine"}:
            continue
        if p.suffix and p.suffix.lower() not in include_set:
            continue
        yield p


def count_lines(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    root = find_project_root(script_dir)

    include_exts = [
        ".py",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".txt",
        ".sh",
        ".ps1",
        ".bat",
    ]
    exclude_dirs = [
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "node_modules",
        "data",
        "models",
        "output",
        "logs",
    ]

    soft_limit = try_get_soft_limit(root)  # typically 500
    hard_limit = max(soft_limit + 100, 600)  # enforce at least 600 as hard cap

    warnings: List[Tuple[Path, int]] = []
    failures: List[Tuple[Path, int]] = []
    scanned = 0

    for file in iter_files(root, include_exts, exclude_dirs):
        lc = count_lines(file)
        if lc is None:
            continue
        scanned += 1
        if lc > hard_limit:
            failures.append((file, lc))
        elif lc > soft_limit:
            warnings.append((file, lc))

    # Output summary
    print(f"File size check: soft={soft_limit} lines, hard={hard_limit} lines")
    print(f"Scanned {scanned} files under {root}")

    if warnings:
        print("\nWARN: Files exceeding soft limit (consider splitting/refactoring):")
        for p, lc in sorted(warnings, key=lambda t: t[1], reverse=True):
            rel = p.relative_to(root)
            print(f"  - {rel} : {lc} lines")

    if failures:
        print("\nFAIL: Files exceeding hard limit (must fix):")
        for p, lc in sorted(failures, key=lambda t: t[1], reverse=True):
            rel = p.relative_to(root)
            print(f"  - {rel} : {lc} lines")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
