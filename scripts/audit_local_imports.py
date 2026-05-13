"""
Local import audit — find imports of local modules that don't resolve.

Parses Python files with ast, builds a map of local modules from the repo,
and reports any import that references a local module path which doesn't
exist. Exits nonzero if unresolved imports are found.

Usage:
    python scripts/audit_local_imports.py
"""
from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories containing local Python packages/modules
SOURCE_DIRS = [
    ROOT / "reconstruction_gui",
    ROOT / "prep360",
    ROOT / "scripts",
    ROOT / "tests",
]

# reconstruction_gui is added to sys.path at runtime, so modules like
# "tabs.source_tab", "widgets", "app_infra" etc. are importable as
# top-level names. These are the known top-level aliases.
GUI_DIR = ROOT / "reconstruction_gui"

# Prefixes that identify a local import (vs external like numpy, torch, etc.)
# Built dynamically from the repo contents.
LOCAL_PREFIXES: set[str] = set()

# Explicit allowlist: imports that are intentionally optional/dynamic
# and may not resolve in all environments. Each entry needs a reason.
ALLOWLIST: dict[str, str] = {
    # Example: "some_module": "dynamically loaded only when feature X is enabled"
}


def build_module_map() -> set[str]:
    """Build a set of all importable module paths from the repo."""
    modules: set[str] = set()

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue
        for py in source_dir.rglob("*.py"):
            rel = py.relative_to(ROOT)
            parts = list(rel.with_suffix("").parts)

            # Full dotted path: reconstruction_gui.tabs.source_tab
            modules.add(".".join(parts))

            # Package path: reconstruction_gui.tabs (for __init__.py)
            if parts[-1] == "__init__":
                modules.add(".".join(parts[:-1]))

    # Top-level aliases from reconstruction_gui being on sys.path
    for py in GUI_DIR.glob("*.py"):
        modules.add(py.stem)
    for d in GUI_DIR.iterdir():
        if d.is_dir() and (d / "__init__.py").exists():
            modules.add(d.name)
            # Also add submodules: tabs.source_tab, tabs.alignment_tab, etc.
            for py in d.rglob("*.py"):
                rel = py.relative_to(GUI_DIR)
                parts = list(rel.with_suffix("").parts)
                if parts[-1] == "__init__":
                    modules.add(".".join(parts[:-1]))
                else:
                    modules.add(".".join(parts))

    return modules


def build_local_prefixes(modules: set[str]) -> set[str]:
    """Extract top-level module names to identify local imports."""
    prefixes: set[str] = set()
    for m in modules:
        prefixes.add(m.split(".")[0])
    return prefixes


def is_local_import(module_name: str, prefixes: set[str]) -> bool:
    """Check if an import looks like it's referencing a local module."""
    top = module_name.split(".")[0]
    return top in prefixes


def module_resolves(module_name: str, modules: set[str]) -> bool:
    """Check if a module name resolves to any known local module."""
    # Exact match
    if module_name in modules:
        return True
    # Package match (importing a package that exists)
    if any(m == module_name or m.startswith(module_name + ".") for m in modules):
        return True
    return False


def module_name_for_path(filepath: Path) -> str:
    """Return the dotted module path represented by a repo-local Python file."""
    rel = filepath.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_name_for_path(filepath: Path) -> str:
    """Return the package context used to resolve relative imports."""
    module_name = module_name_for_path(filepath)
    if not module_name:
        return ""
    if filepath.name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0] if "." in module_name else ""


def resolve_from_module(filepath: Path, node: ast.ImportFrom) -> str | None:
    """Resolve absolute or relative `from ... import ...` module targets."""
    if node.level == 0:
        return node.module

    package_name = package_name_for_path(filepath)
    package_parts = package_name.split(".") if package_name else []
    keep_count = len(package_parts) - max(node.level - 1, 0)
    if keep_count < 0:
        return None

    resolved_parts = package_parts[:keep_count]
    if node.module:
        resolved_parts.extend(node.module.split("."))
    return ".".join(part for part in resolved_parts if part)


def audit_file(filepath: Path, modules: set[str], prefixes: set[str]) -> list[str]:
    """Audit a single Python file for unresolved local imports."""
    issues: list[str] = []
    try:
        source = filepath.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        issues.append(f"{filepath.relative_to(ROOT)}: PARSE ERROR: {e}")
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if not is_local_import(mod, prefixes):
                    continue
                if mod in ALLOWLIST:
                    continue
                if not module_resolves(mod, modules):
                    rel = filepath.relative_to(ROOT)
                    issues.append(f"{rel}:{node.lineno} import {mod}")

        if isinstance(node, ast.ImportFrom):
            mod = resolve_from_module(filepath, node)
            if not mod:
                continue
            if not is_local_import(mod, prefixes):
                continue
            if mod in ALLOWLIST:
                continue
            if not module_resolves(mod, modules):
                rel = filepath.relative_to(ROOT)
                issues.append(f"{rel}:{node.lineno} from {mod}")

    return issues


def main() -> int:
    modules = build_module_map()
    prefixes = build_local_prefixes(modules)

    all_issues: list[str] = []

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue
        for py in source_dir.rglob("*.py"):
            issues = audit_file(py, modules, prefixes)
            all_issues.extend(issues)

    if all_issues:
        print("UNRESOLVED LOCAL IMPORTS:")
        for issue in sorted(all_issues):
            print(f"  {issue}")
        print(f"\n{len(all_issues)} unresolved import(s) found.")
        return 1
    else:
        print("No unresolved local imports found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
