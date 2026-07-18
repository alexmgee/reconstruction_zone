"""Standalone checks for the generated WEB7 shell artifact."""

from __future__ import annotations

import ast
from pathlib import Path


SHELL_HTML_PATH = Path(__file__).resolve().parents[1] / "reconstruction_web" / "shell_html.py"
FORBIDDEN_NON_LITERAL_TOKENS = (
    "open(",
    "read_bytes",
    "read_text",
    "Path(",
    "importlib",
    "__file__",
)


def _source_tree_and_assignment() -> tuple[str, ast.Module, ast.AnnAssign]:
    source_bytes = SHELL_HTML_PATH.read_bytes()
    assert b"\r" not in source_bytes
    source = source_bytes.decode("utf-8", errors="strict")
    tree = ast.parse(source, filename=str(SHELL_HTML_PATH))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SHELL_HTML"
    ]
    assert len(assignments) == 1
    return source, tree, assignments[0]


def _literal_span(source: str, value: ast.expr) -> tuple[str, str, str]:
    assert value.lineno is not None
    assert value.col_offset is not None
    assert value.end_lineno is not None
    assert value.end_col_offset is not None

    lines = source.splitlines(keepends=True)
    start = sum(len(line.encode("utf-8")) for line in lines[: value.lineno - 1]) + value.col_offset
    end = sum(len(line.encode("utf-8")) for line in lines[: value.end_lineno - 1]) + value.end_col_offset
    source_bytes = source.encode("utf-8")
    before = source_bytes[:start].decode("utf-8", errors="strict")
    literal = source_bytes[start:end].decode("utf-8", errors="strict")
    after = source_bytes[end:].decode("utf-8", errors="strict")
    assert before + literal + after == source
    return before, literal, after


def test_shell_html_is_one_plain_module_level_bytes_constant():
    source, tree, assignment = _source_tree_and_assignment()

    assert source.startswith("# generated — do not edit\nfrom typing import Final\n\n")
    assert len(tree.body) == 2
    import_node, assignment_node = tree.body
    assert isinstance(import_node, ast.ImportFrom)
    assert import_node.module == "typing"
    assert [(alias.name, alias.asname) for alias in import_node.names] == [("Final", None)]
    assert assignment_node is assignment
    assert isinstance(assignment.value, ast.Constant)
    assert isinstance(assignment.value.value, bytes)


def test_shell_html_non_literal_source_has_no_file_loader_tokens():
    source, _, assignment = _source_tree_and_assignment()
    before, literal, after = _literal_span(source, assignment.value)

    literal_expression = ast.parse(f"({literal})", mode="eval").body
    assert isinstance(literal_expression, ast.Constant)
    assert literal_expression.value == assignment.value.value

    excised_source = before + after
    for forbidden in FORBIDDEN_NON_LITERAL_TOKENS:
        assert forbidden not in excised_source

    replacement_tree = ast.parse(before + 'b""' + after)
    replacement_assignments = [node for node in replacement_tree.body if isinstance(node, ast.AnnAssign)]
    assert len(replacement_assignments) == 1
    replacement_value = replacement_assignments[0].value
    assert isinstance(replacement_value, ast.Constant)
    assert replacement_value.value == b""
