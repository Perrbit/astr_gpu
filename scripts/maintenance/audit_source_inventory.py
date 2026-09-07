#!/usr/bin/env python3
"""Build and check a lexical inventory of tracked ASTR core solver sources."""

import argparse
import hashlib
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SOURCE_SUFFIXES = {".f90", ".cuf", ".inc"}
INVENTORY_PATH = Path("documents/maintenance/generated/source-inventory.md")
SOURCE_ANCHOR = re.compile(
    r"`((?:src|src_gpu)/[^`:]+)::([A-Za-z][A-Za-z0-9_]*)`"
)


@dataclass(frozen=True)
class FortranFacts:
    path: str
    programs: tuple[str, ...]
    modules: tuple[str, ...]
    subroutines: tuple[str, ...]
    functions: tuple[str, ...]
    uses: tuple[str, ...]
    calls: tuple[str, ...]
    includes: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class InventorySnapshot:
    baseline: str
    facts: tuple[FortranFacts, ...]
    memberships: tuple[tuple[str, str], ...]
    include_owners: tuple[tuple[str, tuple[str, ...]], ...]
    fingerprint: str


def tracked_core_files(repo: Path) -> list[Path]:
    """Return tracked production source paths relative to *repo*."""
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "src", "src_gpu"],
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in result.stdout.decode("utf-8").split("\0"):
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.suffix.lower() in SOURCE_SUFFIXES:
            paths.append(path)
    return sorted(paths, key=lambda item: item.as_posix())


def _scan_outside_quotes(text: str, delimiter: str) -> list[str]:
    parts = []
    start = 0
    quote = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == delimiter:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return parts


def _strip_fortran_comment(line: str) -> str:
    return _scan_outside_quotes(line, "!")[0]


def _collapse_space_outside_quotes(text: str) -> str:
    output = []
    quote = None
    pending_space = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            output.append(char)
            if char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    output.append(text[index + 1])
                    index += 2
                    continue
                quote = None
        elif char in {"'", '"'}:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            quote = char
            output.append(char)
        elif char.isspace():
            pending_space = True
        else:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            output.append(char)
        index += 1
    return "".join(output).strip()


def logical_fortran_lines(text: str) -> list[str]:
    """Fold free-form continuations and return comment-free statements."""
    logical_lines = []
    buffer = ""
    continuing = False

    for physical_line in text.splitlines():
        line = _strip_fortran_comment(physical_line).strip()
        if not line:
            continue
        if continuing and line.startswith("&"):
            line = line[1:].lstrip()
        continues = line.endswith("&")
        if continues:
            line = line[:-1].rstrip()
        buffer = f"{buffer} {line}".strip() if buffer else line
        continuing = continues
        if continuing:
            continue
        for statement in _scan_outside_quotes(buffer, ";"):
            normalized = _collapse_space_outside_quotes(statement)
            if normalized:
                logical_lines.append(normalized)
        buffer = ""

    if buffer:
        for statement in _scan_outside_quotes(buffer, ";"):
            normalized = _collapse_space_outside_quotes(statement)
            if normalized:
                logical_lines.append(normalized)
    return logical_lines


def _unique_sorted(values: list[str]) -> tuple[str, ...]:
    by_key = {}
    for value in values:
        by_key.setdefault(value.lower(), value)
    return tuple(by_key[key] for key in sorted(by_key))


def parse_fortran(path: Path, text: str) -> FortranFacts:
    """Extract lexical declarations and dependency edges from one source."""
    programs = []
    modules = []
    subroutines = []
    functions = []
    uses = []
    calls = []
    includes = []

    for statement in logical_fortran_lines(text):
        lowered = statement.lower()

        match = re.match(r"program\s+([a-z][a-z0-9_]*)\b", lowered)
        if match:
            programs.append(match.group(1))

        match = re.match(
            r"module\s+(?!procedure\b|subroutine\b|function\b)([a-z][a-z0-9_]*)\b",
            lowered,
        )
        if match:
            modules.append(match.group(1))

        if not re.match(r"end\s+subroutine\b", lowered):
            match = re.search(r"\bsubroutine\s+([a-z][a-z0-9_]*)\b", lowered)
            if match:
                subroutines.append(match.group(1))

        if not re.match(r"end\s+function\b", lowered):
            match = re.search(r"\bfunction\s+([a-z][a-z0-9_]*)\b", lowered)
            if match:
                functions.append(match.group(1))

        match = re.match(
            r"use(?:\s*,\s*[^:]*)?\s*(?:::)?\s*([a-z][a-z0-9_]*)\b",
            lowered,
        )
        if match:
            uses.append(match.group(1))

        calls.extend(
            re.findall(r"\bcall\s+([a-z][a-z0-9_]*)\b", lowered)
        )

        match = re.match(r"include\s*['\"]([^'\"]+)['\"]", statement, re.I)
        if not match:
            match = re.match(r"#\s*include\s*['\"]([^'\"]+)['\"]", statement, re.I)
        if match:
            includes.append(match.group(1))

    return FortranFacts(
        path=path.as_posix(),
        programs=_unique_sorted(programs),
        modules=_unique_sorted(modules),
        subroutines=_unique_sorted(subroutines),
        functions=_unique_sorted(functions),
        uses=_unique_sorted(uses),
        calls=_unique_sorted(calls),
        includes=_unique_sorted(includes),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _cmake_tokens(body: str) -> list[str]:
    uncommented = "\n".join(line.split("#", 1)[0] for line in body.splitlines())
    tokens = []
    for match in re.finditer(r'"([^"]*)"|([^\s()]+)', uncommented):
        tokens.append(match.group(1) if match.group(1) is not None else match.group(2))
    return tokens


def _normalize_cmake_source(raw_path: str) -> str | None:
    if raw_path.startswith("${") and "}/" not in raw_path:
        return None
    expanded = raw_path.replace("${CMAKE_CURRENT_SOURCE_DIR}", "src")
    if expanded.startswith("${"):
        return None
    if not expanded.startswith(("src/", "src_gpu/", "../", "/")):
        expanded = f"src/{expanded}"
    elif expanded.startswith("../"):
        expanded = f"src/{expanded}"
    normalized = posixpath.normpath(expanded)
    if normalized.startswith("/"):
        return None
    return normalized


def _parse_cmake_membership_with_errors(text: str) -> tuple[dict[str, str], list[str]]:
    membership = {}
    errors = []
    labels = {"ASTR_SOURCES": "astr CPU", "ASTR_GPU_SOURCES": "astr CUDA"}
    for variable, label in labels.items():
        match = re.search(
            rf"set\s*\(\s*{variable}\b(.*?)\)", text, flags=re.I | re.S
        )
        if not match:
            continue
        for token in _cmake_tokens(match.group(1)):
            path = _normalize_cmake_source(token)
            if path is None or Path(path).suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if path in membership:
                errors.append(f"duplicate astr target membership: {path}")
            else:
                membership[path] = label
    return membership, errors


def parse_cmake_membership(text: str) -> dict[str, str]:
    """Return normalized `astr` target membership from src/CMakeLists.txt."""
    membership, errors = _parse_cmake_membership_with_errors(text)
    if errors:
        raise ValueError("; ".join(errors))
    return membership


def source_baseline(repo: Path) -> str:
    """Return the newest commit affecting audited source or build paths."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "log",
            "-1",
            "--format=%H",
            "--",
            "CMakeLists.txt",
            "src",
            "src_gpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    baseline = result.stdout.strip()
    if not baseline:
        raise RuntimeError("no tracked source baseline found")
    return baseline


def _include_owner_map(facts: Sequence[FortranFacts]) -> dict[str, tuple[str, ...]]:
    tracked_paths = {fact.path for fact in facts}
    owners: dict[str, list[str]] = {}
    for fact in facts:
        source_dir = Path(fact.path).parent
        for included in fact.includes:
            candidate = (source_dir / included).as_posix()
            candidate = posixpath.normpath(candidate)
            if candidate in tracked_paths:
                owners.setdefault(candidate, []).append(fact.path)
    return {
        path: tuple(sorted(set(path_owners)))
        for path, path_owners in sorted(owners.items())
    }


def _snapshot_fingerprint(
    repo: Path, facts: Sequence[FortranFacts], cmake_paths: Sequence[Path]
) -> str:
    digest = hashlib.sha256()
    for fact in facts:
        digest.update(fact.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fact.sha256.encode("ascii"))
        digest.update(b"\n")
    for path in cmake_paths:
        if not (repo / path).is_file():
            continue
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256((repo / path).read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_snapshot(repo: Path) -> tuple[InventorySnapshot, list[str]]:
    """Collect the deterministic inventory and structural errors."""
    paths = tracked_core_files(repo)
    facts = tuple(
        parse_fortran(path, (repo / path).read_text(encoding="utf-8", errors="replace"))
        for path in paths
    )
    cmake_path = repo / "src" / "CMakeLists.txt"
    cmake_text = cmake_path.read_text(encoding="utf-8", errors="replace")
    membership, errors = _parse_cmake_membership_with_errors(cmake_text)
    include_owners = _include_owner_map(facts)

    final_membership = {}
    for fact in facts:
        suffix = Path(fact.path).suffix.lower()
        if suffix == ".inc":
            owners = include_owners.get(fact.path, ())
            if owners:
                final_membership[fact.path] = "include: " + ", ".join(owners)
            else:
                errors.append(f"tracked include has no owner: {fact.path}")
            continue
        if fact.path not in membership:
            errors.append(f"production source missing from astr target: {fact.path}")
        else:
            final_membership[fact.path] = membership[fact.path]

    snapshot = InventorySnapshot(
        baseline=source_baseline(repo),
        facts=facts,
        memberships=tuple(sorted(final_membership.items())),
        include_owners=tuple(sorted(include_owners.items())),
        fingerprint=_snapshot_fingerprint(
            repo, facts, (Path("CMakeLists.txt"), Path("src/CMakeLists.txt"))
        ),
    )
    return snapshot, sorted(set(errors))


def validate_source_anchors(
    markdown: str, facts_by_path: Mapping[str, FortranFacts]
) -> list[str]:
    """Validate machine-readable `path::symbol` Markdown anchors."""
    errors = []
    for path, symbol in SOURCE_ANCHOR.findall(markdown):
        fact = facts_by_path.get(path)
        if fact is None:
            errors.append(f"source anchor path not found: {path}::{symbol}")
            continue
        declarations = {
            item.lower()
            for group in (
                fact.programs,
                fact.modules,
                fact.subroutines,
                fact.functions,
            )
            for item in group
        }
        if symbol.lower() not in declarations:
            errors.append(f"source anchor symbol not found: {path}::{symbol}")
    return sorted(set(errors))


def _format_items(values: Sequence[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values).replace("|", "\\|")


def _responsibility(
    fact: FortranFacts, include_owners: Mapping[str, tuple[str, ...]]
) -> str:
    if fact.programs:
        return "定义 program " + ", ".join(f"`{name}`" for name in fact.programs)
    if fact.modules:
        return "定义 module " + ", ".join(f"`{name}`" for name in fact.modules)
    owners = include_owners.get(fact.path, ())
    if owners:
        return "由 " + ", ".join(f"`{owner}`" for owner in owners) + " include"
    return "无顶层 program/module 声明的实现单元"


def render_inventory(snapshot: InventorySnapshot) -> str:
    """Render one deterministic Markdown inventory."""
    membership = dict(snapshot.memberships)
    include_owners = dict(snapshot.include_owners)
    cpu_count = sum(value == "astr CPU" for value in membership.values())
    gpu_count = sum(value == "astr CUDA" for value in membership.values())
    include_count = sum(value.startswith("include:") for value in membership.values())
    lines = [
        "# ASTR Core Source Inventory",
        "",
        "> 此文件由 `scripts/maintenance/audit_source_inventory.py --write` 生成。",
        "> 架构含义以人工审阅的维护文档为准。",
        "",
        f"- Source baseline: `{snapshot.baseline}`",
        f"- Source fingerprint: `{snapshot.fingerprint}`",
        f"- Production sources: `{len(snapshot.facts)}`",
        f"- CMake membership: CPU `{cpu_count}`, CUDA `{gpu_count}`, include `{include_count}`",
        "",
        "## Scanner Boundary",
        "",
        "该清单只分析 Git 已跟踪文件和词法关系。generic interface、procedure pointer、",
        "preprocessor runtime branch 与动态分派需要人工审阅，不能由本清单推断为确定调用图。",
        "",
        "## Source Index",
        "",
        "| Source | Responsibility | Build membership | Procedures | Uses | Calls | Includes |",
        "|---|---|---|---|---|---|---|",
    ]
    for fact in snapshot.facts:
        procedures = fact.subroutines + fact.functions
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{fact.path}`",
                    _responsibility(fact, include_owners),
                    membership.get(fact.path, "missing"),
                    _format_items(procedures),
                    _format_items(fact.uses),
                    _format_items(fact.calls),
                    _format_items(fact.includes),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _maintenance_markdown(repo: Path) -> list[Path]:
    root = repo / "documents" / "maintenance"
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    snapshot, errors = build_snapshot(repo)
    rendered = render_inventory(snapshot)
    destination = repo / INVENTORY_PATH
    if args.check:
        current = destination.read_text(encoding="utf-8") if destination.is_file() else ""
        if current != rendered:
            errors.append("generated source inventory is stale")
        facts_by_path = {fact.path: fact for fact in snapshot.facts}
        for markdown_path in _maintenance_markdown(repo):
            errors.extend(
                validate_source_anchors(
                    markdown_path.read_text(encoding="utf-8"), facts_by_path
                )
            )

    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"wrote {INVENTORY_PATH} ({len(snapshot.facts)} sources)")
    else:
        print(f"source inventory is current ({len(snapshot.facts)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
