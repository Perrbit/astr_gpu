#!/usr/bin/env python3
"""Build a lexical inventory of tracked ASTR core solver sources."""

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


SOURCE_SUFFIXES = {".f90", ".cuf", ".inc"}


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
