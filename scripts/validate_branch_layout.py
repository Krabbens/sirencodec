#!/usr/bin/env python3
"""Validate the repository layout and branch manifest."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "branch.md"

ALLOWED_STATUSES = {"canonical", "thesis", "experimental", "archived"}
REQUIRED_MANIFEST_KEYS = {
    "branch",
    "status",
    "base",
    "used_in_thesis",
    "install",
    "test",
}
ALLOWED_ROOT_ENTRIES = {
    ".dockerignore",
    ".gitattributes",
    ".github",
    ".gitignore",
    ".pre-commit-config.yaml",
    "CMakeLists.txt",
    "Dockerfile",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "Makefile",
    "README.md",
    "archive",
    "configs",
    "cpp",
    "docs",
    "overleaf",
    "package-lock.json",
    "package.json",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements.txt",
    "scripts",
    "src",
    "tests",
    "tools",
    "uv.lock",
}
BLOCKED_COMPONENTS = {
    "artifacts",
    "c++outputs",
    "data",
    "experiments",
    "runs",
}
BLOCKED_SUFFIXES = {
    ".asd",
    ".ckpt",
    ".flac",
    ".mp3",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".wav",
}
LOCAL_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def parse_manifest() -> dict[str, str]:
    if not MANIFEST.is_file():
        raise ValueError("missing docs/branch.md")
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("docs/branch.md must start with YAML-style metadata")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("docs/branch.md metadata is not closed") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"invalid manifest metadata line: {line!r}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def validate_manifest(metadata: dict[str, str]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_KEYS - metadata.keys())
    if missing:
        errors.append("manifest missing keys: " + ", ".join(missing))
    if metadata.get("status") not in ALLOWED_STATUSES:
        errors.append(
            "manifest status must be one of: "
            + ", ".join(sorted(ALLOWED_STATUSES))
        )
    if metadata.get("used_in_thesis") not in {"yes", "no"}:
        errors.append("manifest used_in_thesis must be yes or no")

    expected_branch = os.environ.get("GITHUB_REF_NAME")
    if expected_branch and not expected_branch.startswith(("pull/", "refs/")):
        if metadata.get("branch") != expected_branch:
            errors.append(
                f"manifest branch is {metadata.get('branch')!r}, "
                f"expected {expected_branch!r}"
            )
    return errors


def validate_paths(paths: list[str]) -> list[str]:
    errors: list[str] = []
    root_entries = {PurePosixPath(path).parts[0] for path in paths}
    unexpected = sorted(root_entries - ALLOWED_ROOT_ENTRIES)
    if unexpected:
        errors.append("unexpected root entries: " + ", ".join(unexpected))

    for path_text in paths:
        path = PurePosixPath(path_text)
        parts_lower = {part.lower() for part in path.parts}
        if parts_lower & BLOCKED_COMPONENTS:
            errors.append(f"blocked generated directory: {path_text}")
        if any(part.lower().startswith("infer_") for part in path.parts[:-1]):
            errors.append(f"blocked inference output directory: {path_text}")
        lower_name = path.name.lower()
        if any(lower_name.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
            errors.append(f"blocked generated file: {path_text}")
        if lower_name.endswith(".tar.gz"):
            errors.append(f"blocked dataset archive: {path_text}")
    return errors


def validate_readme_links() -> list[str]:
    readme = ROOT / "README.md"
    if not readme.is_file():
        return ["missing README.md"]
    errors: list[str] = []
    for target in LOCAL_LINK_RE.findall(readme.read_text(encoding="utf-8")):
        target = target.strip().split("#", 1)[0]
        if not target or "://" in target or target.startswith(("mailto:", "#")):
            continue
        if not (ROOT / target).exists():
            errors.append(f"README link does not exist: {target}")
    return errors


def main() -> int:
    errors: list[str] = []
    try:
        metadata = parse_manifest()
    except ValueError as exc:
        errors.append(str(exc))
        metadata = {}

    errors.extend(validate_manifest(metadata))
    errors.extend(validate_paths(tracked_files()))
    errors.extend(validate_readme_links())

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        "branch layout ok: "
        f"{metadata['branch']} ({metadata['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
