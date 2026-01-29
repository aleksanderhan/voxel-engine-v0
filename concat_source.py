#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_INCLUDE_EXTS = {".rs", ".wgsl"}


def to_posix_rel(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    return f"{base.name}/{rel.as_posix()}"


def comment_prefix_for(path: Path) -> str:
    # Rust + WGSL both accept // line comments.
    return "//"


def parse_csv_list(value: str) -> list[str]:
    """
    Parse comma-separated list entries.

    Entries may be:
      - A directory name (no '/'): matches that directory name anywhere under --src
        Example: "ray" matches ".../shaders/ray/..." and ".../ray/..."
      - A nested relative path (contains '/'): matches that subtree path under --src
        Example: "shaders/ray" matches only ".../<src>/shaders/ray/..."
      - "." meaning the src root itself.

    Backslashes are normalized to '/'.
    Leading/trailing slashes are ignored.
    """
    if not value:
        return []
    items: list[str] = []
    for part in value.split(","):
        s = part.strip()
        if not s:
            continue
        s = s.replace("\\", "/").strip("/")
        items.append(s if s else ".")
    return items


def split_names_and_paths(items: list[str]) -> tuple[set[str], list[str]]:
    """
    Split items into:
      - names: entries with no '/' and not '.', matched by directory basename anywhere
      - paths: entries with '/' or '.', matched as relative subtree under --src
    """
    names: set[str] = set()
    paths: list[str] = []
    for it in items:
        if it == "." or "/" in it:
            paths.append(it)
        else:
            names.add(it)
    return names, paths


def rel_posix_dir_for(p: Path, src_dir: Path) -> str:
    """
    Returns the relative directory (POSIX) to use for subtree matching:
      - if p is a dir: its relative posix path (or "." if root)
      - if p is a file: its parent dir relative posix path (or "." if at root)
    """
    rel = p.relative_to(src_dir)
    if p.is_dir():
        s = rel.as_posix()
        return s if s else "."
    parent = rel.parent.as_posix()
    return parent if parent else "."


def is_under_any_path(rel_dir_posix: str, roots: list[str]) -> bool:
    """
    True if rel_dir_posix is exactly a root, or is inside it (root/...).
    """
    for r in roots:
        if r == ".":
            return True
        if rel_dir_posix == r or rel_dir_posix.startswith(r + "/"):
            return True
    return False


def has_any_ancestor_named(p: Path, src_dir: Path, names: set[str]) -> bool:
    """
    True if any ancestor directory (under src_dir) has basename in `names`.
    For files, checks parent directories. For directories, checks itself + parents.
    """
    if not names:
        return False
    rel_parts = p.relative_to(src_dir).parts
    parts = rel_parts[:-1] if p.is_file() else rel_parts
    return any(part in names for part in parts)


def is_excluded(p: Path, src_dir: Path, exclude_names: set[str], exclude_paths: list[str]) -> bool:
    """
    Exclusion matches if:
      - p is in/under any excluded relative path root, OR
      - p has any ancestor directory whose basename is in exclude_names
    """
    rel_dir = rel_posix_dir_for(p, src_dir)
    return is_under_any_path(rel_dir, exclude_paths) or has_any_ancestor_named(p, src_dir, exclude_names)


def is_included(p: Path, src_dir: Path, include_names: set[str], include_paths: list[str]) -> bool:
    """
    Inclusion matches if:
      - no include filter is provided: everything is included (subject to excludes), OR
      - p is in/under any included relative path root, OR
      - p has any ancestor directory whose basename is in include_names
    """
    if not include_names and not include_paths:
        return True
    rel_dir = rel_posix_dir_for(p, src_dir)
    return is_under_any_path(rel_dir, include_paths) or has_any_ancestor_named(p, src_dir, include_names)


def collect_files(
    src_dir: Path,
    include_exts: set[str],
    include_names: set[str],
    include_paths: list[str],
    exclude_names: set[str],
    exclude_paths: list[str],
) -> list[Path]:
    import os

    files_set: set[Path] = set()

    # Walk roots optimization:
    # - If include_paths is present and include_names is empty, we can walk only those subtrees.
    # - If include_names is present (e.g. "ray"), we must walk all of src_dir to discover matches.
    if include_paths and not include_names:
        walk_roots: list[Path] = []
        for d in include_paths:
            root = src_dir if d == "." else (src_dir / Path(d))
            root = root.resolve()
            try:
                root.relative_to(src_dir)
            except ValueError:
                raise ValueError(f"--include entry escapes --src: {d}")
            if root.exists() and root.is_dir():
                walk_roots.append(root)
        if not walk_roots:
            return []
    else:
        walk_roots = [src_dir]

    for start in walk_roots:
        for root, dirs, filenames in os.walk(start):
            root_path = Path(root)

            # Prune excluded directories (prevents descending into them)
            kept_dirs: list[str] = []
            for d in dirs:
                dp = root_path / d
                # dp is under src_dir by construction of os.walk(start) where start is under src_dir
                if not is_excluded(dp, src_dir, exclude_names, exclude_paths):
                    kept_dirs.append(d)
            dirs[:] = kept_dirs

            # Collect matching files
            for name in filenames:
                fp = root_path / name
                if fp.suffix not in include_exts:
                    continue
                if is_excluded(fp, src_dir, exclude_names, exclude_paths):
                    continue
                if not is_included(fp, src_dir, include_names, include_paths):
                    continue
                files_set.add(fp)

    return sorted(files_set, key=lambda x: x.as_posix())


def concatenate_sources(
    src_dir: Path,
    out_file: Path,
    *,
    include_exts: set[str],
    include_names: set[str],
    include_paths: list[str],
    exclude_names: set[str],
    exclude_paths: list[str],
    encoding: str = "utf-8",
) -> None:
    src_dir = src_dir.resolve()
    out_file = out_file.resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    source_files = collect_files(
        src_dir,
        include_exts=include_exts,
        include_names=include_names,
        include_paths=include_paths,
        exclude_names=exclude_names,
        exclude_paths=exclude_paths,
    )

    # Avoid including the output file if it lives under src_dir
    source_files = [p for p in source_files if p.resolve() != out_file]

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding=encoding, newline="\n") as out:
        for p in source_files:
            header_path = to_posix_rel(p, src_dir)
            prefix = comment_prefix_for(p)

            out.write(f"{prefix} {header_path}\n")
            out.write(f"{prefix} {'-' * len(header_path)}\n")

            try:
                text = p.read_text(encoding=encoding)
            except UnicodeDecodeError:
                text = p.read_bytes().decode(encoding, errors="replace")

            if text and not text.endswith("\n"):
                text += "\n"
            out.write(text)
            out.write("\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate selected source files under --src into one file, with path headers."
    )
    parser.add_argument("--src", default="src", help="Source directory to traverse (default: src).")
    parser.add_argument("--out", default="all_sources.txt", help="Output file path (default: all_sources.txt).")

    parser.add_argument(
        "--include",
        default="",
        help=(
            "Comma-separated directories to include. "
            "Use a bare name (no '/') to match that directory name anywhere under --src. "
            "Use a nested path (with '/') to match a specific subtree under --src. "
            "If provided, ONLY included directories are considered (then --exclude still applies). "
            "Examples: --include ray,shaders/wgsl"
        ),
    )
    parser.add_argument(
        "--exclude",
        default="",
        help=(
            "Comma-separated directories to exclude. "
            "Use a bare name (no '/') to match that directory name anywhere under --src. "
            "Use a nested path (with '/') to match a specific subtree under --src. "
            "Examples: --exclude target,tests/fixtures"
        ),
    )

    args = parser.parse_args(argv)

    include_items = parse_csv_list(args.include)
    exclude_items = parse_csv_list(args.exclude)

    include_names, include_paths = split_names_and_paths(include_items)
    exclude_names, exclude_paths = split_names_and_paths(exclude_items)

    try:
        concatenate_sources(
            Path(args.src),
            Path(args.out),
            include_exts=set(DEFAULT_INCLUDE_EXTS),
            include_names=include_names,
            include_paths=include_paths,
            exclude_names=exclude_names,
            exclude_paths=exclude_paths,
            encoding="utf-8",
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
