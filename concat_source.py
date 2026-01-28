#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import fnmatch


DEFAULT_INCLUDE_EXTS = {".rs", ".wgsl"}


def normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ext
    return ext if ext.startswith(".") else f".{ext}"


def to_posix_rel(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    return f"{base.name}/{rel.as_posix()}"


def comment_prefix_for(path: Path) -> str:
    # Rust + WGSL both accept // line comments.
    return "//"


def should_skip_path(
    p: Path,
    src_dir: Path,
    exclude_dirnames: set[str],
    exclude_path_globs: list[str],
) -> bool:
    # Skip if any parent directory name matches excluded dirnames
    # (also skips the directory itself when p is a dir).
    parts = p.relative_to(src_dir).parts
    for part in parts[:-1] if p.is_file() else parts:
        if part in exclude_dirnames:
            return True

    # Skip if the relative posix path matches any excluded glob pattern.
    # Note: patterns are matched against paths relative to src_dir, POSIX-style.
    rel_posix = p.relative_to(src_dir).as_posix()
    for pat in exclude_path_globs:
        if fnmatch.fnmatch(rel_posix, pat):
            return True

    return False


def collect_files(
    src_dir: Path,
    include_exts: set[str],
    exclude_dirnames: set[str],
    exclude_path_globs: list[str],
) -> list[Path]:
    files: list[Path] = []

    # Manual walk so we can prune excluded directories early.
    for root, dirs, filenames in __import__("os").walk(src_dir):
        root_path = Path(root)

        # Prune dirs in-place (prevents descending into them)
        kept_dirs: list[str] = []
        for d in dirs:
            dp = root_path / d
            if not should_skip_path(dp, src_dir, exclude_dirnames, exclude_path_globs):
                kept_dirs.append(d)
        dirs[:] = kept_dirs

        # Collect files
        for name in filenames:
            fp = root_path / name
            if fp.suffix in include_exts and not should_skip_path(
                fp, src_dir, exclude_dirnames, exclude_path_globs
            ):
                files.append(fp)

    files.sort(key=lambda x: x.as_posix())
    return files


def concatenate_sources(
    src_dir: Path,
    out_file: Path,
    *,
    include_exts: set[str],
    exclude_dirnames: set[str],
    exclude_path_globs: list[str],
    encoding: str = "utf-8",
) -> None:
    src_dir = src_dir.resolve()
    out_file = out_file.resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    source_files = collect_files(
        src_dir,
        include_exts=include_exts,
        exclude_dirnames=exclude_dirnames,
        exclude_path_globs=exclude_path_globs,
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
    parser.add_argument("--encoding", default="utf-8", help="Text encoding to use (default: utf-8).")

    parser.add_argument(
        "--include-ext",
        action="append",
        default=[],
        help="Extension to include (repeatable). If omitted, defaults to: .rs, .wgsl",
    )
    parser.add_argument(
        "--exclude-ext",
        action="append",
        default=[],
        help="Extension to exclude (repeatable). Example: --exclude-ext .wgsl",
    )

    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to skip anywhere under --src (repeatable). Example: --exclude-dir target",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=[],
        help=(
            "Glob pattern (relative to --src, POSIX style) to skip (repeatable). "
            "Examples: --exclude-path 'tests/**' --exclude-path '**/generated/**'"
        ),
    )

    args = parser.parse_args(argv)

    # Build include set
    if args.include_ext:
        include_exts = {normalize_ext(e) for e in args.include_ext if e.strip()}
    else:
        include_exts = set(DEFAULT_INCLUDE_EXTS)

    # Apply excludes to include set
    exclude_exts = {normalize_ext(e) for e in args.exclude_ext if e.strip()}
    include_exts -= exclude_exts

    exclude_dirnames = {d.strip() for d in args.exclude_dir if d.strip()}
    exclude_path_globs = [p.strip() for p in args.exclude_path if p.strip()]

    try:
        concatenate_sources(
            Path(args.src),
            Path(args.out),
            include_exts=include_exts,
            exclude_dirnames=exclude_dirnames,
            exclude_path_globs=exclude_path_globs,
            encoding=args.encoding,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
