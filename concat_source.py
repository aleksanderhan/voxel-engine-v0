#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


RS_EXT = ".rs"
WGSL_EXT = ".wgsl"
INCLUDE_EXTS = {RS_EXT, WGSL_EXT}


def to_posix_rel(path: Path, base: Path) -> str:
    # Relative to base, but include base folder name in the printed path
    rel = path.relative_to(base)
    # e.g. base=".../src", rel="renderer/mod.rs" -> "src/renderer/mod.rs"
    return f"{base.name}/{rel.as_posix()}"


def comment_prefix_for(path: Path) -> str:
    # Rust + WGSL both accept // line comments.
    return "//"


def collect_files(src_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix in INCLUDE_EXTS:
            files.append(p)
    files.sort(key=lambda x: x.as_posix())
    return files


def concatenate_sources(src_dir: Path, out_file: Path, encoding: str = "utf-8") -> None:
    src_dir = src_dir.resolve()
    out_file = out_file.resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    # Avoid including the output file if it lives under src_dir
    source_files = [p for p in collect_files(src_dir) if p.resolve() != out_file]

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
                # Fallback: read as bytes and decode with replacement to avoid crashing
                text = p.read_bytes().decode(encoding, errors="replace")

            # Ensure there's exactly one newline before the next header
            if text and not text.endswith("\n"):
                text += "\n"
            out.write(text)
            out.write("\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate all .rs and .wgsl files under src/ into one file, with path headers."
    )
    parser.add_argument(
        "--src",
        default="src",
        help="Source directory to traverse (default: src).",
    )
    parser.add_argument(
        "--out",
        default="all_sources.txt",
        help="Output file path (default: all_sources.txt).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding to use when reading/writing (default: utf-8).",
    )
    args = parser.parse_args(argv)

    try:
        concatenate_sources(Path(args.src), Path(args.out), encoding=args.encoding)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
