"""Insight Condenser

Scans a run directory for bot history JSON files (default pattern: 'bot-<id>-history.json'),
recursively extracts all values under a specified key (default: 'insight'), and writes them
into a single JSON file. Each extracted entry is annotated with the bot_id and the round
(index within that bot's history array, starting at 1).

Usage:
    python insight_condenser.py /path/to/run_dir

Examples:
    # Basic usage with defaults
    python insight_condenser.py ./runs/2025-08-25

    # Custom key and output file
    python insight_condenser.py ./runs --key insight --output insights.json

    # Match a different filename pattern and minify output
    python insight_condenser.py ./runs --pattern "bot-*-history.json" --minify
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


# ---------------------------- Configuration ---------------------------- #
DEFAULT_KEY = "insight"
DEFAULT_PATTERN = "bot-*-history.json"
DEFAULT_OUTPUT = "insight_condensed.json"


# ------------------------------ Data Model ----------------------------- #
@dataclass(frozen=True)
class InsightRecord:
    round: int
    bot_id: int
    value: Any

    def to_dict(self, key: str) -> Dict[str, Any]:
        return {"round": self.round, "bot_id": self.bot_id, key: self.value}


# ------------------------------ Utilities ------------------------------ #
BOT_FILE_REGEX = re.compile(r"^bot-(\d+)-history\.json$")


def extract_bot_id(filename: str) -> int | None:
    """Return numeric bot_id from a filename like 'bot-123-history.json', else None."""
    m = BOT_FILE_REGEX.match(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def iter_json_files(run_dir: Path, pattern: str) -> Iterator[Path]:
    """Yield files in run_dir matching the glob pattern (non-recursive)."""
    yield from sorted(run_dir.glob(pattern))


def find_key_recursive(obj: Any, key: str) -> Iterator[Any]:
    """Yield all values for `key` found anywhere within nested dict/list structures."""
    if isinstance(obj, dict):
        if key in obj:
            yield obj[key]
        for v in obj.values():
            yield from find_key_recursive(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from find_key_recursive(item, key)
    # primitives: nothing to do


def load_json_array(path: Path) -> List[Any]:
    """Load a JSON array from file, returning an empty list on structured errors with logs."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logging.warning("%s does not contain a JSON array; skipping.", path.name)
        return []
    except json.JSONDecodeError as e:
        logging.error("Failed to parse %s: %s", path.name, e)
        return []
    except OSError as e:
        logging.error("Failed to read %s: %s", path.name, e)
        return []


def collect_insights(
    run_dir: Path, key: str, pattern: str
) -> List[InsightRecord]:
    records: List[InsightRecord] = []

    files = list(iter_json_files(run_dir, pattern))
    if not files:
        logging.warning("No files matched pattern '%s' in %s", pattern, run_dir)
        return records

    # Filter to filenames that match the canonical bot-<id>-history.json format
    filtered: List[Path] = []
    for p in files:
        bid = extract_bot_id(p.name)
        if bid is None:
            logging.debug("Skipping non-bot history file: %s", p.name)
            continue
        filtered.append(p)

    # Sort deterministically by bot_id, then by filename
    filtered.sort(key=lambda p: (extract_bot_id(p.name) or 0, p.name))

    if not filtered:
        logging.warning("No files matched required format 'bot-<id>-history.json'.")
        return records

    for path in filtered:
        bot_id = extract_bot_id(path.name)
        assert bot_id is not None  # guarded above
        data = load_json_array(path)

        # Round is the 1-based index within THIS file's array
        for round_idx, obj in enumerate(data, start=1):
            for val in find_key_recursive(obj, key):
                records.append(InsightRecord(round=round_idx, bot_id=bot_id, value=val))

    return records


def write_output(records: Sequence[InsightRecord], key: str, out_path: Path, minify: bool) -> None:
    out = [r.to_dict(key) for r in records]
    try:
        with out_path.open("w", encoding="utf-8") as f:
            if minify:
                json.dump(out, f, separators=(",", ":"))
            else:
                json.dump(out, f, indent=4, ensure_ascii=False)
        logging.info("Wrote %d records to %s", len(out), out_path)
    except OSError as e:
        logging.error("Failed to write %s: %s", out_path, e)
        raise


# --------------------------------- CLI --------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Condense bot history JSON files by extracting all values under a given key"
        )
    )
    p.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing bot-*-history.json files",
    )
    p.add_argument(
        "-k",
        "--key",
        default=DEFAULT_KEY,
        help=f"Key to extract recursively (default: '{DEFAULT_KEY}')",
    )
    p.add_argument(
        "-p",
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern to match files (default: '{DEFAULT_PATTERN}')",
    )
    p.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"Output JSON filename (default: '{DEFAULT_OUTPUT}')",
    )
    p.add_argument(
        "--minify",
        action="store_true",
        help="Write compact JSON without indentation",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)",
    )
    return p


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    run_dir: Path = args.run_dir
    if not run_dir.exists() or not run_dir.is_dir():
        logging.error("Run directory not found or not a directory: %s", run_dir)
        return 2

    logging.info(
        "Scanning %s for pattern '%s' and key '%s'...",
        run_dir,
        args.pattern,
        args.key,
    )

    records = collect_insights(run_dir=run_dir, key=args.key, pattern=args.pattern)

    out_path = args.output if args.output.is_absolute() else run_dir / args.output
    try:
        write_output(records, key=args.key, out_path=out_path, minify=args.minify)
    except Exception:
        return 1

    # Print a succinct summary for CLI UX
    print(
        f"Extracted {len(records)} records for key '{args.key}' -> {out_path}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())