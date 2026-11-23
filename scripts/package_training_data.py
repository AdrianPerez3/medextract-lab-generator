#!/usr/bin/env python3
"""Split and package synthetic training data for VLM fine-tuning.

Reads the monolithic JSONL files (sft_train.jsonl, dpo_train.jsonl,
grpo_train.jsonl) produced by the synthetic generator, shuffles them
deterministically, and splits into train/val/test partitions.

Usage:
    python scripts/package_training_data.py --dataset /tmp/synth_test --split 80,10,10
    python scripts/package_training_data.py --dataset demo-run/ --split 90,5,5 --seed 123
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .lib.dataset_output import staged_output
else:
    from lib.dataset_output import staged_output


# ---------------------------------------------------------------------------
# JSONL processing utilities
# ---------------------------------------------------------------------------


def read_jsonl(filepath: Path) -> list[str]:
    """Read a JSONL file, returning non-empty lines with trailing newlines stripped.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of raw JSON line strings (without trailing newline).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    lines: list[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.strip():
                lines.append(stripped)
    return lines


def _invalid_constant(value: str):
    raise ValueError(f"Invalid JSON number: {value}")


def validate_jsonl_lines(lines: list[str], source_name: str) -> list[str]:
    """Validate that every line is valid JSON.

    Args:
        lines: List of JSON line strings.
        source_name: Human-readable name for error messages.

    Returns:
        List of error strings (empty if all lines are valid).
    """
    errors: list[str] = []
    for i, line in enumerate(lines, start=1):
        try:
            obj = json.loads(line, parse_constant=_invalid_constant)
            if not isinstance(obj, dict):
                raise ValueError("Expected a JSON object")
        except ValueError as exc:
            errors.append(f"{source_name}:{i}: {exc}")
    return errors


def split_lines(
    lines: list[str],
    train_pct: float,
    val_pct: float,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Shuffle and split lines into train/val/test partitions.

    The test partition receives whatever remains after train and val.

    Args:
        lines: Input lines to split.
        train_pct: Training fraction (0-100).
        val_pct: Validation fraction (0-100).
        seed: Random seed for deterministic shuffling.

    Returns:
        Dict with keys "train", "val", "test", each mapping to a list of lines.
    """
    rng = random.Random(seed)
    shuffled = list(lines)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_pct / 100.0)
    n_val = int(total * val_pct / 100.0)
    # Remainder goes to test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_jsonl(filepath: Path, lines: list[str]) -> int:
    """Write lines to a JSONL file.

    Args:
        filepath: Output path.
        lines: List of JSON line strings.

    Returns:
        Number of lines written.
    """
    with open(filepath, "x", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
    return len(lines)


# ---------------------------------------------------------------------------
# Split one JSONL file into train/val/test
# ---------------------------------------------------------------------------


def split_jsonl_file(
    source_path: Path,
    output_dir: Path,
    prefix: str,
    train_pct: float,
    val_pct: float,
    seed: int,
) -> dict[str, Any] | None:
    """Split a single JSONL file into train/val/test partitions.

    Creates:
        output_dir/{prefix}_train.jsonl
        output_dir/{prefix}_val.jsonl
        output_dir/{prefix}_test.jsonl

    Args:
        source_path: Path to the source JSONL file.
        output_dir: Directory for output files.
        prefix: Filename prefix (e.g. "sft", "dpo", "grpo").
        train_pct: Training percentage (0-100).
        val_pct: Validation percentage (0-100).
        seed: Random seed.

    Returns:
        Stats dict, or None if the source file is missing.
    """
    if not source_path.exists():
        return None

    # Read all lines
    lines = read_jsonl(source_path)
    total = len(lines)

    if total == 0:
        raise ValueError(f"Empty training source: {source_path}")
    errors = validate_jsonl_lines(lines, source_path.name)
    if errors:
        raise ValueError(errors[0])
    if source_path.resolve().parent == output_dir.resolve():
        raise ValueError("Split output cannot overwrite its training source")

    # Split
    splits = split_lines(lines, train_pct, val_pct, seed)

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, dict[str, Any]] = {}

    for split_name, split_lines_data in splits.items():
        out_path = output_dir / f"{prefix}_{split_name}.jsonl"
        n_written = write_jsonl(out_path, split_lines_data)
        file_size = out_path.stat().st_size

        split_stats[split_name] = {
            "lines": n_written,
            "file": out_path.name,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 1),
        }

    return {
        "source": source_path.name,
        "total_lines": total,
        "splits": {k: v["lines"] for k, v in split_stats.items()},
        "split_details": split_stats,
        "validation_errors": len(errors),
        "validation_error_details": errors[:20] if errors else [],
    }


# ---------------------------------------------------------------------------
# Main packaging logic
# ---------------------------------------------------------------------------


def _json_digest(obj: Any) -> str:
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _manifest_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line, parse_constant=_invalid_constant)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{number}: expected a report")
            yield record


def _v2_splits(dataset: Path, metadata: dict, train_pct: float, val_pct: float, seed: int):
    """Link shared page bytes or exact visible/normalized targets into indivisible groups."""
    parents = {}
    owners = {}
    report_images = {}
    targets = {}

    def root(report):
        while parents[report] != report:
            parents[report] = parents[parents[report]]
            report = parents[report]
        return report

    def join(report, key):
        other = owners.setdefault(key, report)
        parents[root(report)] = root(other)

    for index, manifest in enumerate(_manifest_rows(dataset / "reports.jsonl")):
        rid = f"RPT_{index:06d}"
        if manifest["report_id"] != rid:
            raise ValueError(f"Expected manifest report {rid}")
        parents[rid] = rid
        pages = manifest["pages"]
        if type(pages) is not int or pages <= 0:
            raise ValueError(f"Invalid page count for {rid}")
        images = tuple(f"images/{rid}{'' if n == 1 else f'_p{n:03d}'}.jpg" for n in range(1, pages + 1))
        if manifest["images"] != list(images):
            raise ValueError(f"Unexpected page paths for {rid}")
        report_images[images] = rid
        for image in images:
            digest = _file_sha256(dataset / image)
            if manifest["sha256"][image] != digest:
                raise ValueError(f"Checksum mismatch: {image}")
            join(rid, ("page", digest))
        for field in ("ground_truth", "normalized_ground_truth"):
            path = f"{field}/{rid}.json"
            if manifest[field] != path:
                raise ValueError(f"Unexpected {field} path for {rid}")
            content = (dataset / path).read_bytes()
            if hashlib.sha256(content).hexdigest() != manifest["sha256"][path]:
                raise ValueError(f"Checksum mismatch: {path}")
            label = json.loads(content)
            if not isinstance(label, dict) or not isinstance(label.get("results"), list) or not label["results"]:
                raise ValueError(f"Invalid {field} for {rid}")
            digest = _json_digest(label)
            join(rid, (field, digest))
            if field == "ground_truth":
                targets[rid] = digest
    if not parents or len(parents) != metadata["count"]:
        raise ValueError("Report manifest count differs from metadata")
    groups = {}
    for rid in parents:
        groups.setdefault(root(rid), []).append(rid)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(parents) * train_pct / 100)
    n_val = int(len(parents) * val_pct / 100)
    result = {"train": [], "val": [], "test": []}
    used = 0
    for group in shuffled:
        split = "train" if used < n_train else "val" if used < n_train + n_val else "test"
        result[split].extend(group)
        used += len(group)
    return result, report_images, targets, len(groups)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_v2_stream(dataset, staging, image_prefix, kind, splits, image_ids, targets):
    """Index small byte offsets so streams can be shuffled without holding every target in RAM."""
    source = dataset / f"{kind}_train.jsonl"
    offsets = {}
    details = {}
    with source.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                raise ValueError(f"Blank training record in {source.name} at byte {offset}")
            record = json.loads(line, parse_constant=_invalid_constant)
            rid = image_ids.get(tuple(record["images"]))
            if rid is None or rid in offsets:
                raise ValueError(f"Unknown or duplicate report in {source.name} at byte {offset}")
            chosen = json.loads(record["solution"] if kind == "grpo" else record["messages"][-1]["content"])
            if _json_digest(chosen) != targets[rid]:
                raise ValueError(f"Training target differs from report {rid} in {source.name}")
            if kind == "dpo":
                rejected = json.loads(record["rejected_response"], parse_constant=_invalid_constant)
                if (not isinstance(rejected, dict) or not isinstance(rejected.get("results"), list)
                        or rejected == chosen):
                    raise ValueError(f"Invalid DPO negative for {rid}")
            offsets[rid] = offset
        if len(offsets) != len(image_ids):
            raise ValueError(f"Missing reports in {source.name}")
        for split, ids in splits.items():
            name = f"{kind}_{split}.jsonl"
            path = staging / name
            with path.open("x", encoding="utf-8") as output:
                for rid in ids:
                    stream.seek(offsets[rid])
                    record = json.loads(stream.readline(), parse_constant=_invalid_constant)
                    if image_ids.get(tuple(record["images"])) != rid:
                        raise ValueError(f"{source.name} changed while packaging {rid}")
                    record["images"] = [(image_prefix / image).as_posix() for image in record["images"]]
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            size = path.stat().st_size
            details[split] = {"file": name, "lines": len(ids),
                              "size_bytes": size, "size_kb": round(size / 1024, 1)}
    return {"source": source.name, "total_lines": len(offsets),
            "splits": {k: len(v) for k, v in splits.items()}, "split_details": details,
            "validation_errors": 0, "validation_error_details": []}


def package_dataset(dataset_dir: Path, output_dir: Path | None, train_pct: float,
                    val_pct: float, seed: int) -> dict[str, Any]:
    dataset = dataset_dir.resolve()
    destination = output_dir.absolute() if output_dir is not None else dataset / "splits"
    remainder = 100 - (train_pct + val_pct)
    if math.isclose(remainder, 0, rel_tol=0, abs_tol=1e-10):
        remainder = 0.0
    percentages = (train_pct, val_pct, remainder)
    if any(not math.isfinite(v) or v < 0 or v > 100 for v in percentages):
        raise ValueError("Split percentages must be finite, nonnegative and total 100")
    if not dataset.is_dir() or destination.resolve() == dataset:
        raise ValueError("Specify a dataset directory and a separate split destination")
    formats = ("sft", "dpo", "grpo")
    metadata_path = dataset / "metadata.json"
    meta = json.loads(metadata_path.read_bytes()) if metadata_path.exists() else {}
    schema = meta.get("schema_version", 1)
    if (dataset / "reports.jsonl").exists() and schema != 2:
        raise ValueError("A report manifest requires schema-v2 generator metadata")
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError(f"Unsupported generator schema: {schema}")
    present = [kind for kind in formats if (dataset / f"{kind}_train.jsonl").is_file()]
    if not present:
        raise ValueError("No SFT, DPO or GRPO training streams found")
    with staged_output(destination) as staging:
        results = {k: {"skipped": True, "reason": "file not found"} for k in formats if k not in present}
        extra = {}
        if schema == 2:
            if any(type(meta.get(k)) is not bool or meta[k] != (k in present) for k in formats):
                raise ValueError("Training streams disagree with generator metadata")
            splits, image_ids, targets, group_count = _v2_splits(dataset, meta, train_pct, val_pct, seed)
            # Split rows normally locate the images relative to their own folder.
            # Windows volumes have no relative path between them.
            try:
                image_prefix = Path(os.path.relpath(dataset, destination.resolve()))
                image_base = "directory containing the split JSONL"
            except ValueError:
                image_prefix = Path(".")
                image_base = "original dataset"
            for kind in present:
                results[kind] = _split_v2_stream(dataset, staging, image_prefix, kind, splits, image_ids, targets)
            with (staging / "report_splits.json").open("x", encoding="utf-8") as stream:
                json.dump(splits, stream, indent=2)
            extra = {"group_count": group_count, "assignment": "report_splits.json",
                     "grouping": "shared JPEG page bytes or exact visible/normalized target",
                     "image_path_base": image_base}
        else:
            for kind in present:
                stats = split_jsonl_file(dataset / f"{kind}_train.jsonl", staging, kind, train_pct, val_pct, seed)
                results[kind] = stats
            extra = {"grouping": "independent legacy stream rows", "image_path_base": "original dataset"}
        report = {
            "dataset": Path(os.path.relpath(dataset, destination.resolve())).as_posix()
                       if destination.resolve().is_relative_to(dataset) else None,
            "output_dir": ".",
            "split_pct": dict(zip(("train", "val", "test"), percentages)),
            "seed": seed, "total_lines": sum(v["total_lines"] for v in results.values() if "total_lines" in v),
            "total_validation_errors": 0, "formats": results, "all_valid": True, **extra,
        }
        with (staging / "packaging_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    for kind in present:
        print(f"{kind}: {results[kind]['splits']}")
    print(f"Saved splits to {destination}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split and package synthetic training data for VLM fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/package_training_data.py --dataset /tmp/synth_test --split 80,10,10\n"
            "  python scripts/package_training_data.py --dataset demo-run/ --split 90,5,5 --seed 123\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset directory containing JSONL files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="80,10,10",
        help="Comma-separated train,val,test percentages (default: 80,10,10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <dataset>/splits/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns 0 on success, 1 on error."""
    args = parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ERROR: Dataset directory not found: {dataset}")
        return 1

    # Parse split ratios
    try:
        parts = [float(x.strip()) for x in args.split.split(",")]
    except ValueError:
        print(f"ERROR: Invalid --split format: {args.split!r}. Expected: 80,10,10")
        return 1

    if len(parts) != 3:
        print(f"ERROR: --split must have exactly 3 values (train,val,test), got {len(parts)}")
        return 1

    train_pct, val_pct, test_pct = parts
    total_pct = train_pct + val_pct + test_pct

    if not math.isfinite(total_pct) or not math.isclose(total_pct, 100.0, abs_tol=1e-8):
        print(f"ERROR: Split percentages must sum to 100, got {total_pct}")
        return 1

    if any(p < 0 for p in parts):
        print("ERROR: Split percentages must be non-negative")
        return 1

    output_dir = Path(args.output) if args.output else None

    try:
        package_dataset(dataset, output_dir, train_pct, val_pct, args.seed)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
