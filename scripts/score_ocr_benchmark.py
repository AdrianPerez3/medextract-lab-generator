"""Rescore exact printed values and their row associations in the MedExtract OCR benchmark."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import BadZipFile, ZipFile

if __package__:
    from .score_row_associations import score
else:
    from score_row_associations import score


def archive_bytes(archive, name):
    # Only scoring metadata is needed. Never extract archive paths or open images.
    info = archive.getinfo(name)
    if info.file_size > 16 * 1024 * 1024:
        raise ValueError(f"{name}: scoring metadata exceeds 16 MiB")
    return archive.read(info)


def report_records(lines, source):
    records = {}
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{source}, line {number}: {exc}") from exc
        report_id = row.get("report_id") if isinstance(row, dict) else None
        if not isinstance(report_id, str) or not report_id.strip():
            raise ValueError(f"{source}, line {number}: missing report_id")
        if report_id in records:
            raise ValueError(f"{source}, line {number}: duplicate report {report_id}")
        records[report_id] = row
    if not records:
        raise ValueError(f"{source}: expected at least one report")
    return records


def load_labels(archive, row, field):
    name = row.get(field)
    if not isinstance(name, str) or "\\" in name:
        raise ValueError(f"{row['report_id']}: invalid {field} path")
    parts = name.split("/")
    if len(parts) != 2 or parts[0] != field or not parts[1].endswith(".json"):
        raise ValueError(f"{row['report_id']}: expected {field}/<filename>.json")
    return archive_bytes(archive, name)


def score_method(labels, manifest, documents):
    if manifest.keys() != documents.keys():
        missing = sorted(manifest.keys() - documents.keys())
        extra = sorted(documents.keys() - manifest.keys())
        raise ValueError(
            f"prediction report IDs differ: missing={missing}, extra={extra}"
        )

    has_review_counts = "review_count" in next(iter(documents.values()))
    for report_id, doc in documents.items():
        if ("review_count" in doc) != has_review_counts:
            raise ValueError("provide review_count for either all reports or none")
        if has_review_counts and (
            type(doc["review_count"]) is not int or doc["review_count"] < 0
        ):
            raise ValueError(f"{report_id}: review_count must be a nonnegative integer")

    exact = submitted = expected = reviews = 0
    for report_id, (coordinates, ground_truth) in labels.items():
        doc = documents[report_id]
        try:
            result = score(coordinates, ground_truth, doc, "image-pixels")
        except ValueError as exc:
            raise ValueError(f"{report_id}: {exc}") from exc
        exact += result["association_and_exact_value"]["matched_rows"]
        submitted += result["submitted_predictions"]
        expected += result["expected_rows"]
        reviews += doc.get("review_count", 0)

    return {
        "reports": len(labels),
        "exact": exact,
        "submitted": submitted,
        "expected": expected,
        "review_entries": reviews if has_review_counts else None,
        "precision": exact / submitted if submitted else None,
        "recall": exact / expected,
        "f1": 2 * exact / (submitted + expected),
    }


def benchmark(path, predictions=None, draw=None, condition=None):
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate ZIP member name")
        manifest = report_records(
            archive_bytes(archive, "reports.jsonl").splitlines(), "reports.jsonl"
        )
        labels = {}
        for report_id, row in manifest.items():
            if draw and row.get("draw") != draw:
                continue
            if condition and row.get("condition") != condition:
                continue
            coordinates = json.loads(load_labels(archive, row, "coordinates"))
            if (
                not isinstance(coordinates, dict)
                or coordinates.get("report_id") != report_id
            ):
                raise ValueError(f"{report_id}: coordinate report_id mismatch")
            labels[report_id] = coordinates, load_labels(archive, row, "ground_truth")
        if not labels:
            raise ValueError("no reports match the requested draw and condition")

        if predictions is not None:
            with predictions.open("rb") as stream:
                docs = report_records(stream, predictions.name)
            return {predictions.stem: score_method(labels, manifest, docs)}

        results = {}
        for name in sorted(names):
            parts = name.split("/")
            if (
                len(parts) == 2
                and parts[0] == "predictions"
                and parts[1].endswith(".jsonl")
            ):
                docs = report_records(archive_bytes(archive, name).splitlines(), name)
                results[parts[1].removesuffix(".jsonl")] = score_method(
                    labels, manifest, docs
                )
        if not results:
            raise ValueError("benchmark has no predictions/*.jsonl streams")
        return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Outputs JSON. Each submitted point pair counts toward precision. "
        "Each distinct expected row can earn credit once. Review entries count proposals.",
    )
    parser.add_argument("archive", type=Path, help="downloaded benchmark ZIP")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="score a local JSONL instead of the bundled streams",
    )
    parser.add_argument("--draw", help="select one draw from reports.jsonl")
    parser.add_argument("--condition", help="select one condition from reports.jsonl")
    args = parser.parse_args()
    try:
        result = benchmark(args.archive, args.predictions, args.draw, args.condition)
    except (OSError, KeyError, ValueError, OverflowError, BadZipFile) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
