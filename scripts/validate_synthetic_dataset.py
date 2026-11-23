#!/usr/bin/env python3
"""Validate a generated synthetic dataset for completeness and quality.

Schema v2 reconciles all manifest rows, output files, labels, terminology, training
exports and counts from disk. Older runs use the original structural checks.

Usage:
    python scripts/validate_synthetic_dataset.py --dataset /tmp/synth_test --verbose
    python scripts/validate_synthetic_dataset.py --dataset demo-run/ --min-loinc 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from heapq import nsmallest
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root (for resolving dictionary data)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "src" / "paddle_ocr" / "dictionaries" / "data"

if __package__:
    from .lib.dataset_integrity import decode_json, read_json
else:  # Direct invocation as python scripts/validate_synthetic_dataset.py.
    sys.path.insert(0, str(_PROJECT_ROOT))
    from scripts.lib.dataset_integrity import decode_json, read_json


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def load_dictionary_loincs(data_dir: Path = _DATA_DIR) -> set[str]:
    """Load all unique LOINC codes from the project dictionary."""
    dict_path = data_dir / "lab_tests_loinc_es.json"
    if not dict_path.exists():
        print(f"WARNING: Dictionary not found at {dict_path}")
        return set()

    with open(dict_path, "r", encoding="utf-8") as f:
        dict_data: dict[str, Any] = json.load(f)

    all_loincs: set[str] = set()
    for entry in dict_data.values():
        loinc = entry.get("loinc", "")
        if loinc:
            all_loincs.add(loinc)
    return all_loincs


def load_reference_ranges(data_dir: Path = _DATA_DIR) -> dict[str, Any]:
    """Load reference ranges from the project data directory."""
    ref_path = data_dir / "reference_ranges.json"
    if not ref_path.exists():
        print(f"WARNING: Reference ranges not found at {ref_path}")
        return {}

    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


class _Diagnostics:
    """Keep exact counts and the bounded sample published by the legacy report."""

    def __init__(self, limit: int = 50):
        self.count = 0
        self.examples: list[str] = []
        self.limit = limit

    def add(self, message: str) -> None:
        self.count += 1
        if len(self.examples) < self.limit:
            self.examples.append(message)

    def extend(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.add(message)


def check_ground_truth_schema(gt: dict[str, Any], filename: str) -> list[str]:
    """Validate ground truth JSON schema. Returns list of error strings."""
    errors: list[str] = []

    if not isinstance(gt, dict) or "results" not in gt:
        errors.append(f"{filename}: missing 'results' key")
        return errors

    results = gt["results"]
    if not isinstance(results, list) or not results:
        errors.append(f"{filename}: 'results' must be a nonempty list")
        return errors

    for i, test in enumerate(results):
        if not isinstance(test, dict):
            errors.append(f"{filename}: results[{i}] is not a dict")
            continue
        for field in ("test_name", "loinc", "value"):
            if field not in test:
                errors.append(f"{filename}: results[{i}] missing '{field}'")
                continue
            value = test[field]
            if isinstance(value, str) and value.strip():
                continue
            if field == "value" and (type(value) is int or (type(value) is float and math.isfinite(value))):
                continue
            errors.append(f"{filename}: results[{i}] invalid '{field}'")
        for field in ("unit", "panel"):
            if test.get(field) is not None and not isinstance(test[field], str):
                errors.append(f"{filename}: results[{i}] invalid '{field}'")

    return errors


def check_value_plausibility(
    gt: dict[str, Any],
    ref_ranges: dict[str, Any],
    filename: str,
) -> list[str]:
    """Check that numeric values fall within plausible bounds.

    Uses a tenfold tolerance around the simulation bounds to allow
    for pathological values while catching egregious generation errors.

    Returns list of warning strings.
    """
    warnings: list[str] = []
    results = gt.get("results", []) if isinstance(gt, dict) else []
    if not isinstance(results, list):
        return warnings

    for test in results:
        if not isinstance(test, dict):
            continue

        loinc = test.get("loinc", "")
        if not isinstance(loinc, str) or loinc not in ref_ranges:
            continue

        ref = ref_ranges[loinc]
        # Skip ordinal or answer-set tests
        if ref.get("type") == "ordinal" or "answers" in ref:
            continue

        val_str = str(test.get("value", ""))
        if not val_str:
            continue

        try:
            # Mixed separators identify the decimal by its final position.
            # A single point remains decimal for historical normalized JSON.
            if "," in val_str and "." in val_str:
                if val_str.rfind(",") > val_str.rfind("."):
                    numeric_val = float(val_str.replace(".", "").replace(",", "."))
                else:
                    numeric_val = float(val_str.replace(",", ""))
            elif "," in val_str:
                numeric_val = float(val_str.replace(",", "."))
            else:
                numeric_val = float(val_str)
        except (ValueError, TypeError):
            continue  # Non-numeric value -- skip

        if not math.isfinite(numeric_val):
            warnings.append(f"{filename}: LOINC {loinc} has a non-finite numeric value")
            continue
        ref_min = ref.get("min")
        ref_max = ref.get("max")

        if ref_min is not None and ref_max is not None:
            # Allow generous tolerance for pathological profiles
            low, high = float(ref_min), float(ref_max)
            lower_bound = low * (10.0 if low < 0 else 0.1)
            upper_bound = high * (0.1 if high < 0 else 10.0)

            if numeric_val < lower_bound or numeric_val > upper_bound:
                warnings.append(
                    f"{filename}: LOINC {loinc} value {val_str} "
                    f"outside plausible range [{ref_min}, {ref_max}] "
                    f"(tolerance [{lower_bound:.2f}, {upper_bound:.2f}])"
                )

    return warnings


def _scan_jsonl(filepath: Path, report_error: Callable[[str], None]) -> int:
    if not filepath.is_file():
        report_error(f"{filepath.name}: file not found")
        return 0

    count = 0
    with filepath.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            count += 1
            try:
                record = decode_json(line.decode("utf-8"))
            except ValueError as exc:
                report_error(f"{filepath.name}:{number}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                report_error(f"{filepath.name}:{number}: expected a JSON object")
    return count


def check_jsonl_integrity(filepath: Path) -> tuple[int, list[str]]:
    """Return the number of nonblank JSONL records and all structural errors."""
    errors: list[str] = []
    count = _scan_jsonl(filepath, errors.append)
    return count, errors


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------


def validate_dataset(
    dataset_dir: Path,
    min_loinc: int = 0,
    verbose: bool = False,
    data_dir: Path = _DATA_DIR,
    verify_source: bool = False,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Run all validation checks on a dataset directory.

    Args:
        dataset_dir: Path to the dataset root directory.
        min_loinc: Minimum number of times each LOINC should appear.
        verbose: If True, print detailed error lists.

    Returns:
        Validation report dict.
    """
    if min_loinc < 0:
        return {"passed": False, "error": "Minimum coverage cannot be negative"}
    metadata = dataset_dir / "metadata.json"
    try:
        settings = read_json(metadata) if metadata.is_file() else {}
        version = settings.get("schema_version")
        if version is not None or (dataset_dir / "reports.jsonl").exists() or (dataset_dir / "normalized_ground_truth").exists():
            from scripts.lib.dataset_integrity import check_run
            report = check_run(dataset_dir, data_dir, min_loinc, verify_source)
            if expected_count is not None and report["reports"] != expected_count:
                raise ValueError(f"Expected {expected_count} reports, verified {report['reports']}")
            print(json.dumps(report, indent=2))
            return report
        if verify_source or expected_count is not None:
            raise ValueError("Source verification and expected counts require a schema-v2 run")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, RuntimeError) as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return {"passed": False, "error": str(exc)}
    report: dict[str, Any] = {"dataset": ".", "passed": True}
    failures: list[str] = []
    warnings_list: list[str] = []

    # ---------------------------------------------------------------
    # 0. Basic directory checks
    # ---------------------------------------------------------------
    gt_dir = dataset_dir / "ground_truth"
    if not gt_dir.exists():
        print(f"ERROR: {gt_dir} not found")
        report["passed"] = False
        report["error"] = "ground_truth directory not found"
        return report

    # One id table drives both the deterministic label scan and media matching.
    # Bits 1 and 2 mark a PDF and an image respectively.
    with os.scandir(gt_dir) as entries:
        media_by_id = {entry.name[:-5]: 0 for entry in entries if entry.name.endswith(".json") and entry.is_file()}
    report_count = len(media_by_id)
    print(f"Found {report_count} ground truth files")

    # ---------------------------------------------------------------
    # 1. Load reference data
    # ---------------------------------------------------------------
    all_loincs = load_dictionary_loincs(data_dir)
    ref_ranges = load_reference_ranges(data_dir)
    if min_loinc and not all_loincs:
        failures.append("Cannot verify minimum coverage without a dictionary")

    # ---------------------------------------------------------------
    # 2. Scan ground truth files
    # ---------------------------------------------------------------
    loinc_counter: Counter[str] = Counter()
    synonym_counter: Counter[str] = Counter()
    profile_counter: Counter[str] = Counter()
    region_counter: Counter[str] = Counter()
    template_counter: Counter[str] = Counter()
    panel_counter: Counter[str] = Counter()
    schema_errors = _Diagnostics()
    plausibility_warnings = _Diagnostics()
    total_tests = 0
    tests_with_unit = 0

    for report_id in sorted(media_by_id):
        gt_file = gt_dir / f"{report_id}.json"
        try:
            gt = read_json(gt_file)
        except (ValueError, OSError) as exc:
            schema_errors.add(f"{gt_file.name}: cannot read: {exc}")
            continue

        # Schema check
        s_errors = check_ground_truth_schema(gt, gt_file.name)
        schema_errors.extend(s_errors)

        # Plausibility check
        p_warnings = check_value_plausibility(gt, ref_ranges, gt_file.name)
        plausibility_warnings.extend(p_warnings)

        # Aggregate stats
        results = gt.get("results", []) if isinstance(gt, dict) else []
        if not isinstance(results, list):
            continue

        for test in results:
            if not isinstance(test, dict):
                continue
            total_tests += 1

            loinc = test.get("loinc", "")
            if isinstance(loinc, str) and loinc:
                loinc_counter[loinc] += 1

            test_name = test.get("test_name", "")
            if isinstance(test_name, str) and test_name:
                synonym_counter[test_name] += 1

            unit = test.get("unit", "")
            if isinstance(unit, str) and unit:
                tests_with_unit += 1

            panel = test.get("panel", "")
            if isinstance(panel, str) and panel:
                panel_counter[panel] += 1

    # ---------------------------------------------------------------
    # 3. Load coverage_report.json for profile/region/template info
    # ---------------------------------------------------------------
    coverage_path = dataset_dir / "coverage_report.json"
    if coverage_path.exists():
        with open(coverage_path, "r", encoding="utf-8") as f:
            coverage_data = json.load(f)
        profile_counter = Counter(coverage_data.get("profiles", {}))
        region_counter = Counter(coverage_data.get("regions", {}))
        template_counter = Counter(coverage_data.get("templates", {}))
    else:
        # Fallback: no profile/region data from coverage_report
        pass

    # ---------------------------------------------------------------
    # 4. LOINC coverage analysis
    # ---------------------------------------------------------------
    covered_loincs = set(loinc_counter.keys()) & all_loincs
    uncovered = all_loincs - set(loinc_counter.keys())
    under_min = {
        code: loinc_counter[code] for code in all_loincs if loinc_counter[code] < min_loinc
    }

    loinc_coverage = {
        "dictionary_loincs": len(all_loincs),
        "covered": len(covered_loincs),
        "uncovered_count": len(uncovered),
        "coverage_pct": round(
            len(covered_loincs) / max(len(all_loincs), 1) * 100, 2
        ),
        "min_count": min((loinc_counter[code] for code in all_loincs), default=0),
        "max_count": max((loinc_counter[code] for code in all_loincs), default=0),
        "mean_count": round(
            sum(loinc_counter[code] for code in all_loincs) / max(len(all_loincs), 1), 2
        ),
        "under_min_loinc_count": len(under_min),
    }

    # ---------------------------------------------------------------
    # 5. Synonym coverage analysis
    # ---------------------------------------------------------------
    top_synonyms = synonym_counter.most_common(10)
    bottom_synonyms = synonym_counter.most_common()[:-11:-1] if len(synonym_counter) >= 10 else []

    synonym_coverage = {
        "unique_synonyms": len(synonym_counter),
        "total_synonym_uses": sum(synonym_counter.values()),
        "top_10": [{"name": name, "count": count} for name, count in top_synonyms],
        "bottom_10": [{"name": name, "count": count} for name, count in bottom_synonyms],
    }

    # ---------------------------------------------------------------
    # 6. File integrity checks
    # ---------------------------------------------------------------
    pdf_dir = dataset_dir / "pdfs"
    img_dir = dataset_dir / "images"

    pdf_count = img_count = small_pdfs = empty_pdfs = 0
    extra_pdf = extra_image = False
    if pdf_dir.exists():
        with os.scandir(pdf_dir) as entries:
            for entry in entries:
                if not entry.name.endswith(".pdf") or not entry.is_file():
                    continue
                pdf_count += 1
                size = entry.stat().st_size
                small_pdfs += size < 1024
                empty_pdfs += size == 0
                report_id = entry.name[:-4]
                if report_id in media_by_id:
                    media_by_id[report_id] |= 1
                else:
                    extra_pdf = True
    if img_dir.exists():
        with os.scandir(img_dir) as entries:
            for entry in entries:
                if not entry.name.endswith((".jpg", ".png")) or not entry.is_file():
                    continue
                img_count += 1
                report_id = entry.name[:-4]
                if report_id in media_by_id:
                    media_by_id[report_id] |= 2
                else:
                    extra_image = True
    orphaned_gt = sum(mask == 0 for mask in media_by_id.values())
    file_integrity = {
        "pdfs": pdf_count,
        "images": img_count,
        "small_pdfs_under_1kb": small_pdfs,
        "empty_pdfs": empty_pdfs,
        "orphaned_ground_truth": orphaned_gt,
        "ground_truth_files": report_count,
    }

    # ---------------------------------------------------------------
    # 7. Training data JSONL validation
    # ---------------------------------------------------------------
    jsonl_stats: dict[str, Any] = {}
    jsonl_errors = _Diagnostics(limit=10)

    for jsonl_name in ["sft_train.jsonl", "dpo_train.jsonl", "grpo_train.jsonl"]:
        jsonl_path = dataset_dir / jsonl_name
        mode = jsonl_name.split("_")[0]
        if not jsonl_path.is_file() and settings.get(mode) is not True:
            jsonl_stats[jsonl_name] = {"present": False, "lines": 0, "errors": 0}
            continue
        before = jsonl_errors.count
        count = _scan_jsonl(jsonl_path, jsonl_errors.add)
        if count != report_count:
            jsonl_errors.add(f"{jsonl_name}: expected {report_count} report rows, found {count}")
        if settings.get(mode) is False:
            jsonl_errors.add(f"{jsonl_name}: stream present despite disabled export")
        jsonl_stats[jsonl_name] = {"lines": count, "errors": jsonl_errors.count - before}

    # ---------------------------------------------------------------
    # 8. Determine pass/fail
    # ---------------------------------------------------------------
    if schema_errors.count:
        failures.append(f"{schema_errors.count} schema error(s)")
    if empty_pdfs:
        failures.append(f"{empty_pdfs} empty PDF(s)")
    if small_pdfs:
        failures.append(f"{small_pdfs} PDF(s) under 1KB")
    if jsonl_errors.count:
        failures.append(f"{jsonl_errors.count} JSONL parse error(s)")

    if settings.get("count", report_count) != report_count:
        failures.append("Report count differs from legacy metadata")
    fmt = settings.get("format")
    if ((fmt in ("both", "pdf") and (extra_pdf or any(not mask & 1 for mask in media_by_id.values())))
            or (fmt in ("both", "jpg") and (extra_image or any(not mask & 2 for mask in media_by_id.values())))):
        failures.append("Legacy output format and media IDs disagree")
    if not report_count or not total_tests or orphaned_gt:
        failures.append("Missing report, result or media files")
    if min_loinc and (under_min or uncovered):
        failures.append(f"Minimum coverage {min_loinc} was not reached")

    # Warnings (do not cause failure)
    if len(covered_loincs) < len(all_loincs) * 0.01:
        warnings_list.append(
            f"Very low LOINC coverage: {len(covered_loincs)}/{len(all_loincs)} "
            f"({loinc_coverage['coverage_pct']}%)"
        )
    if under_min and len(covered_loincs) > len(all_loincs) * 0.5:
        # Only warn about under-min if coverage is substantial
        warnings_list.append(
            f"{len(under_min)} LOINCs appear fewer than {min_loinc} times"
        )
    if plausibility_warnings.count:
        warnings_list.append(
            f"{plausibility_warnings.count} plausibility warning(s)"
        )
    if orphaned_gt:
        warnings_list.append(
            f"{orphaned_gt} ground truth file(s) with no matching PDF/image"
        )

    passed = len(failures) == 0
    report["passed"] = passed

    # ---------------------------------------------------------------
    # 9. Print summary
    # ---------------------------------------------------------------
    sep = "=" * 62
    print(f"\n{sep}")
    print("  LEGACY STRUCTURAL VALIDATION")
    print(sep)
    print(f"  Dataset:             {dataset_dir}")
    print(f"  Ground truth files:  {report_count}")
    print(f"  Total test results:  {total_tests}")
    print(f"  Tests with unit:     {tests_with_unit}")
    print()
    print("  -- File Integrity --")
    print(f"  PDFs:                {pdf_count}")
    print(f"  Images (JPG/PNG):    {img_count}")
    if small_pdfs:
        print(f"  Small PDFs (<1KB):   {small_pdfs}")
    if empty_pdfs:
        print(f"  Empty PDFs (0B):     {empty_pdfs}")
    if orphaned_gt:
        print(f"  Orphaned GT files:   {orphaned_gt}")

    print()
    print("  -- LOINC Coverage --")
    print(f"  Dictionary LOINCs:   {loinc_coverage['dictionary_loincs']}")
    print(f"  Covered:             {loinc_coverage['covered']} ({loinc_coverage['coverage_pct']}%)")
    print(f"  Uncovered:           {loinc_coverage['uncovered_count']}")
    print(f"  Min count:           {loinc_coverage['min_count']}")
    print(f"  Max count:           {loinc_coverage['max_count']}")
    print(f"  Mean count:          {loinc_coverage['mean_count']}")
    if under_min:
        print(f"  Under min ({min_loinc}x):     {len(under_min)}")

    print()
    print("  -- Synonym Coverage --")
    print(f"  Unique synonyms:     {synonym_coverage['unique_synonyms']}")
    print(f"  Total uses:          {synonym_coverage['total_synonym_uses']}")
    if top_synonyms:
        print("  Top 10:")
        for name, count in top_synonyms:
            display = name[:45] + "..." if len(name) > 48 else name
            print(f"    {count:>5}x  {display}")
    if bottom_synonyms:
        print("  Bottom 10:")
        for name, count in bottom_synonyms:
            display = name[:45] + "..." if len(name) > 48 else name
            print(f"    {count:>5}x  {display}")

    print()
    print("  -- Panel Distribution --")
    if panel_counter:
        for panel, count in panel_counter.most_common(15):
            print(f"    {count:>5}x  {panel}")
        if len(panel_counter) > 15:
            print(f"    ... and {len(panel_counter) - 15} more panels")

    print()
    print("  -- Profile Distribution --")
    if profile_counter:
        for profile, count in profile_counter.most_common():
            print(f"    {count:>5}x  {profile}")
    else:
        print("    (no profile data available)")

    print()
    print("  -- Region Distribution --")
    if region_counter:
        for region, count in region_counter.most_common():
            print(f"    {count:>5}x  {region}")
    else:
        print("    (no region data available)")

    if template_counter:
        print()
        print("  -- Template Distribution --")
        for template, count in template_counter.most_common(20):
            print(f"    {count:>5}x  {template}")
        if len(template_counter) > 20:
            print(f"    ... and {len(template_counter) - 20} more templates")

    print()
    print("  -- Schema Validation --")
    print(f"  Schema errors:       {schema_errors.count}")
    print(f"  Plausibility warns:  {plausibility_warnings.count}")

    print()
    print("  -- Training Data (JSONL) --")
    for jsonl_name, stats in jsonl_stats.items():
        status = "OK" if stats["errors"] == 0 else f"{stats['errors']} errors"
        exists = (dataset_dir / jsonl_name).exists()
        if exists:
            size_kb = (dataset_dir / jsonl_name).stat().st_size / 1024
            print(f"  {jsonl_name:24s} {stats['lines']:>6} lines  {size_kb:>8.1f} KB  [{status}]")
        else:
            print(f"  {jsonl_name:24s}   (not found)")

    # Verbose output
    if verbose:
        if uncovered and len(uncovered) <= 100:
            print(f"\n  Uncovered LOINCs ({len(uncovered)}):")
            for loinc in sorted(uncovered)[:30]:
                print(f"    {loinc}")
            if len(uncovered) > 30:
                print(f"    ... and {len(uncovered) - 30} more")
        elif uncovered:
            print(f"\n  Uncovered LOINCs: {len(uncovered)} (too many to list)")

        if schema_errors.count:
            print("\n  Schema errors (showing first 20):")
            for err in schema_errors.examples[:20]:
                print(f"    {err}")
            if schema_errors.count > 20:
                print(f"    ... and {schema_errors.count - 20} more")

        if plausibility_warnings.count:
            print("\n  Plausibility warnings (showing first 20):")
            for warn in plausibility_warnings.examples[:20]:
                print(f"    {warn}")
            if plausibility_warnings.count > 20:
                print(f"    ... and {plausibility_warnings.count - 20} more")

        if jsonl_errors.count:
            print("\n  JSONL errors (showing first 10):")
            for err in jsonl_errors.examples[:10]:
                print(f"    {err}")

        if orphaned_gt:
            print("\n  Orphaned ground truth files (showing first 10):")
            for gt_id in nsmallest(10, (key for key, mask in media_by_id.items() if mask == 0)):
                print(f"    {gt_id}")

    # Final verdict
    print()
    print(sep)
    if failures:
        for fail in failures:
            print(f"  FAIL: {fail}")
    if warnings_list:
        for warn in warnings_list:
            print(f"  WARNING: {warn}")

    if passed:
        print("  RESULT: VALIDATION PASSED")
    else:
        print("  RESULT: VALIDATION FAILED")
    print(sep)

    # ---------------------------------------------------------------
    # 10. Build report dict
    # ---------------------------------------------------------------
    report.update({
        "total_files": report_count,
        "total_tests": total_tests,
        "tests_with_unit": tests_with_unit,
        "loinc_coverage": loinc_coverage,
        "synonym_coverage": synonym_coverage,
        "file_integrity": file_integrity,
        "panel_distribution": dict(panel_counter.most_common()),
        "profile_distribution": dict(profile_counter.most_common()),
        "region_distribution": dict(region_counter.most_common()),
        "template_distribution": dict(template_counter.most_common()) if template_counter else {},
        "schema_errors": schema_errors.count,
        "schema_error_details": schema_errors.examples[:50],
        "plausibility_warnings": plausibility_warnings.count,
        "plausibility_warning_details": plausibility_warnings.examples[:50],
        "training_data": jsonl_stats,
        "jsonl_errors": jsonl_errors.count,
        "failures": failures,
        "warnings": warnings_list,
        "passed": passed,
    })

    # Write report
    report_path = dataset_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a generated synthetic dataset for completeness and quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/validate_synthetic_dataset.py --dataset /tmp/synth_test --verbose\n"
            "  python scripts/validate_synthetic_dataset.py --dataset demo-run/ --min-loinc 5\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset directory to validate",
    )
    parser.add_argument(
        "--min-loinc",
        type=int,
        default=0,
        help="Required minimum across eligible codes (default: the run setting)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error lists and uncovered LOINCs",
    )
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Input data used by this generator run")
    parser.add_argument("--verify-source", action="store_true", help="Require the exact generator source hashes recorded by a schema-v2 run")
    parser.add_argument("--expected-count", type=int, help="Require exactly this many schema-v2 reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns 0 on pass, 1 on fail."""
    args = parse_args(argv)
    dataset = Path(args.dataset)

    if not dataset.exists():
        print(f"ERROR: Dataset directory not found: {dataset}")
        return 1

    if args.expected_count is not None and args.expected_count <= 0:
        print("ERROR: --expected-count must be positive", file=sys.stderr)
        return 1

    try:
        report = validate_dataset(
            dataset_dir=dataset, min_loinc=args.min_loinc, verbose=args.verbose,
            data_dir=args.data_dir, verify_source=args.verify_source, expected_count=args.expected_count,
        )
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1

    return 0 if report.get("passed", False) else 1


if __name__ == "__main__":
    sys.exit(main())
