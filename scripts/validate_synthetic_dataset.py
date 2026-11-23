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
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root (for resolving dictionary data)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "src" / "paddle_ocr" / "dictionaries" / "data"


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

    required_fields = ["test_name", "value", "loinc"]
    for i, test in enumerate(results):
        if not isinstance(test, dict):
            errors.append(f"{filename}: results[{i}] is not a dict")
            continue
        for field in required_fields:
            if field not in test:
                errors.append(f"{filename}: results[{i}] missing '{field}'")

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
        if loinc not in ref_ranges:
            continue

        ref = ref_ranges[loinc]
        # Skip ordinal or answer-set tests
        if ref.get("type") == "ordinal" or "answers" in ref:
            continue

        val_str = str(test.get("value", ""))
        if not val_str:
            continue

        try:
            # Handle European format: "1.234,56" -> 1234.56
            # Heuristic: if both . and , present, comma is decimal
            if "," in val_str and "." in val_str:
                numeric_val = float(val_str.replace(".", "").replace(",", "."))
            elif "," in val_str:
                numeric_val = float(val_str.replace(",", "."))
            else:
                numeric_val = float(val_str)
        except (ValueError, TypeError):
            continue  # Non-numeric value -- skip

        ref_min = ref.get("min")
        ref_max = ref.get("max")

        if ref_min is not None and ref_max is not None:
            # Allow generous tolerance for pathological profiles
            lower_bound = float(ref_min) * 0.1
            upper_bound = float(ref_max) * 10.0

            if numeric_val < lower_bound or numeric_val > upper_bound:
                warnings.append(
                    f"{filename}: LOINC {loinc} value {val_str} "
                    f"outside plausible range [{ref_min}, {ref_max}] "
                    f"(tolerance [{lower_bound:.2f}, {upper_bound:.2f}])"
                )

    return warnings


def check_jsonl_integrity(filepath: Path) -> tuple[int, list[str]]:
    """Validate a JSONL file: every line must be valid JSON.

    Returns (line_count, list_of_error_strings).
    """
    if not filepath.exists():
        return 0, [f"{filepath.name}: file not found"]

    errors: list[str] = []
    line_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            line_count += 1
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{filepath.name}:{i}: invalid JSON: {exc}")

    return line_count, errors


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
        settings = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
        version = settings.get("schema_version")
        if version is not None or (dataset_dir / "reports.jsonl").exists() or (dataset_dir / "normalized_ground_truth").exists():
            if str(_PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(_PROJECT_ROOT))
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

    gt_files = sorted(gt_dir.glob("*.json"))
    print(f"Found {len(gt_files)} ground truth files")

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
    schema_errors: list[str] = []
    plausibility_warnings: list[str] = []
    total_tests = 0
    tests_with_unit = 0

    for gt_file in gt_files:
        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                gt = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            schema_errors.append(f"{gt_file.name}: cannot read: {exc}")
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
            if loinc:
                loinc_counter[loinc] += 1

            test_name = test.get("test_name", "")
            if test_name:
                synonym_counter[test_name] += 1

            unit = test.get("unit", "")
            if unit:
                tests_with_unit += 1

            panel = test.get("panel", "")
            if panel:
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

    pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")) if img_dir.exists() else []

    small_pdfs = [f.name for f in pdf_files if f.stat().st_size < 1024]
    empty_pdfs = [f.name for f in pdf_files if f.stat().st_size == 0]

    # Check for orphaned ground truth (GT file exists but no PDF/image)
    gt_ids = {f.stem for f in gt_files}
    pdf_ids = {f.stem for f in pdf_files}
    img_ids = {f.stem for f in img_files}
    orphaned_gt = gt_ids - (pdf_ids | img_ids)

    file_integrity = {
        "pdfs": len(pdf_files),
        "images": len(img_files),
        "small_pdfs_under_1kb": len(small_pdfs),
        "empty_pdfs": len(empty_pdfs),
        "orphaned_ground_truth": len(orphaned_gt),
        "ground_truth_files": len(gt_files),
    }

    # ---------------------------------------------------------------
    # 7. Training data JSONL validation
    # ---------------------------------------------------------------
    jsonl_stats: dict[str, Any] = {}
    jsonl_errors: list[str] = []

    for jsonl_name in ["sft_train.jsonl", "dpo_train.jsonl", "grpo_train.jsonl"]:
        jsonl_path = dataset_dir / jsonl_name
        mode = jsonl_name.split("_")[0]
        if not jsonl_path.is_file() and settings.get(mode) is not True:
            jsonl_stats[jsonl_name] = {"present": False, "lines": 0, "errors": 0}
            continue
        count, errs = check_jsonl_integrity(jsonl_path)
        if count != len(gt_files):
            errs.append(f"{jsonl_name}: expected {len(gt_files)} report rows, found {count}")
        if settings.get(mode) is False:
            errs.append(f"{jsonl_name}: stream present despite disabled export")
        jsonl_stats[jsonl_name] = {"lines": count, "errors": len(errs)}
        jsonl_errors.extend(errs)

    # ---------------------------------------------------------------
    # 8. Determine pass/fail
    # ---------------------------------------------------------------
    if schema_errors:
        failures.append(f"{len(schema_errors)} schema error(s)")
    if empty_pdfs:
        failures.append(f"{len(empty_pdfs)} empty PDF(s)")
    if small_pdfs:
        failures.append(f"{len(small_pdfs)} PDF(s) under 1KB")
    if jsonl_errors:
        failures.append(f"{len(jsonl_errors)} JSONL parse error(s)")

    if settings.get("count", len(gt_files)) != len(gt_files):
        failures.append("Report count differs from legacy metadata")
    fmt = settings.get("format")
    if ((fmt in ("both", "pdf") and gt_ids != pdf_ids)
            or (fmt in ("both", "jpg") and gt_ids != img_ids)):
        failures.append("Legacy output format and media IDs disagree")
    if not gt_files or not total_tests or orphaned_gt:
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
    if plausibility_warnings:
        warnings_list.append(
            f"{len(plausibility_warnings)} plausibility warning(s)"
        )
    if orphaned_gt:
        warnings_list.append(
            f"{len(orphaned_gt)} ground truth file(s) with no matching PDF/image"
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
    print(f"  Ground truth files:  {len(gt_files)}")
    print(f"  Total test results:  {total_tests}")
    print(f"  Tests with unit:     {tests_with_unit}")
    print()
    print("  -- File Integrity --")
    print(f"  PDFs:                {len(pdf_files)}")
    print(f"  Images (JPG/PNG):    {len(img_files)}")
    if small_pdfs:
        print(f"  Small PDFs (<1KB):   {len(small_pdfs)}")
    if empty_pdfs:
        print(f"  Empty PDFs (0B):     {len(empty_pdfs)}")
    if orphaned_gt:
        print(f"  Orphaned GT files:   {len(orphaned_gt)}")

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
    print(f"  Schema errors:       {len(schema_errors)}")
    print(f"  Plausibility warns:  {len(plausibility_warnings)}")

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

        if schema_errors:
            print("\n  Schema errors (showing first 20):")
            for err in schema_errors[:20]:
                print(f"    {err}")
            if len(schema_errors) > 20:
                print(f"    ... and {len(schema_errors) - 20} more")

        if plausibility_warnings:
            print("\n  Plausibility warnings (showing first 20):")
            for warn in plausibility_warnings[:20]:
                print(f"    {warn}")
            if len(plausibility_warnings) > 20:
                print(f"    ... and {len(plausibility_warnings) - 20} more")

        if jsonl_errors:
            print("\n  JSONL errors (showing first 10):")
            for err in jsonl_errors[:10]:
                print(f"    {err}")

        if orphaned_gt:
            print("\n  Orphaned ground truth files (showing first 10):")
            for gt_id in sorted(orphaned_gt)[:10]:
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
        "total_files": len(gt_files),
        "total_tests": total_tests,
        "tests_with_unit": tests_with_unit,
        "loinc_coverage": loinc_coverage,
        "synonym_coverage": synonym_coverage,
        "file_integrity": file_integrity,
        "panel_distribution": dict(panel_counter.most_common()),
        "profile_distribution": dict(profile_counter.most_common()),
        "region_distribution": dict(region_counter.most_common()),
        "template_distribution": dict(template_counter.most_common()) if template_counter else {},
        "schema_errors": len(schema_errors),
        "schema_error_details": schema_errors[:50],
        "plausibility_warnings": len(plausibility_warnings),
        "plausibility_warning_details": plausibility_warnings[:50],
        "training_data": jsonl_stats,
        "jsonl_errors": len(jsonl_errors),
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
