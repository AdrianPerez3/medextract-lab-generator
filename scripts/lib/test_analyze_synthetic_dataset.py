"""Independent fixtures for measured schema-v2 coverage, without rendering media."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.analyze_synthetic_dataset import analyze, html_report


CODES = ["718-7", "777-3", "789-8"]
CLI = Path(__file__).resolve().parents[1] / "analyze_synthetic_dataset.py"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_dataset(path):
    (path / "ground_truth").mkdir()
    records = []
    for i, (codes, pages) in enumerate([(["718-7", "777-3"], 1), (["718-7"], 2)]):
        rid = f"RPT_{i:06d}"
        label = f"ground_truth/{rid}.json"
        write_json(
            path / label,
            {"results": [{"loinc": code, "value": "do-not-print"} for code in codes]},
        )
        records.append(
            {
                "report_id": rid,
                "ground_truth": label,
                "result_count": len(codes),
                "pages": pages,
                "images": [],
                "column_count": 5,
                "tier": None,
                "sha256": {
                    label: hashlib.sha256((path / label).read_bytes()).hexdigest()
                },
            }
        )
    (path / "reports.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    write_json(
        path / "metadata.json",
        {
            "schema_version": 2,
            "count": 2,
            "format": "pdf",
            "eligible_loincs_sha256": hashlib.sha256(
                "\n".join(CODES).encode()
            ).hexdigest(),
        },
    )
    write_json(
        path / "coverage_report.json",
        {
            "loinc_counts": dict(zip(CODES, [2, 1, 0])),
            "total_reports": 2,
            "total_pages": 3,
            "total_images": 0,
            "total_loincs_in_dict": 3,
            "unique_loincs_covered": 2,
        },
    )
    return records


def test_measures_zero_and_rare_separately(tmp_path):
    make_dataset(tmp_path)
    data = analyze(tmp_path, rare_at_most=1)
    assert (data["reports"], data["pages"], data["observations"]) == (2, 3, 3)
    assert data["loinc_counts"] == {"718-7": 2, "777-3": 1, "789-8": 0}
    assert data["missing_codes"] == ["789-8"] and data["rare_codes"] == ["777-3"]
    assert data["pages_per_report"] == {1: 1, 2: 1} and data[
        "observations_per_report"
    ] == {1: 1, 2: 1}
    assert "do-not-print" not in json.dumps(data)


@pytest.mark.parametrize(
    "change",
    [
        "tamper_label",
        "forged_count",
        "unrecorded_code",
        "path_escape",
        "wrong_universe",
        "duplicate_report",
        "page_mismatch",
        "invalid_json",
        "negative_threshold",
        "boolean_coverage",
        "wrong_totals",
    ],
)
def test_corruption_is_detected_without_publishing_html(tmp_path, change):
    records = make_dataset(tmp_path)
    label = records[0]["ground_truth"]
    source = tmp_path / label
    coverage_path = tmp_path / "coverage_report.json"
    coverage = json.loads(coverage_path.read_text())
    if change == "tamper_label":
        source.write_text(source.read_text() + " ")
    if change == "forged_count":
        coverage["loinc_counts"]["718-7"] = 3
    if change == "unrecorded_code":
        write_json(source, {"results": [{"loinc": "718-7"}, {"loinc": "999-9"}]})
        records[0]["sha256"][label] = hashlib.sha256(source.read_bytes()).hexdigest()
    if change == "path_escape":
        records[0]["ground_truth"] = "../other.json"
    if change == "wrong_universe":
        coverage["loinc_counts"]["1000-0"] = 0
    if change == "duplicate_report":
        records[1] = records[0]
    if change == "page_mismatch":
        records[0]["images"] = ["images/RPT_000000.jpg"]
    if change == "boolean_coverage":
        coverage["loinc_counts"]["777-3"] = True
    if change == "wrong_totals":
        coverage["total_pages"] = 4
    write_json(coverage_path, coverage)
    (tmp_path / "reports.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
        if change != "invalid_json"
        else "{"
    )
    out = tmp_path / "analysis.html"
    run = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--dataset",
            str(tmp_path),
            "--html",
            str(out),
            "--rare-at-most",
            "-1" if change == "negative_threshold" else "1",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        run.returncode != 0
        and not out.exists()
        and not run.stdout
        and "Traceback" not in run.stderr
    )


def test_html_cli_contains_visible_labels_and_preserves_existing_file(tmp_path):
    make_dataset(tmp_path)
    out = tmp_path / "result.html"
    args = [sys.executable, str(CLI), "--dataset", str(tmp_path), "--html", str(out)]
    run = subprocess.run(args, check=True, capture_output=True, text=True)
    assert json.loads(run.stdout)["observed_codes"] == 2
    html = out.read_text()
    assert "Pages per report" in html and "789-8" in html and "do-not-print" not in html
    assert "<script" not in html and "https://" not in html
    out.write_text("preserve me")
    retry = subprocess.run(args, capture_output=True)
    assert retry.returncode != 0 and out.read_text() == "preserve me"


def test_manifest_pages_and_tiers_in_image_runs(tmp_path):
    records = make_dataset(tmp_path)
    meta_file = tmp_path / "metadata.json"
    coverage_file = tmp_path / "coverage_report.json"
    meta = json.loads(meta_file.read_text())
    meta["format"] = "jpg"
    write_json(meta_file, meta)
    coverage = json.loads(coverage_file.read_text())
    coverage["total_images"] = 3
    write_json(coverage_file, coverage)
    records[0].update(tier=0, images=["images/RPT_000000.jpg"])
    records[1].update(
        tier=4, images=["images/RPT_000001.jpg", "images/RPT_000001_p002.jpg"]
    )
    manifest = tmp_path / "reports.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = analyze(tmp_path, rare_at_most=0)
    assert result["images"] == 3
    assert result["reports_by_image_tier"] == {0: 1, 4: 1}
    assert result["rare_codes"] == [] and result["missing_codes"] == ["789-8"]
    records[1]["images"].reverse()
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="out-of-order manifest image"):
        analyze(tmp_path)


def test_top_ten_share_counts_all_observations_and_resolves_ties(tmp_path):
    records = make_dataset(tmp_path)
    label = records[0]["ground_truth"]
    path = tmp_path / label
    target = json.loads(path.read_text())
    added = [f"{1000 + i}-0" for i in range(9)]
    target["results"].extend({"loinc": code} for code in added)
    write_json(path, target)
    records[0]["sha256"][label] = hashlib.sha256(path.read_bytes()).hexdigest()
    records[0]["result_count"] = 11
    (tmp_path / "reports.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    coverage_path = tmp_path / "coverage_report.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["loinc_counts"].update(dict.fromkeys(added, 1))
    coverage.update(unique_loincs_covered=11, total_loincs_in_dict=12)
    write_json(coverage_path, coverage)
    meta_path = tmp_path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["eligible_loincs_sha256"] = hashlib.sha256(
        "\n".join(sorted(added + CODES)).encode()
    ).hexdigest()
    write_json(meta_path, meta)
    summary = analyze(tmp_path)
    bottom_chart = (
        html_report(summary)
        .split("<h2>Least frequent LOINC codes</h2>")[1]
        .split("</section>")[0]
    )
    assert 'width="50.000"' in bottom_chart and 'width="100.000"' not in bottom_chart
    top = summary["top_10_concentration"]
    assert list(top["counts"]) == ["718-7", *added]
    assert (top["observations"], top["total_observations"], top["share_percent"]) == (
        11,
        12,
        91.67,
    )


@pytest.mark.parametrize(
    "destination",
    [
        "reports.jsonl",
        "metadata.json",
        "coverage_report.json",
        "ground_truth/RPT_000000.json",
        "alias.html",
    ],
)
def test_html_output_cannot_overwrite_dataset_files(tmp_path, destination):
    make_dataset(tmp_path)
    target = tmp_path / destination
    if destination == "alias.html":
        target.symlink_to(tmp_path / "ground_truth/RPT_000000.json")
    before = target.read_bytes()
    result = subprocess.run(
        [sys.executable, str(CLI), "--dataset", str(tmp_path), "--html", str(target)],
        capture_output=True,
    )
    assert result.returncode != 0 and not result.stdout
    assert target.read_bytes() == before
