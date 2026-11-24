"""Golden paired reports with equal strings on neighboring rows and different OCR decisions."""

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.score_paired_ocr import compare, text_error


def pred(value, index, value_index=None):
    return {
        "page": 1,
        "test_point": [10, 20 * index + 10],
        "value_point": [30, 20 * (index if value_index is None else value_index) + 10],
        "value_text": value,
    }


def accepted(p):
    return {"page": 1, "status": "accepted", "selected": p, "baseline": p}


def review(p, alt=None):
    return {
        "page": 1,
        "status": "review",
        "reason": "disagreement",
        "selected": None,
        "baseline": p,
        "alternate": alt,
    }


def stream(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path


def corpus(folder, values, clean, copy):
    results = [{"value": v, "loinc": "X", "test_name": "Analyte"} for v in values]
    ground = json.dumps({"results": results}).encode()
    (folder / "ground_truth").mkdir()
    (folder / "coordinates").mkdir()
    (folder / "ground_truth/p.json").write_bytes(ground)
    manifest, data = [], []
    for view, decisions in [("clean", clean), ("photocopy", copy)]:
        rid = "p_" + view
        rows = []
        for i in range(len(results)):
            top = 10 * i
            rows.append(
                {
                    "result_index": i,
                    "loinc": "X",
                    "bbox": [0, top, 20, top + 10],
                    "cells": {
                        "test_name": [0, top, 10, top + 10],
                        "value": [10, top, 20, top + 10],
                    },
                }
            )
        height = len(rows) * 10
        page = {
            "page": 1,
            "rows": rows,
            "height": height,
            "width": 20,
            "image": {"path": f"images/{rid}.jpg", "height": height * 2, "width": 40},
        }
        coords = {
            "report_id": rid,
            "schema_version": 1,
            "coordinate_space": "pdf_points_top_left",
            "ground_truth_sha256": sha256(ground).hexdigest(),
            "pages": [page],
        }
        rel = f"coordinates/{rid}.json"
        (folder / rel).write_text(json.dumps(coords))
        manifest.append(
            {
                "report_id": rid,
                "pair_id": "p",
                "condition": view,
                "split": "test",
                "coordinates": rel,
                "ground_truth": "ground_truth/p.json",
            }
        )
        data.append(
            {
                "report_id": rid,
                "decisions": decisions,
                "predictions": [
                    d["selected"] for d in decisions if d["status"] == "accepted"
                ],
            }
        )
    return stream(folder / "reports.jsonl", manifest), stream(
        folder / "method.jsonl", reversed(data)
    )


def test_transitions_wrong_integer_controls_and_review_cost(tmp_path):
    values = ["4.4", "44", "1,60", "-2", "44"]
    clean = [accepted(pred(v, i)) for i, v in enumerate(values)]
    read = [
        accepted(pred("44", 0)),
        accepted(pred("44", 1)),
        review(pred("1.60", 2)),
        accepted(pred("-2", 3, 4)),
    ]
    manifest, method = corpus(tmp_path, values, clean, read)
    baseline = []
    for row in map(json.loads, method.read_text().splitlines()):
        if row["report_id"].endswith("photocopy"):
            p = pred("1.60", 2)
            row["decisions"][2] = accepted(p)
            row["predictions"] = [d["selected"] for d in row["decisions"]]
        baseline.append(row)
    result = compare(manifest, method, stream(tmp_path / "baseline.jsonl", baseline))
    assert result["report_pairs"] == 1 and result["paired_rows"] == 5
    assert result["transitions"]["exact"] == {
        "exact": 1,
        "wrong": 1,
        "review": 1,
        "missing": 2,
        "ambiguous": 0,
    }
    copy = result["views"]["photocopy"]
    assert copy["accepted_geometry"]["different_row"] == 1
    assert copy["wrong_text_classes"]["power_of_ten"] == 1
    assert copy["review_proposals_per_100_rows"] == 20
    assert copy["baseline_wrong_to_review_rows_per_100"] == 20
    assert copy["baseline_wrong_classes_to_review"] == {"decimal_mark": 1}
    assert result["rows"][0]["expected"] == "4.4"
    assert result["rows"][0]["photocopy"]["accepted"] == ["44"]
    assert result["rows"][2]["photocopy"]["reviews"][0]["sources"]["baseline"] == [
        "1.60"
    ]


def test_duplicate_and_ambiguous_reviews_do_not_hide_work_or_inflate_recall(tmp_path):
    clean = [accepted(pred("44", 0)), accepted(pred("44", 0)), accepted(pred("4.4", 0))]
    copy = [
        review(pred("44", 0), pred("46", 1)),
        review(pred("44", 0)),
        review(pred("44", 0)),
    ]
    manifest, method = corpus(tmp_path, ["44", "44"], clean, copy)
    result = compare(manifest, method)
    assert result["transitions"]["ambiguous"]["review"] == 1
    assert result["transitions"]["missing"]["missing"] == 1
    assert result["views"]["clean"]["accepted_predictions"] == 3
    assert result["views"]["photocopy"]["review_proposals"] == 3
    assert result["views"]["photocopy"]["unassigned_review_proposals"] == 1
    assert result["views"]["photocopy"]["states"]["review"] == 1


@pytest.mark.parametrize(
    ("expected", "read", "result"),
    [
        ("4.4", "44", "power_of_ten"),
        ("44", "4,40", "power_of_ten"),
        ("0.0", "0", "numeric_format"),
        ("1,60", "1.60", "decimal_mark"),
        ("−4.4", "4.4", "sign"),
        ("-1.8", "18", "sign_and_power_of_ten"),
        ("+40", "-4.00", "sign_and_power_of_ten"),
        ("+1.0", "1", "numeric_format"),
        ("4.4", "45", "other_numeric"),
        (">4.4", "44", "other_text"),
        ("4", "4", "exact"),
    ],
)
def test_exact_print_and_numeric_error_classes(expected, read, result):
    assert text_error(expected, read) == result


@pytest.mark.parametrize(
    "damage",
    [
        "missing_view",
        "split",
        "geometry",
        "ground",
        "outside",
        "duplicates",
        "group",
        "checksum",
        "inconsistent_selected",
    ],
)
def test_cli_rejects_inconsistent_pair_or_decision_stream(tmp_path, damage):
    manifest, method = corpus(tmp_path, ["4.4"], [accepted(pred("4.4", 0))], [])
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    doc = json.loads((tmp_path / records[1]["coordinates"]).read_text())
    if damage == "missing_view":
        records.pop()
        rows = [
            r
            for r in map(json.loads, method.read_text().splitlines())
            if r["report_id"] == "p_clean"
        ]
        stream(method, rows)
    elif damage == "split":
        records[1]["split"] = "train"
    elif damage == "group":
        records[1]["group_id"] = "another"
    elif damage == "checksum":
        records[1]["sha256"] = {records[1]["coordinates"]: "0" * 64}
    elif damage == "geometry":
        doc["pages"][0]["rows"][0]["cells"]["test_name"][2] = 9
    elif damage == "ground":
        ground = json.dumps({"results": [{"loinc": "X", "value": "44"}]}).encode()
        records[1]["ground_truth"] = "ground_truth/q.json"
        (tmp_path / records[1]["ground_truth"]).write_bytes(ground)
        doc["ground_truth_sha256"] = sha256(ground).hexdigest()
    elif damage == "outside":
        records[1]["ground_truth"] = "../private.json"
    elif damage == "duplicates":
        records[1]["condition"] = "clean"
    else:
        stream(
            method,
            [
                dict(r, predictions=[])
                for r in map(json.loads, method.read_text().splitlines())
            ],
        )
    if len(records) == 2:
        (tmp_path / records[1]["coordinates"]).write_text(json.dumps(doc))
    stream(manifest, records)
    cli = Path(__file__).resolve().parents[1] / "score_paired_ocr.py"
    run = subprocess.run(
        [sys.executable, cli, "--manifest", manifest, "--decisions", method],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2 and "Traceback" not in run.stderr and not run.stdout


def test_cli_replays_without_images_or_optional_libraries(tmp_path):
    manifest, method = corpus(
        tmp_path, ["44"], [accepted(pred("44", 0))], [accepted(pred("4.4", 0))]
    )
    cli = Path(__file__).resolve().parents[1] / "score_paired_ocr.py"
    run = subprocess.run(
        [sys.executable, "-S", cli, "--manifest", manifest, "--decisions", method],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(run.stdout)["transitions"]["exact"]["wrong"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "swapped_paths",
        "wrong_count",
        "boolean_count",
        "digest",
        "missing_digest",
    ],
)
def test_manifest_binds_each_views_image_identity_without_pixels(tmp_path, damage):
    manifest, method = corpus(tmp_path, ["44"], [accepted(pred("44", 0))], [])
    reports = [json.loads(line) for line in manifest.read_text().splitlines()]
    for r in reports:
        path = tmp_path / r["coordinates"]
        coords = json.loads(path.read_bytes())
        info = coords["pages"][0]["image"]
        info["sha256"] = sha256(info["path"].encode()).hexdigest()
        path.write_text(json.dumps(coords))
        r["pages"] = 1
        r["images"] = [info["path"]]
        r["sha256"] = {
            info["path"]: info["sha256"],
            r["coordinates"]: sha256(path.read_bytes()).hexdigest(),
            r["ground_truth"]: sha256(
                (tmp_path / r["ground_truth"]).read_bytes()
            ).hexdigest(),
        }
    if damage == "swapped_paths":
        reports[0]["images"], reports[1]["images"] = (
            reports[1]["images"],
            reports[0]["images"],
        )
    elif damage == "wrong_count":
        reports[0]["pages"] = 5
    elif damage == "boolean_count":
        reports[0]["pages"] = True
    elif damage == "digest":
        reports[0]["sha256"][reports[0]["images"][0]] = "f" * 64
    elif damage == "missing_digest":
        del reports[0]["sha256"][reports[0]["images"][0]]
    stream(manifest, reports)
    if damage == "none":
        assert compare(manifest, method)["paired_rows"] == 1
    else:
        with pytest.raises(ValueError, match="manifest"):
            compare(manifest, method)


def test_wrong_sign_and_scale_are_both_reported(tmp_path):
    clean = [accepted(pred("-1.8", 0)), accepted(pred("4.4", 1))]
    copy = [accepted(pred("18", 0)), accepted(pred("44", 1))]
    manifest, method = corpus(tmp_path, ["-1.8", "4.4"], clean, copy)
    view = compare(manifest, method)["views"]["photocopy"]
    assert view["wrong_text_classes"] == {"sign_and_power_of_ten": 1, "power_of_ten": 1}
    assert view["power_of_ten_magnitude_rows"] == 2
    assert view["opposite_sign_rows"] == 1


def test_cli_summary_preserves_counts_and_excludes_row_evidence(tmp_path):
    manifest, method = corpus(tmp_path, ["44"], [accepted(pred("44", 0))], [])
    cli = Path(__file__).resolve().parents[1] / "score_paired_ocr.py"
    run = subprocess.run(
        [
            sys.executable,
            cli,
            "--manifest",
            manifest,
            "--decisions",
            method,
            "--summary",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    full = compare(manifest, method)
    assert json.loads(run.stdout) == {
        key: value for key, value in full.items() if key != "rows"
    }
