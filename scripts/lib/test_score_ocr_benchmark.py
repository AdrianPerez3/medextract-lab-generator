"""Exercise the release ZIP and an independent prediction stream through the public CLI."""

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest


CLI = Path(__file__).resolve().parents[1] / "score_ocr_benchmark.py"


def jsonl(records):
    return "".join(json.dumps(row) + "\n" for row in records)


def prediction(value, row=0):
    y = 2 * (10 * row + 5)
    return {
        "page": 1,
        "test_point": [10, y],
        "value_point": [30, y],
        "value_text": value,
    }


@pytest.fixture
def archive(tmp_path):
    manifest = []
    path = tmp_path / "benchmark.zip"
    with ZipFile(path, "w") as output:
        for rid, condition, values in [
            ("A", "clean", ["7,0"]),
            ("B", "photocopy", ["4.4", "9"]),
        ]:
            truth = json.dumps(
                {"results": [{"loinc": "X", "value": v} for v in values]}
            ).encode()
            rows = []
            for i in range(len(values)):
                y = 10 * i
                rows.append(
                    {
                        "result_index": i,
                        "loinc": "X",
                        "bbox": [0, y, 20, y + 10],
                        "cells": {
                            "test_name": [0, y, 10, y + 10],
                            "value": [10, y, 20, y + 10],
                        },
                    }
                )
            coords = {
                "schema_version": 1,
                "report_id": rid,
                "coordinate_space": "pdf_points_top_left",
                "ground_truth_sha256": sha256(truth).hexdigest(),
                "pages": [
                    {
                        "page": 1,
                        "width": 20,
                        "height": 20,
                        "rows": rows,
                        "image": {
                            "width": 40,
                            "height": 40,
                            "path": f"images/{rid}.jpg",
                        },
                    }
                ],
            }
            gt_path = f"ground_truth/{rid}.json"
            coord_path = f"coordinates/{rid}.json"
            manifest.append(
                {
                    "report_id": rid,
                    "condition": condition,
                    "draw": "one",
                    "coordinates": coord_path,
                    "ground_truth": gt_path,
                }
            )
            output.writestr(gt_path, truth)
            output.writestr(coord_path, json.dumps(coords))
        output.writestr("reports.jsonl", jsonl(manifest))
        # Reversed order and a duplicate submission on A are intentional.
        output.writestr(
            "predictions/test.jsonl",
            jsonl(
                [
                    {
                        "report_id": "B",
                        "predictions": [prediction("44"), prediction("9", 1)],
                        "review_count": 2,
                    },
                    {
                        "report_id": "A",
                        "predictions": [prediction("7,0")] * 2,
                        "review_count": 0,
                    },
                ]
            ),
        )
    return path


def cli(archive, *args):
    return subprocess.run(
        [sys.executable, str(CLI), str(archive), *args],
        text=True,
        capture_output=True,
        cwd=archive.parent,
    )


def test_replay_micro_scores_and_slice(archive):
    run = cli(archive)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["test"] == {
        "reports": 2,
        "exact": 2,
        "submitted": 4,
        "expected": 3,
        "review_entries": 2,
        "precision": 2 / 4,
        "recall": 2 / 3,
        "f1": 4 / 7,
    }
    run = cli(archive, "--condition", "photocopy", "--draw", "one")
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["test"] == {
        "reports": 1,
        "exact": 1,
        "submitted": 2,
        "expected": 2,
        "review_entries": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_external_predictions_can_abstain(archive, tmp_path):
    custom = tmp_path / "custom.jsonl"
    custom.write_text(jsonl([{"report_id": rid, "predictions": []} for rid in "AB"]))
    run = cli(archive, "--predictions", custom)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["custom"] == {
        "reports": 2,
        "exact": 0,
        "submitted": 0,
        "expected": 3,
        "review_entries": None,
        "precision": None,
        "recall": 0,
        "f1": 0,
    }


@pytest.mark.parametrize(
    ("ids", "error"),
    [
        ("A", "missing=['B']"),
        ("ABC", "extra=['C']"),
        ("ABA", "duplicate report A"),
    ],
)
def test_external_prediction_coverage(archive, tmp_path, ids, error):
    custom = tmp_path / "custom.jsonl"
    custom.write_text(jsonl([{"report_id": rid, "predictions": []} for rid in ids]))
    run = cli(archive, "--predictions", custom)
    assert run.returncode == 2
    assert error in run.stderr


def test_review_counts_must_cover_every_report(archive, tmp_path):
    custom = tmp_path / "custom.jsonl"
    custom.write_text(
        jsonl(
            [
                {"report_id": "A", "predictions": [], "review_count": 0},
                {"report_id": "B", "predictions": []},
            ]
        )
    )
    run = cli(archive, "--predictions", custom)
    assert run.returncode == 2
    assert "provide review_count" in run.stderr
