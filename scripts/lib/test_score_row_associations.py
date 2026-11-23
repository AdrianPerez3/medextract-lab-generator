"""Independent cell positions, point pairs and expected association counts."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.score_row_associations import score


CLI = Path(__file__).resolve().parents[1] / "score_row_associations.py"


def fixture():
    # Two equal texts on page one deliberately cannot reveal a swap by text alone.
    codes = ["100-1", "101-9", "100-1"]
    visible = {
        "results": [
            {"loinc": c, "value": v} for c, v in zip(codes, ["7,0", "7,0", "12"])
        ]
    }
    data = json.dumps(visible).encode()
    pages = []
    for number, indexes in enumerate(([0, 1], [2]), 1):
        rows = []
        for j, idx in enumerate(indexes):
            top = 20 + 10 * j
            rows.append(
                {
                    "result_index": idx,
                    "loinc": codes[idx],
                    "bbox": [10, top, 85, top + 10],
                    "cells": {
                        "test_name": [10, top, 55, top + 10],
                        "value": [55, top, 75, top + 10],
                        "unit": [75, top, 85, top + 10],
                    },
                }
            )
        pages.append(
            {
                "page": number,
                "width": 100.5,
                "height": 120.25,
                "rows": rows,
                "image": {"path": f"images/{number}.jpg", "width": 220, "height": 262},
            }
        )
    return {
        "schema_version": 1,
        "coordinate_space": "pdf_points_top_left",
        "report_id": "RPT_000002",
        "ground_truth_sha256": sha256(data).hexdigest(),
        "pages": pages,
    }, data


def pred(ty, vy, text="7,0", page=1):
    return {
        "page": page,
        "test_point": [20, ty],
        "value_point": [60, vy],
        "value_text": text,
    }


def run(items, **kwargs):
    c, v = fixture()
    return score(c, v, {"predictions": items}, **kwargs)


def test_perfect_across_pages_despite_repeated_code():
    actual = run([pred(24, 24), pred(34, 34), pred(24, 24, "12", 2)])
    assert actual["expected_rows"] == actual["submitted_predictions"] == 3
    assert (
        actual["association"]
        == actual["association_and_exact_value"]
        == {"matched_rows": 3, "missing_rows": 0, "precision": 1.0, "recall": 1.0}
    )
    assert actual["duplicate_test_anchor_predictions"] == 0


def test_swapped_equal_values_are_still_wrong_and_keep_both_directions():
    actual = run([pred(24, 34), pred(34, 24), pred(26, 26, "12", 2)])
    assert (
        actual["association"]["matched_rows"]
        == actual["association_and_exact_value"]["matched_rows"]
        == 1
    )
    assert (
        actual["association"]["recall"] == actual["association"]["precision"] == 1 / 3
    )
    assert actual["prediction_geometry"]["different_row"] == 2
    assert actual["wrong_row_offsets"] == {"-1": 1, "1": 1}
    assert actual["exact_text_at_value_cell"] == actual["exact_text_for_test_row"] == 3


def test_exact_means_printed_string_and_no_credit_for_neighbor_transcription():
    c, v = fixture()
    changed = json.loads(v)
    changed["results"][1]["value"] = "8,0"
    v = json.dumps(changed).encode()
    c["ground_truth_sha256"] = sha256(v).hexdigest()
    actual = score(c, v, {"predictions": [pred(24, 34, "8,0"), pred(34, 34, "8.0")]})
    assert actual["association"]["matched_rows"] == 1
    assert actual["association_and_exact_value"]["matched_rows"] == 0
    assert actual["exact_text_at_value_cell"] == 1
    assert actual["exact_text_for_test_row"] == 0


def test_duplicate_one_to_one_order_independent_and_keeps_all_in_denominator():
    entries = [pred(24, 34), pred(24, 24, "7.0"), pred(24, 24), pred(24, 24)]
    forward = run(entries)
    assert forward == run(entries[::-1])
    assert (
        forward["association"]
        == forward["association_and_exact_value"]
        == {"matched_rows": 1, "missing_rows": 2, "precision": 0.25, "recall": 1 / 3}
    )
    assert forward["duplicate_test_anchor_predictions"] == 3
    assert forward["rows_without_test_anchor"] == 2


def test_diagnoses_missing_cells_outside_page_and_unknown_page():
    bad_cell = {
        **pred(24, 24),
        "test_point": [60, 24],
    }  # name anchor landed in the value cell
    off_page = {**pred(24, 24), "value_point": [100.5, -1]}
    header = pred(10, 10)
    actual = run([bad_cell, off_page, header, pred(24, 24, page=9)])
    assert actual["prediction_geometry"] == {
        "same_row": 0,
        "different_row": 0,
        "outside_target_cell": 2,
        "outside_page": 1,
        "unknown_page": 1,
    }
    assert actual["rows_without_test_anchor"] == 2
    assert actual["association"]["missing_rows"] == 3


def test_empty_has_zero_recall_and_undefined_precision():
    actual = run([])
    assert (
        actual["association"]
        == actual["association_and_exact_value"]
        == {"matched_rows": 0, "missing_rows": 3, "precision": None, "recall": 0.0}
    )


def test_half_open_cells_for_adjacent_rows_and_columns():
    # y=30 is in the second row; x=55 is in the value, not the name cell.
    actual = run(
        [
            {
                "page": 1,
                "test_point": [10, 30],
                "value_point": [55, 30],
                "value_text": "7,0",
            },
            {
                "page": 1,
                "test_point": [55, 30],
                "value_point": [75, 30],
                "value_text": "7,0",
            },
        ]
    )
    assert actual["association_and_exact_value"]["matched_rows"] == 1
    assert actual["prediction_geometry"]["outside_target_cell"] == 1


def test_shared_edge_roundoff_in_rendered_coordinates_belongs_to_lower_row():
    c, v = fixture()
    row = c["pages"][0]["rows"][0]
    row["bbox"][3] += 1e-13
    for cell in row["cells"].values():
        cell[3] += 1e-13
    r = score(c, v, {"predictions": [pred(25, 30)]})
    assert r["wrong_row_offsets"] == {"1": 1}


def test_image_coordinates_use_actual_axis_ratios_and_require_image_on_all_pages():
    items = [pred(24, 24), pred(34, 34), pred(24, 24, "12", 2)]
    # Pick points close to both cell boundaries so rounding to nominal DPI fails.
    items[0]["test_point"] = [54.997, 29.999]
    px = deepcopy(items)
    for p in px:
        for key in ("test_point", "value_point"):
            x, y = p[key]
            p[key] = [x * 220 / 100.5, y * 262 / 120.25]
    assert (
        run(px, space="image-pixels")["association_and_exact_value"]
        == run(items)["association_and_exact_value"]
    )
    c, v = fixture()
    del c["pages"][1]["image"]
    with pytest.raises(ValueError, match="image"):
        score(c, v, {"predictions": []}, space="image-pixels")


@pytest.mark.parametrize(
    "key,value",
    [
        ("test_point", [float("nan"), 22]),
        ("value_point", [float("inf"), 22]),
        ("value_point", [False, 2]),
        ("test_point", [42]),
        ("page", 0),
        ("page", True),
        ("value_text", 7),
    ],
)
def test_invalid_prediction(key, value):
    with pytest.raises(ValueError):
        run([{**pred(24, 24), key: value}])


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "row_index",
        "row_code",
        "row_overlap",
        "cell_overlap",
        "missing_cell",
        "cell_bounds",
        "page_order",
        "page_size",
        "space",
        "report_id",
    ],
)
def test_rejects_ambiguous_or_unbound_input(damage):
    c, v = fixture()
    page = c["pages"][0]
    row = page["rows"][1]
    cases = {
        "digest": (c, "ground_truth_sha256", "f" * 64),
        "row_index": (row, "result_index", 0),
        "row_code": (row, "loinc", "101-8"),
        "row_overlap": (row, "bbox", [10, 29, 85, 40]),
        "cell_overlap": (row["cells"], "value", [54, 30, 75, 40]),
        "missing_cell": (row, "cells", {"value": [55, 30, 75, 40]}),
        "cell_bounds": (row["cells"], "value", [55, 30, 175, 40]),
        "page_order": (page, "page", 2),
        "page_size": (page, "height", float("nan")),
        "space": (c, "coordinate_space", "pdf_points_bottom_left"),
        "report_id": (c, "report_id", ""),
    }
    target, key, value = cases[damage]
    target[key] = value
    with pytest.raises(ValueError):
        score(c, v, {"predictions": []})


def test_cli_round_trip_and_digest_mismatch(tmp_path):
    c, v = fixture()
    sidecar = tmp_path / "coords.json"
    labels = tmp_path / "visible.json"
    guesses = tmp_path / "pairs.json"
    sidecar.write_text(json.dumps(c))
    labels.write_bytes(v)
    guesses.write_text(
        json.dumps({"report_id": c["report_id"], "predictions": [pred(25, 25)]})
    )
    cmd = [
        sys.executable,
        str(CLI),
        "--coordinates",
        str(sidecar),
        "--ground-truth",
        str(labels),
        "--predictions",
        str(guesses),
    ]
    ok = subprocess.run(cmd, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["association_and_exact_value"]["matched_rows"] == 1
    labels.write_bytes(v + b"\n")
    bad = subprocess.run(cmd, capture_output=True, text=True)
    assert bad.returncode != 0 and "SHA-256" in bad.stderr and bad.stdout == ""
