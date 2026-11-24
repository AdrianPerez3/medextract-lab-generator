"""Exercise the public saved-TSV example and its deliberately swapped pair."""

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "image-to-row"


def run_script(name, *args, workdir):
    # Saved TSV replay and scoring should work without any site packages.
    completed = subprocess.run(
        [sys.executable, "-S", ROOT / "scripts" / name, *args],
        cwd=workdir,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_replay_and_intentionally_swapped_rows(tmp_path):
    def score(predictions):
        return run_script(
            "score_row_associations.py",
            "--coordinates",
            EXAMPLE / "coordinates.json",
            "--ground-truth",
            ROOT / "examples" / "visible.json",
            "--predictions",
            predictions,
            "--space",
            "image-pixels",
            workdir=tmp_path,
        )

    replay = run_script(
        "extract_ocr_rows.py",
        "--tsv",
        EXAMPLE / "report.tsv",
        "--report-id",
        "RPT_000000",
        workdir=tmp_path,
    )
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    baseline = score(path)
    assert replay["review"] == []
    assert baseline["expected_rows"] == baseline["submitted_predictions"] == 17
    assert baseline["association"]["matched_rows"] == 17
    assert baseline["association_and_exact_value"]["matched_rows"] == 12

    # The control swaps both the value text and its observed location.
    # Verify that it is derived from the two real OCR predictions.
    left, right = replay["predictions"][:2]
    assert (left["value_text"], right["value_text"]) == ("6,79", "4,87")
    fixture = EXAMPLE / "swapped.json"
    swapped = json.loads(fixture.read_bytes())["predictions"]
    for item, name, value in zip(swapped, (left, right), (right, left), strict=True):
        assert item == {
            "page": name["page"],
            "test_point": name["test_point"],
            "value_point": value["value_point"],
            "value_text": value["value_text"],
        }
    control = score(fixture)
    assert control["submitted_predictions"] == control["exact_text_at_value_cell"] == 2
    assert control["prediction_geometry"]["different_row"] == 2
    assert control["association_and_exact_value"]["matched_rows"] == 0
