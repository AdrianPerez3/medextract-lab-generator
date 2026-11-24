"""Verify the punctuation challenge's observation semantics and the actual printed rows."""

from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import random
import subprocess
import sys

import pypdfium2 as pdfium
import pytest

from scripts import generate_paired_ocr_challenge as challenge


def test_true_integers_decimal_marks_signs_and_oxygen_saturation():
    values = []
    for draw in range(10):
        rows = challenge.make_rows(random.Random(42), draw)
        assert len(rows) == len({r["loinc"] for r in rows}) == 7
        by_code = {r["loinc"]: r for r in rows}
        assert "11556-8" not in by_code
        oxygen = by_code["2711-0"]
        assert oxygen["test_name"] == "Saturación venosa de oxígeno"
        assert oxygen["unit"] == "%"
        assert by_code["4544-3"]["value"] == "44"
        assert by_code["2160-0"]["value"][-1] == "0"
        assert by_code["1925-7"]["value"][0] == ("-" if draw % 2 == 0 else "+")
        values.append(by_code["2823-3"]["value"])
    assert Counter(values) == {"4.4": 5, "4,4": 5}


@pytest.mark.parametrize("layout", range(len(challenge.LAYOUTS)))
def test_pdf_contains_ground_truth_in_each_announced_cell(layout):
    rng = random.Random(layout)
    rows = challenge.make_rows(rng, layout)
    document, geometry = challenge.draw_page("SYNPAIR_test", layout, rng, rows)
    with (
        closing(pdfium.PdfDocument(document)) as doc,
        closing(doc[0]) as page,
        closing(page.get_textpage()) as text,
    ):
        _, height = page.get_size()
        for index, row in enumerate(geometry):
            assert (row["result_index"], row["loinc"]) == (index, rows[index]["loinc"])
            for field, (left, top, right, bottom) in row["cells"].items():
                printed = text.get_text_bounded(
                    left, height - bottom, right, height - top
                )
                assert printed.strip() == rows[index][field], (index, field, printed)


def test_public_command_generates_complete_paired_challenge(tmp_path):
    output = tmp_path / "paired"
    script = Path(challenge.__file__)
    result = subprocess.run(
        [sys.executable, str(script), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "recipe": "photocopy-v1",
        "seed": challenge.DEFAULT_SEED,
        "pairs": 40,
        "views": 80,
        "images": 80,
    }
    rows = [
        json.loads(line) for line in (output / "reports.jsonl").read_text().splitlines()
    ]
    assert len({r["report_id"] for r in rows}) == len(rows) == 80
    assert Counter(r["condition"] for r in rows) == {"clean": 40, "photocopy": 40}
    assert len({r["pair_id"] for r in rows}) == 40
    assert all(r["split"] == "test" for r in rows)
    vocabulary = json.loads((output / "paired-ocr-terminology.json").read_bytes())
    names = {t["LOINC_NUM"]: t["local_test_name_es"] for t in vocabulary["terms"]}
    for label in (output / "ground_truth").glob("*.json"):
        for row in json.loads(label.read_bytes())["results"]:
            assert row["test_name"] == names[row["loinc"]]
    for path in ("LOINC_short_license.txt", "licenses/LoincLicense_5.8.txt"):
        assert (output / path).read_bytes() == (
            challenge.PROJECT_ROOT / path
        ).read_bytes()


def test_all_challenge_codes_and_units_match_the_versioned_terminology():
    vocabulary = (
        Path(challenge.__file__).resolve().parents[1]
        / "docs/benchmarks/paired-ocr-terminology.json"
    )
    release = json.loads(vocabulary.read_bytes())
    assert release["loinc_version"] == "2.82"
    terms = {r["LOINC_NUM"]: r for r in release["terms"]}
    tests = (
        challenge.POTASSIUM,
        challenge.HEMATOCRIT,
        challenge.BASE_EXCESS,
        challenge.CREATININE,
        challenge.VENOUS_SATURATION,
        *challenge.EXTRA,
    )
    assert terms.keys() == {test.code for test in tests}
    for test in tests:
        term = terms[test.code]
        assert term["STATUS"] == "ACTIVE"
        assert term["EXAMPLE_UCUM_UNITS"] == test.unit
        assert term["LONG_COMMON_NAME"]
        assert term["local_test_name_es"] == test.name


def test_new_challenge_output_is_published_by_one_directory_rename(
    tmp_path, monkeypatch
):
    """Windows refuses renaming a populated directory onto an existing empty stage."""
    original_rename = Path.rename
    destinations = []

    def rename_to_unused_directory(path, target):
        if target.exists():
            raise FileExistsError("Windows directory rename target exists")
        destinations.append(target)
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", rename_to_unused_directory)
    output = tmp_path / "paired"
    assert challenge.generate(output)["pairs"] == 40
    assert destinations == [output]
    assert (output / "paired-ocr-terminology.json").is_file()
