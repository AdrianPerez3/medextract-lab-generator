"""Reproduce the public result tables from the frozen geometry and OCR decisions."""

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.score_paired_ocr import compare


DOCS = Path(__file__).resolve().parents[2] / "docs" / "benchmarks"


def test_frozen_paired_results(tmp_path):
    published = json.loads((DOCS / "paired-photocopy-results.json").read_bytes())
    with ZipFile(DOCS / "paired-photocopy-replay.zip") as replay:
        inventory = replay.read("SHA256SUMS").decode().splitlines()
        members = set()
        for line in inventory:
            digest, name = line.split("  ", 1)
            assert name not in members and not Path(name).is_absolute() and ".." not in Path(name).parts
            assert sha256(replay.read(name)).hexdigest() == digest
            members.add(name)
        assert set(replay.namelist()) == members | {"SHA256SUMS"}
        replay.extractall(tmp_path)
    manifest = tmp_path / "inputs" / "reports.jsonl"
    assert sha256(manifest.read_bytes()).hexdigest() == published["manifest_sha256"]
    expected_methods = {"paddle-only", "tesseract-only", "combined", "guard", "vl", "both"}
    assert set(published["methods"]) == expected_methods
    for method, totals in published["methods"].items():
        baseline = tmp_path / "decisions" / "combined.jsonl" if method in {"guard", "vl", "both"} else None
        score = compare(manifest, tmp_path / "decisions" / (method + ".jsonl"), baseline)
        assert (score["report_pairs"], score["paired_rows"]) == (40, 280)
        assert totals == {key: score[key] for key in ("views", "transitions")}
