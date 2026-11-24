"""Paired image generation through the public generator and command line."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw
import pytest

import generate_complete_synthetic as generator
from scripts.make_paired_views import make_paired_views, write_photocopy
from scripts.score_row_associations import score


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_photocopy_preserves_canvas_and_foreground_positions(tmp_path):
    page = Image.new("RGB", (131, 167), "white")
    draw = ImageDraw.Draw(page)
    for box in [(5, 10, 20, 21), (71, 131, 84, 148)]:
        draw.rectangle(box, fill="black")
    copies = [tmp_path / f"{i}.jpg" for i in range(3)]
    for path, seed in zip(copies, [123, 123, 456], strict=True):
        write_photocopy(page, path, seed=seed)
    assert copies[0].read_bytes() == copies[1].read_bytes() != copies[2].read_bytes()
    with Image.open(copies[0]) as image:
        assert image.mode == "L" and image.size == page.size
        assert image.point(lambda p: 255 if p < 115 else 0).getbbox() == pytest.approx(
            (5, 10, 85, 149), abs=1
        )


@pytest.fixture()
def source(tmp_path):
    source = tmp_path / "source"
    assert (
        generator.main(
            [
                "--count",
                "2",
                # Long catalog titles produce a multi-page source in this fixture.
                "--name-style",
                "catalog",
                "--format",
                "jpg",
                "--coordinates",
                "--workers",
                "1",
                "--tiers",
                "1,0,0,0,0",
                "--dpi",
                "72",
                "--output",
                str(source),
            ]
        )
        == 0
    )
    return source


def test_real_generator_pairs_are_scoreable_and_identically_split(source, tmp_path):
    manifest = source / "reports.jsonl"
    originals = records(manifest)
    originals[0]["split"] = "validation"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in originals))
    paired = tmp_path / "paired"
    cli = Path(__file__).resolve().parents[1] / "make_paired_views.py"
    run = subprocess.run(
        [sys.executable, str(cli), str(source), str(paired), "--seed", "25"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    outputs = records(paired / "reports.jsonl")
    assert len(outputs) == 4
    for i, original in enumerate(originals):
        clean, copy = outputs[i * 2 : i * 2 + 2]
        assert (clean["condition"], copy["condition"]) == ("clean", "photocopy")
        assert (
            clean["group_id"]
            == copy["group_id"]
            == clean["pair_id"]
            == copy["pair_id"]
            == original["report_id"]
        )
        assert clean["split"] == copy["split"] == original.get("split", "test")
        assert clean["report_id"] != copy["report_id"]
        assert clean["ground_truth"] == copy["ground_truth"]
        labels = (paired / clean["ground_truth"]).read_bytes()
        assert labels == (source / original["ground_truth"]).read_bytes()
        coords = [
            json.loads((paired / view["coordinates"]).read_bytes())
            for view in (clean, copy)
        ]
        for view, sidecar in zip([clean, copy], coords, strict=True):
            assert sidecar["report_id"] == view["report_id"]
            assert sidecar["ground_truth_sha256"] == sha256(labels).hexdigest()
            assert all(
                sha256((paired / name).read_bytes()).hexdigest() == value
                for name, value in view["sha256"].items()
            )
            assert (
                score(
                    sidecar,
                    labels,
                    {"report_id": view["report_id"], "predictions": []},
                    "image-pixels",
                )["expected_rows"]
                > 0
            )
        for j, (a, b) in enumerate(
            zip(coords[0]["pages"], coords[1]["pages"], strict=True)
        ):
            assert a["rows"] == b["rows"]
            assert a["image"]["path"] != b["image"]["path"]
            assert (paired / clean["images"][j]).read_bytes() == (
                source / original["images"][j]
            ).read_bytes()
            with Image.open(paired / copy["images"][j]) as img:
                assert img.size == (a["image"]["width"], a["image"]["height"])


@pytest.mark.parametrize(
    "fault", ["labels", "invalid_tier", "outside_root", "duplicate"]
)
def test_broken_input_never_publishes_partial_pairs(source, tmp_path, fault):
    manifest = source / "reports.jsonl"
    data = records(manifest)
    last = data[-1]
    if fault == "labels":
        (source / last["ground_truth"]).write_text("{}")
    elif fault == "invalid_tier":
        last["tier"] = 3
    elif fault == "outside_root":
        last["ground_truth"] = "../outside.json"
    else:
        data.append(last)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in data))
    dest = tmp_path / "paired"
    with pytest.raises(ValueError):
        make_paired_views(source, dest, seed=12)
    assert not dest.exists()


def test_nested_output_isolated_and_reproducible_across_manifest_order(
    source, tmp_path
):
    manifest = source / "reports.jsonl"
    inputs = records(manifest)
    assert any(r["pages"] > 1 for r in inputs)
    for r in inputs:
        r.update(
            group_id="shared",
            family="custom-layout",
            split="validation",
            source_dpi=72,
            operator_notes="private benchmark note",
        )
    manifest.write_text("".join(json.dumps(r) + "\n" for r in inputs))
    nested = source / "benchmarks" / "paired"
    assert make_paired_views(source, nested, seed=-41)["pairs"] == 2
    expected = {r["report_id"]: r for r in records(nested / "reports.jsonl")}
    assert all(
        r["group_id"] == "shared"
        and r["split"] == "validation"
        and r["family"] == "custom-layout"
        and r["source_dpi"] == 72
        for r in expected.values()
    )
    assert all("operator_notes" not in r for r in expected.values())
    manifest.write_text("".join(json.dumps(r) + "\n" for r in reversed(inputs)))
    other = tmp_path / "reordered"
    make_paired_views(source, other, seed=-41)
    assert expected == {r["report_id"]: r for r in records(other / "reports.jsonl")}


def test_refuses_output_in_source_artifact_inventory_or_an_existing_result(
    source, tmp_path
):
    for child in ("images/subset", "coordinates/subset", "ground_truth/subset"):
        with pytest.raises(ValueError, match="outside the source artifact directories"):
            make_paired_views(source, source / child, seed=1)
        assert not (source / child).exists()
    with pytest.raises(ValueError):
        make_paired_views(source, source.parent, seed=1)
    dest = tmp_path / "existing"
    dest.mkdir()
    saved = dest / "saved.txt"
    saved.write_text("prior results")
    with pytest.raises(ValueError):
        make_paired_views(source, dest, seed=1)
    assert saved.read_text() == "prior results"


def test_group_cannot_cross_split_before_pairing(source, tmp_path):
    manifest = source / "reports.jsonl"
    inputs = records(manifest)
    for r, split in zip(inputs, ["train", "test"], strict=True):
        r.update(group_id="shared", split=split)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in inputs))
    dest = tmp_path / "paired"
    with pytest.raises(ValueError, match="multiple splits"):
        make_paired_views(source, dest, seed=1)
    assert not dest.exists()
