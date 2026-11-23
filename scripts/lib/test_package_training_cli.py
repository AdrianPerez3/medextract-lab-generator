"""Public split command: legacy streams and aligned, duplicate-aware schema-v2 partitions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import package_training_data as package


def cli(dataset, *args):
    return subprocess.run(
        [sys.executable, str(Path(package.__file__)), "--dataset", str(dataset), *args],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True,
    )


@pytest.mark.parametrize("split", ["nan,0,100", "inf,0,0", "101,-1,0", "30,30,30"])
def test_bad_percentages_leave_no_output(tmp_path, split):
    (tmp_path / "sft_train.jsonl").write_text('{"id":1}\n')
    assert cli(tmp_path, "--split", split).returncode != 0
    assert not (tmp_path / "splits").exists()


@pytest.mark.parametrize("bad", ["", "broken\n", '{"value":NaN}\n', '[]\n'])
def test_invalid_or_empty_stream_rolls_back_other_formats(tmp_path, bad):
    (tmp_path / "sft_train.jsonl").write_text('{"id":1}\n')
    (tmp_path / "dpo_train.jsonl").write_text(bad)
    assert cli(tmp_path).returncode != 0
    assert not (tmp_path / "splits").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dpo_train.jsonl", "sft_train.jsonl"]


def test_empty_dataset_and_existing_outputs(tmp_path):
    assert cli(tmp_path).returncode != 0
    assert not list(tmp_path.iterdir())
    source = tmp_path / "sft_train.jsonl"
    source.write_text("".join(json.dumps({"id": n}) + "\n" for n in range(10)))
    original = source.read_bytes()
    assert cli(tmp_path, "--output", str(tmp_path)).returncode != 0
    assert source.read_bytes() == original
    packaged = cli(tmp_path)
    assert packaged.returncode == 0, packaged.stderr
    out = tmp_path / "splits"
    receipt = json.loads((out / "packaging_report.json").read_text())
    assert str(tmp_path) not in (out / "packaging_report.json").read_text()
    assert receipt["dataset"] == ".." and receipt["output_dir"] == "."
    assert receipt["formats"]["sft"]["source"] == "sft_train.jsonl"
    assert all((out / item["file"]).is_file() for item in receipt["formats"]["sft"]["split_details"].values())
    assert receipt["formats"]["sft"]["splits"] == {"train": 8, "val": 1, "test": 1}
    assert receipt["grouping"] == "independent legacy stream rows"
    assert cli(tmp_path).returncode != 0
    assert source.read_bytes() == original


def test_v2_duplicate_groups_shared_across_shuffled_streams_and_paths_resolve(tmp_path, monkeypatch):
    from PIL import Image

    data = tmp_path / "data"
    for name in ("images", "ground_truth", "normalized_ground_truth"):
        (data / name).mkdir(parents=True)
    modes = ["sft", "dpo", "grpo"]
    streams = {mode: [] for mode in modes}
    manifest = []
    for num in range(10):
        rid = f"RPT_{num:06}"
        image = f"images/{rid}.jpg"
        # 0 / 1 have identical pixels, 1 / 2 have identical targets.
        Image.new("RGB", (8, 8), (0 if num < 2 else num * 20, 0, 0)).save(data / image)
        target = {"results": [{"value": 1 if num in (1, 2) else num}]}
        row = {"report_id": rid, "pages": 1, "images": [image], "sha256": {}}
        for attr in ("ground_truth", "normalized_ground_truth"):
            row[attr] = f"{attr}/{rid}.json"
            (data / row[attr]).write_text(json.dumps(target))
        for ref in [image, row["ground_truth"], row["normalized_ground_truth"]]:
            row["sha256"][ref] = hashlib.sha256((data / ref).read_bytes()).hexdigest()
        manifest.append(row)
        for mode in modes:
            item = {"images": [image], "messages": [{"role": "user", "content": "<image>"}]}
            if mode == "grpo":
                item["solution"] = json.dumps(target)
            else:
                item["messages"].append({"role": "assistant", "content": json.dumps(target)})
            if mode == "dpo":
                item["rejected_response"] = '{"results":[]}'
            streams[mode].append(item)

    (data / "reports.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (data / "metadata.json").write_text(json.dumps({"schema_version": 2, "count": 10, **dict.fromkeys(modes, True)}))
    for mode in modes:
        sequence = reversed(streams[mode]) if mode == "grpo" else streams[mode]
        (data / f"{mode}_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sequence))
    output = tmp_path / "outside"
    result = cli(data, "--split", "50,30,20", "--output", str(output))
    assert result.returncode == 0, result.stderr
    receipt = (output / "packaging_report.json").read_text()
    assert str(tmp_path) not in receipt and json.loads(receipt)["dataset"] is None
    split_ids = json.loads((output / "report_splits.json").read_text())
    groups = [set(v) for v in split_ids.values()]
    assert any({f"RPT_{n:06d}" for n in (0, 1, 2)} <= group for group in groups)
    assert len(set.union(*groups)) == sum(map(len, groups)) == 10
    for mode in modes:
        for split, ids in split_ids.items():
            rows = [json.loads(s) for s in (output / f"{mode}_{split}.jsonl").read_text().splitlines()]
            assert [Path(r["images"][0]).stem for r in rows] == ids
            assert all((output / r["images"][0]).resolve().parent == data / "images" for r in rows)

    # A relative path cannot span Windows drive letters.
    def different_volume(*args):
        raise ValueError("paths are on different drives")

    volume = tmp_path / "another-volume"
    with monkeypatch.context() as patch:
        patch.setattr(package.os.path, "relpath", different_volume)
        receipt = package.package_dataset(data, volume, 50, 30, 42)
    assert receipt["image_path_base"] == "original dataset"
    for row in (volume / "sft_train.jsonl").read_text().splitlines():
        assert (data / json.loads(row)["images"][0]).is_file()

    (data / manifest[0]["images"][0]).write_bytes(b"changed page")
    assert cli(data, "--output", str(tmp_path / "tampered")).returncode != 0
    assert not (tmp_path / "tampered").exists()


def test_missing_v2_metadata_cannot_downgrade_partitioning_to_legacy(tmp_path):
    (tmp_path / "reports.jsonl").write_text('{"report_id":"RPT_000000"}\n')
    (tmp_path / "sft_train.jsonl").write_text('{"id":1}\n')
    result = cli(tmp_path)
    assert result.returncode != 0 and "schema-v2" in result.stderr
    assert not (tmp_path / "splits").exists()
