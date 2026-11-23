"""Integration tests for the same command and on-disk files a generator user gets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import generate_complete_synthetic as generator
from scripts.lib.ground_truth_writer import GroundTruthWriter


SCRIPT = Path(generator.__file__)


def cli(output, *args, hash_seed="1"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--count", "2", "--workers", "1", "--seed", "42",
         "--tiers", "1,0,0,0,0", "--output", str(output), *args],
        cwd=output.parent, capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_multimodal_output_is_deterministic_across_workers_and_hash_seeds(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert cli(first).returncode == 0
    parallel = cli(second, "--workers", "2", hash_seed="2")
    assert parallel.returncode == 0, parallel.stderr
    metadata = json.loads((second / "metadata.json").read_text())
    assert (metadata["schema_version"], metadata["workers"]) == (2, 2)
    for filename, expected in metadata["source_sha256"].items():
        assert hashlib.sha256((SCRIPT.parent / filename).read_bytes()).hexdigest() == expected
    for path in first.rglob("*"):
        if path.is_file() and path.name not in ("coverage_report.json", "metadata.json"):
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes(), path.name
    reports = records(first / "reports.jsonl")
    assert len({r["seed"] for r in reports}) == 2
    coverage = json.loads((first / "coverage_report.json").read_text())
    histogram = coverage["loinc_counts"]
    assert len(histogram) == coverage["total_loincs_in_dict"]
    assert sum(histogram.values()) == sum(r["result_count"] for r in reports)
    assert coverage["min_loinc_count"] == min(histogram.values()) == 0
    assert metadata["eligible_loincs_sha256"] == hashlib.sha256(
        "\n".join(sorted(histogram)).encode()
    ).hexdigest()
    for mode in ("sft", "dpo", "grpo"):
        rows = records(first / f"{mode}_train.jsonl")
        assert len(rows) == len(reports)
        for row, report in zip(rows, reports):
            assert (first / report["pdf"]).is_file()
            assert (first / report["normalized_ground_truth"]).is_file()
            target = json.loads((first / report["ground_truth"]).read_text())
            assert len(row["images"]) == report["pages"]
            assert row["images"] == report["images"]
            assert row["messages"][1]["content"].count("<image>") == report["pages"]
            assert all((first / image).stat().st_size > 0 for image in report["images"])
            if mode == "grpo":
                assert json.loads(row["solution"]) == target
            else:
                assert json.loads(row["messages"][-1]["content"]) == target
            if mode == "dpo":
                assert json.loads(row["rejected_response"]) != target


@pytest.mark.parametrize("fmt", ["pdf", "jpg"])
def test_single_format_output_and_disabled_exports(tmp_path, fmt):
    out = tmp_path / fmt
    out.mkdir()  # Supported for existing clients such as batch schedulers
    args = ("--no-sft", "--no-grpo") if fmt == "jpg" else ()
    result = cli(out, "--format", fmt, *args)
    assert result.returncode == 0, result.stderr
    items = records(out / "reports.jsonl")
    assert all((out / item["ground_truth"]).is_file() for item in items)
    assert (out / "pdfs").is_dir() == (fmt == "pdf")
    assert (out / "images").is_dir() == (fmt == "jpg")
    for item in items:
        if fmt == "jpg":
            assert item["pdf"] is None
            assert len(item["images"]) == item["pages"]
            assert all((out / image).is_file() for image in item["images"])
        else:
            assert (out / item["pdf"]).is_file()
            assert not item["images"]
    assert (out / "dpo_train.jsonl").is_file() == (fmt == "jpg")
    assert not (out / "sft_train.jsonl").exists()
    assert not (out / "grpo_train.jsonl").exists()
    baseline = (out / "reports.jsonl").read_bytes()
    assert cli(out, "--format", fmt).returncode != 0
    assert (out / "reports.jsonl").read_bytes() == baseline


@pytest.mark.parametrize("args", [
    ["--dpi", "71"], ["--dpi", "601"], ["--dpi", "144.5"],
    ["--count", "0"], ["--workers", "-1"], ["--min-loinc-coverage", "-1"],
    ["--tiers", "NaN,0,0,0,1"], ["--tiers", "2,-1,0,0,0"], ["--format", "pdf", "--dpo"],
])
def test_invalid_arguments_fail_before_creating_output(tmp_path, args):
    with pytest.raises(SystemExit) as error:
        generator.main(["--output", str(tmp_path / "output"), *args])
    assert error.value.code != 0
    assert not list(tmp_path.iterdir())


def test_unmet_minimum_or_write_error_does_not_publish(tmp_path, monkeypatch):
    out = tmp_path / "output"
    failed = cli(out, "--format", "pdf", "--min-loinc-coverage", "999")
    assert failed.returncode != 0 and "eligible LOINCs" in failed.stderr
    assert not list(tmp_path.iterdir())

    def reject_write(*args, **kwargs):
        raise OSError("simulated image write failure")

    monkeypatch.setattr(generator, "_pdf_to_degraded_image", reject_write)
    assert generator.main(["--output", str(out), "--count", "1", "--workers", "1"]) != 0
    assert not list(tmp_path.iterdir())


def test_writer_propagates_delayed_stream_errors(tmp_path):
    class FullDisk:
        def close(self):
            raise OSError("simulated disk full during close")

    writer = GroundTruthWriter(tmp_path)
    writer._sft_file = FullDisk()
    with pytest.raises(OSError, match="disk full"):
        writer.close()


def test_concurrent_publish_preserves_one_complete_output(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    rendezvous = Barrier(2)
    output = tmp_path / "shared"

    def stage(args, directory):
        (directory / "dataset").write_text(str(args.seed))
        rendezvous.wait(timeout=5)

    monkeypatch.setattr(generator, "_generate", stage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(lambda n: generator.main(
            ["--output", str(output), "--seed", str(n)]
        ), (31, 42)))
    assert sorted(codes) == [0, 1]
    assert (output / "dataset").read_text() in {"31", "42"}
    assert list(tmp_path.iterdir()) == [output]


def test_minimum_coverage_succeeds_for_every_code_in_selected_inputs(tmp_path):
    # Limit the dictionary to one term from the real pinned catalog. Every
    # result and the coverage denominator must then agree with this input.
    data = tmp_path / "data"
    data.mkdir()
    code = "718-7"
    (data / "reference_ranges.json").write_text(json.dumps({
        code: {"min": 1, "max": 30, "unit": "g/dL"}
    }))
    (data / "loinc_metadata.json").write_text("{}")
    (data / "panel_definitions.json").write_text('{"panels":[]}')
    out = tmp_path / "limited"
    result = cli(out, "--format", "pdf", "--data-dir", str(data), "--min-loinc-coverage", "2")
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "coverage_report.json").read_text())
    assert report["min_loinc_count"] == 2
    assert report["loinc_counts"] == {code: 2}
    assert report["loinc_coverage_pct"] == 100


def test_only_explicit_reference_opt_in_changes_ranges_and_flags(tmp_path):
    from scripts.lib.dataset_integrity import check_run
    from PIL import Image

    targets = []
    for name, dpi, more in (("omitted", 144, []), ("simulated", 150, ["--simulated-reference-ranges"])):
        out = tmp_path / name
        result = cli(out, "--format", "both", "--dpi", str(dpi), "--count", "4", *more)
        assert result.returncode == 0, result.stderr
        assert check_run(out, generator._DATA_DIR, 0, verify_source=True)["reports"] == 4
        metadata = records(out / "reports.jsonl")
        settings = json.loads((out / "metadata.json").read_text())
        assert settings["dpi"] == dpi and settings["reference_policy"] == name
        assert settings["numeric_vocabulary"] == "guarded"
        assert settings["profile_kind"] == "synthetic_generation_scenario"
        rows = []
        for item in metadata:
            assert item["reference_policy"] == name
            with Image.open(out / item["images"][0]) as page:
                assert page.size == ((1191, 1684) if dpi == 144 else (1241, 1754))
            raw = json.loads((out / item["normalized_ground_truth"]).read_text())["results"]
            printed = json.loads((out / item["ground_truth"]).read_text())["results"]
            if name == "omitted":
                assert all(r["reference_range"] == "" and r["flag"] is None for r in raw + printed)
            else:
                assert any(r["reference_range"] for r in raw)
            rows.extend({k: v for k, v in r.items() if k not in {"flag", "reference_range"}} for r in raw)
        targets.append(rows)
    assert targets[0] == targets[1]


def test_prior_default_resolution_and_explicit_extended_vocabulary(tmp_path):
    out = tmp_path / "extended"
    result = cli(out, "--format", "pdf", "--include-heuristic-numerics")
    assert result.returncode == 0, result.stderr
    settings = json.loads((out / "metadata.json").read_text())
    assert settings["dpi"] == 300 and settings["numeric_vocabulary"] == "extended_simulation"
    assert settings["reference_policy"] == "omitted"


def test_pdf_only_does_not_load_image_engines(tmp_path):
    probe = (
        "import sys, typing; import generate_complete_synthetic as g; "
        "assert 'degradation_engine' in typing.get_type_hints(g._pdf_to_degraded_image); "
        "assert g.main(sys.argv[1:]) == 0; "
        "assert not {'cv2','numpy','pypdfium2'}.intersection(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, "--count", "1", "--workers", "1",
         "--format", "pdf", "--output", str(tmp_path / "pdf")],
        cwd=SCRIPT.parent, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_automatic_workers_respect_image_cost_and_explicit_override(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert 1 <= generator.parse_args(["--dpi", "300"]).workers <= 4
    assert generator.parse_args(["--dpi", "600"]).workers == 1
    assert generator.parse_args(["--dpi", "600", "--workers", "6"]).workers == 6


def test_custom_metadata_can_include_rank_as_exported_text(tmp_path):
    import shutil

    data = tmp_path / "inputs"
    shutil.copytree(generator._DATA_DIR, data)
    (data / "loinc_metadata.json").write_text('{"6690-2":{"COMMON_TEST_RANK":"1"}}')
    result = cli(tmp_path / "run", "--data-dir", str(data), "--count", "1", "--format", "pdf")
    assert result.returncode == 0, result.stderr
