"""Public demo release checks, independent of the historical training collection."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_demo_integrity_and_generated_files(tmp_path):
    from scripts.lib.report_composer import ReportComposer

    data_dir = ROOT / "src/paddle_ocr/dictionaries/data"
    ranges = json.loads((data_dir / "reference_ranges.json").read_text())
    composer = ReportComposer(str(data_dir))
    assert set(composer.eligible_loincs) == set(ranges)
    assert len(ranges) == 32
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(ROOT / "generate_complete_synthetic.py"),
                    "--count", "8", "--workers", "1", "--tiers", "1,0,0,0,0",
                    "--seed", "29", "--output", str(out)], cwd=tmp_path, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_synthetic_dataset.py"),
                    "--dataset", str(out), "--expected-count", "8"], cwd=tmp_path, check=True)
    reports = [json.loads(line) for line in (out / "reports.jsonl").read_text().splitlines()]
    assert len(reports) == 8
    assert all(r["pages"] == len(r["images"]) >= 1 for r in reports)


def test_demo_lipid_panel_uses_its_supported_ldl_observation():
    from scripts.lib.report_composer import ReportComposer
    from scripts.lib.correlation_engine import DERIVED_FORMULAS

    composer = ReportComposer(str(ROOT / "src/paddle_ocr/dictionaries/data"))
    lipid = composer._enriched_panels["perfil_lipidico"]["loincs"]
    assert set(lipid) == {"2093-3", "2571-8", "2085-9", "2089-1"}
    assert "2089-1" not in {formula.target_loinc for formula in DERIVED_FORMULAS}


def test_demo_reports_use_simulation_identifiers_and_only_select_supported_panels():
    import random
    import re
    from scripts.lib.report_composer import CoverageTracker, ReportComposer

    composer = ReportComposer(str(ROOT / "src/paddle_ocr/dictionaries/data"))
    coverage = CoverageTracker()
    patient_ids = set()
    for index in range(64):
        report = composer.compose(f"RPT_{index:06}", random.Random(index), coverage)
        assert re.fullmatch(r"SIM-CASO-[0-9A-F]{16}", report.patient_name)
        assert report.patient_id == report.patient_name.replace("CASO", "ID")
        assert re.fullmatch(r"SIM-CLIN-[0-9A-F]{16}", report.doctor_name)
        assert report.patient_id not in patient_ids
        patient_ids.add(report.patient_id)
    assert set(coverage.panel_counts) <= set(composer._enriched_panels)


def test_demo_lipid_components_fit_the_reported_total_without_deriving_ldl():
    import random
    from decimal import Decimal
    from scripts.lib.report_composer import ReportComposer

    composer = ReportComposer(str(ROOT / "src/paddle_ocr/dictionaries/data"))
    codes = ["2093-3", "2085-9", "2089-1", "2571-8"]
    distinct_from_calculated = 0
    for seed in range(100):
        _, rows = composer._ce.generate_correlated_report(codes, random.Random(seed))
        assert set(rows) == set(codes)
        total, hdl, ldl, triglycerides = [
            Decimal(rows[code].value_formatted.replace(",", ".")) for code in codes
        ]
        assert hdl + ldl <= total
        distinct_from_calculated += hdl + ldl + triglycerides / 5 != total
        for code in codes:
            row = rows[code]
            assert float(Decimal(row.value_formatted.replace(",", "."))) == row.value
            assert composer._vg.within_simulation_bounds(code, row.value)
            assert row.flag == composer._vg._compute_flag(row.value, row.ref_low, row.ref_high)

    assert distinct_from_calculated >= 95


def test_incompatible_cholesterol_envelopes_omit_an_unfillable_component(tmp_path):
    import random
    from shutil import copytree
    from scripts.lib.report_composer import ReportComposer

    data = copytree(ROOT / "src/paddle_ocr/dictionaries/data", tmp_path / "data")
    path = data / "reference_ranges.json"
    ranges = json.loads(path.read_text())
    # Keep the total and HDL envelopes intact; even the largest allowed total
    # cannot accommodate an LDL lower bound of 400 mg/dL.
    ranges["2089-1"].update(min=400, max=410)
    path.write_text(json.dumps(ranges))
    composer = ReportComposer(str(data))
    for seed in range(10):
        _, rows = composer._ce.generate_correlated_report(
            ["2093-3", "2085-9", "2089-1"], random.Random(seed)
        )
        assert set(rows) == {"2093-3", "2085-9"}
        assert rows["2085-9"].value <= rows["2093-3"].value
        _, single = composer._ce.generate_correlated_report(["2089-1"], random.Random(seed))
        assert 400 <= single["2089-1"].value <= 410


def test_release_inputs_must_be_regular_files(tmp_path, monkeypatch):
    import pytest
    from standalone_release import build_source_release as builder

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    regular = tmp_path / "example.txt"
    regular.write_text("public fixture")
    assert builder.source_file("example.txt") == regular
    (tmp_path / "alias.txt").symlink_to(regular)
    with pytest.raises(ValueError, match="ordinary release file"):
        builder.source_file("alias.txt")


def test_alias_owners_include_other_codes():
    from scripts.lib.build_terminology import record_owners, remove_ambiguous_names

    owners = {}
    record_owners(owners, ["Sólo ejemplo", "Shared"], "requested")
    record_owners(owners, ["SHARED"], "outside")
    terms = {"requested": {"names": {"en": ["Sólo ejemplo", "Shared"]}}}
    remove_ambiguous_names(terms, owners)
    assert terms == {"requested": {"names": {"en": ["Sólo ejemplo"]}}}


def test_release_builder_refuses_changed_demo_inputs(tmp_path, monkeypatch):
    import pytest
    from standalone_release import build_source_release as builder

    original = builder.source_file
    copy = tmp_path / "ranges.json"
    copy.write_text("{}")
    def source(name):
        if name == builder.DATA_DIR + "/reference_ranges.json":
            return copy
        return original(name)
    monkeypatch.setattr(builder, "source_file", source)
    destination = tmp_path / "candidate.zip"
    with pytest.raises(ValueError, match="Public demo fixture differs"):
        builder.archive(tmp_path, destination)
    assert not destination.exists()
