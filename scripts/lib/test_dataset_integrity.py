"""Exercise the on-disk contract through independent artifact mutations."""
import hashlib
import json
import shutil

import pytest

import generate_complete_synthetic as generator
from scripts.lib.dataset_integrity import check_run

DATA = generator._DATA_DIR


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    root = tmp_path_factory.mktemp("integrity") / "run"
    assert generator.main(["--output", str(root), "--seed", "42", "--count", "1", "--workers", "1",
                           "--format", "both", "--tiers", "1,0,0,0,0", "--sft", "--dpo", "--grpo"]) == 0
    return root


def test_valid_run(clean):
    summary = check_run(clean, DATA, 0, verify_source=True)
    assert summary["reports"] == 1 and summary["pages"] == summary["images"] >= 1


@pytest.mark.parametrize("mutation,reason", [
    ("pair", "Name or unit"), ("unit", "Name or unit"), ("literal", "Visible result"),
    ("jpeg", "digest"), ("dangling", "page-image paths"), ("dpo", "distinct"),
    ("training", "target differs"), ("coverage", "observed"), ("extra", "Unexpected files"),
    ("inputs", "input digest"), ("source", "source differs"),
])
def test_rejects_damaged_run(tmp_path, clean, mutation, reason):
    root = tmp_path / "run"
    shutil.copytree(clean, root)
    manifest = root / "reports.jsonl"
    rec = read(manifest)
    if mutation in {"pair", "unit", "literal"}:
        relative = rec["ground_truth"]
        path = root / relative
        gt = read(path)
        row = gt["results"][0]
        row[{"pair": "test_name", "unit": "unit", "literal": "value"}[mutation]] = "WRONG"
        write(path, gt)
        rec["sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(manifest, rec)
    elif mutation == "jpeg":
        with (root / rec["images"][0]).open("ab") as stream:
            stream.write(b"changed but still decodable")
    elif mutation == "extra":
        (root / "ground_truth" / "extra.json").write_text("{}")
    elif mutation == "dangling":
        rec["images"].append("images/absent.jpg")
        write(manifest, rec)
    elif mutation in ("dpo", "training"):
        p = root / ("dpo_train.jsonl" if mutation == "dpo" else "sft_train.jsonl")
        record = read(p)
        if mutation == "dpo":
            record["rejected_response"] = record["messages"][-1]["content"]
        else:
            record["messages"][-1]["content"] = "{}"
        write(p, record)
    elif mutation == "coverage":
        p = root / "coverage_report.json"
        stats = read(p)
        stats["unique_loincs_covered"] += 1
        write(p, stats)
    else:
        p = root / "metadata.json"
        metadata = read(p)
        hashes = metadata["input_sha256" if mutation == "inputs" else "source_sha256"]
        hashes[next(iter(hashes))] = "0" * 64
        write(p, metadata)
    with pytest.raises(ValueError, match=reason):
        check_run(root, DATA, 0, verify_source=True)


def test_rejects_jointly_impossible_differential():
    from scripts.lib.dataset_integrity import check_projection
    raw, visible = [], []
    for code in ("770-8", "736-9", "5905-5", "713-8", "706-2"):
        row = {"loinc": code, "test_name": code, "unit": "%", "value": "21.0",
               "reference_range": "10.0 - 80.0", "flag": None, "panel": "CBC"}
        raw.append(row)
        visible.append({**row, "value": "21,00", "reference_range": "10,0 - 80,0"})
    with pytest.raises(ValueError, match="differential percentages"):
        check_projection(visible, raw, 5, True, True)


@pytest.mark.parametrize("mutation,reason", [
    ("dpi", "dimensions disagree"), ("missing", "policy metadata"), ("downgrade", "revision|Eligible code set|Input hash inventory"), ("flag", "forbids"),
    ("reference", "forbids"), ("manifest", "reference policy"), ("notice", "synthetic training notice"),
])
def test_modern_policy_is_enforced_after_digest_recomputation(tmp_path, clean, mutation, reason):
    root = tmp_path / "run"
    shutil.copytree(clean, root)
    path = root / "metadata.json"
    meta = read(path)
    manifest = root / "reports.jsonl"
    rec = read(manifest)
    if mutation in {"flag", "reference"}:
        key = rec["normalized_ground_truth"]
        doc = read(root / key)
        row = doc["results"][0]
        row["flag" if mutation == "flag" else "reference_range"] = "A" if mutation == "flag" else "1 - 2"
        write(root / key, doc)
        rec["sha256"][key] = hashlib.sha256((root / key).read_bytes()).hexdigest()
    elif mutation == "notice":
        from reportlab.pdfgen.canvas import Canvas
        key = rec["pdf"]
        replacement = Canvas(str(root / key))
        for _ in range(rec["pages"]):
            replacement.drawString(36, 36, "Example report")
            replacement.showPage()
        replacement.save()
        rec["sha256"][key] = hashlib.sha256((root / key).read_bytes()).hexdigest()
    elif mutation == "manifest":
        rec["reference_policy"] = "simulated"
    elif mutation == "downgrade":
        del meta["simulation_revision"]
    elif mutation == "missing":
        del meta["numeric_vocabulary"]
    else:
        meta["dpi"] = 144
    write(manifest, rec)
    write(path, meta)
    with pytest.raises(ValueError, match=reason):
        check_run(root, DATA, 0)


def test_legacy_v2_projection_still_accepts_simulated_intervals(tmp_path):
    root = tmp_path / "old"
    data = tmp_path / "data"
    data.mkdir()
    write(data / "reference_ranges.json", {"718-7": {"unit":"g/dL","min":1,"max":30}})
    write(data / "loinc_metadata.json", {})
    write(data / "panel_definitions.json", {"panels":[]})
    assert generator.main(["--output", str(root), "--data-dir", str(data), "--count", "1", "--format", "pdf",
                           "--include-heuristic-numerics", "--simulated-reference-ranges"]) == 0
    path = root / "metadata.json"
    meta = read(path)
    for key in ("numeric_vocabulary", "reference_policy", "profile_kind", "dpi", "simulation_revision"):
        del meta[key]
    del meta["input_sha256"]["simulation_policy"]
    write(path, meta)
    manifest = root / "reports.jsonl"
    rec = read(manifest)
    del rec["reference_policy"]
    write(manifest, rec)
    assert check_run(root, data, 0)["reports"] == 1


@pytest.mark.parametrize(("dilution", "expected"), [("1:40", None), ("1:320", "H"), ("1:640", "H")])
def test_titre_flags_follow_printed_strict_threshold(dilution, expected):
    from random import Random
    from unittest.mock import patch
    from scripts.lib.dataset_integrity import check_projection
    from scripts.lib.value_generator import ValueGenerator
    vg = ValueGenerator.__new__(ValueGenerator)
    vg._ranges = {"titre": {"type": "titre"}}
    rng = Random(42)
    with patch.object(rng, "choice", return_value=dilution):
        result = vg._generate_semiquantitative("titre", rng, None, False)
    assert result.ref_formatted == "< 1:320"
    assert result.flag == expected
    raw = {"value": dilution, "reference_range": "< 1:320", "flag": expected, "unit": "",
           "loinc": "titre", "test_name": "titre", "panel": "test"}
    check_projection([raw], [raw], 5, True, False, simulation_revision=2)
    wrong = {**raw, "flag": "L"}
    with pytest.raises(ValueError, match="titre flag"):
        check_projection([wrong], [wrong], 5, True, False, simulation_revision=2)


def test_single_titre_without_comparator_has_no_flag():
    from random import Random
    from unittest.mock import patch
    from scripts.lib.value_generator import ValueGenerator
    vg = ValueGenerator.__new__(ValueGenerator)
    vg._ranges = {"titre": {"type": "titre"}}
    with patch.object(vg, "_parse_titre_series", return_value=["1:10"]):
        result = vg._generate_semiquantitative("titre", Random(4), None, False)
    assert (result.value, result.ref_formatted, result.flag) == ("1:10", "1:10", None)


def test_confirmed_erroneous_282_terms_cannot_be_generated_even_with_custom_ranges(tmp_path, monkeypatch):
    from scripts.lib import value_generator as values
    terms = {code: {"scale": "Qn", "property": "Naric", "units": [unit]}
             for code, unit in [("107370-9", "/uL"), ("107371-7", "uL")]}
    monkeypatch.setattr(values, "load_terms", lambda: terms)
    ranges = tmp_path / "ranges.json"
    write(ranges, {code: {"min": 1, "max": 2, "unit": term["units"][0]} for code, term in terms.items()})
    for extended in (True, False):
        vg = values.ValueGenerator(str(ranges), "", include_heuristic_numerics=extended)
        assert vg.loinc_codes == []
        for code in (*terms, "114503-6", "114504-4"):
            assert not vg.has_loinc(code)


@pytest.mark.parametrize("code", ["1385-4", "19568-5", "42242-8", "42493-7"])
def test_nonactive_283_terms_are_excluded_in_both_modes(code, tmp_path, monkeypatch):
    from scripts.lib import value_generator as values
    monkeypatch.setattr(values, "load_terms", lambda: {code: {"scale":"Ord","property":"PrThr","units":[]}})
    ranges = tmp_path / "ranges.json"
    write(ranges, {code: {"scale":"Ord"}})
    for extended in (False, True):
        vg = values.ValueGenerator(str(ranges), "", include_heuristic_numerics=extended)
        assert not vg.has_loinc(code)


def test_answer_policy_cannot_be_widened_by_local_ranges_or_numeric_mode(tmp_path, monkeypatch):
    import random
    from scripts.lib import value_generator as values
    terms = {code: {"scale":"Ord", "property":"PrThr", "units":[]} for code in ("bound","unbound")}
    monkeypatch.setattr(values, "load_terms", lambda: terms)
    monkeypatch.setattr(values, "load_simulation_policy", lambda: {"binary_answer_lists":{"bound":"LL360-9"}})
    ranges = tmp_path / "ranges.json"
    write(ranges, {code: {"answers":["Heterozygous", "Wild type"]} for code in terms})
    for extended in (False, True):
        vg = values.ValueGenerator(str(ranges), "", include_heuristic_numerics=extended)
        assert vg.loinc_codes == ["bound"]
        assert {vg.generate("bound", random.Random(seed)).value for seed in range(50)} == {"Positivo", "Negativo"}


def test_source_alias_policy_preserves_qualifiers_and_medical_symbols(monkeypatch):
    from scripts.lib import terminology
    from scripts.lib.build_terminology import display_key
    unsafe = "Cristales de urato., Sedimento"
    qualified = "Cristales de urato amorfo, Sedimento"
    monkeypatch.setattr(terminology, "load_simulation_policy", lambda: {"excluded_display_names":{"12454-5":[unsafe]}})
    assert terminology.simulation_names("12454-5", {"names":{"mexico":[unsafe], "spain":[qualified]}}) == {"spain":[qualified]}
    assert display_key(unsafe) == display_key("Cristales de urato , Sedimento")
    assert display_key("RBC # Bld") != display_key("RBC Bld")
    assert display_key("1.05") != display_key("105")


def test_unsupported_strip_and_titre_models_abstain_with_custom_ranges(tmp_path, monkeypatch):
    from scripts.lib import value_generator as values
    terms = {"5803-2":{"scale":"Qn","property":"LsCnc","units":["[pH]"]},
             "5811-5":{"scale":"Qn","property":"SpGrav","units":["1"]},
             "71707-4":{"scale":"SemiQn","property":"Titr","units":[]}}
    monkeypatch.setattr(values, "load_terms", lambda: terms)
    ranges = tmp_path / "ranges.json"
    write(ranges, {code: {"min":1,"max":2,"unit":t["units"][0] if t["units"] else ""} for code,t in terms.items()})
    for extended in (False, True):
        assert values.ValueGenerator(str(ranges), "", include_heuristic_numerics=extended).loinc_codes == []


def test_expected_count_failure_does_not_announce_success(clean, capsys):
    from scripts import validate_synthetic_dataset as validator

    assert validator.main(["--dataset", str(clean), "--expected-count", "2"]) == 1
    output = capsys.readouterr()
    assert '"passed": true' not in output.out
    assert "Expected 2 reports, verified 1" in output.err


def test_checks_jpeg_pixels_after_matching_digest(tmp_path, clean):
    root = tmp_path / "run"
    shutil.copytree(clean, root)
    manifest = root / "reports.jsonl"
    record = read(manifest)
    name = record["images"][0]
    page = root / name
    image = page.read_bytes()
    page.write_bytes(image[:len(image) // 2])
    record["sha256"][name] = hashlib.sha256(page.read_bytes()).hexdigest()
    write(manifest, record)
    with pytest.raises(ValueError, match="truncated"):
        check_run(root, DATA, 0)


def test_current_binary_reference_and_flag_are_reconciled():
    from scripts.lib.dataset_integrity import check_projection

    base = {"test_name": "Simulated binary", "value": "Positivo", "unit": "", "reference_range": "Negativo",
            "flag": "A", "loinc": "demo", "panel": ""}
    for value, reference, flag in (("Positivo", "Negativo", "A"), ("Negativo", "Negativo", None)):
        row = {**base, "value": value, "reference_range": reference, "flag": flag}
        check_projection([row], [row], 5, False, True, "simulated", 4)
    for value, reference, flag in (("Positivo", "Negativo", None), ("Negativo", "Negativo", "A"),
                                   ("Positivo", "Positivo", "A")):
        row = {**base, "value": value, "reference_range": reference, "flag": flag}
        with pytest.raises(ValueError, match="binary"):
            check_projection([row], [row], 5, False, True, "simulated", 4)
        # Revisions before the binary answer policy retain their old rules.
        check_projection([row], [row], 5, False, True, "simulated", 3)


@pytest.mark.parametrize(("field", "invalid"), [("seed", None), ("seed", True), ("seed", 2**53),
                                                ("tier", 6), ("tier", False), ("result_count", True)])
def test_current_manifest_scalar_types(tmp_path, clean, field, invalid):
    root = tmp_path / "run"
    shutil.copytree(clean, root)
    path = root / "reports.jsonl"
    record = read(path)
    record[field] = invalid
    write(path, record)
    with pytest.raises(ValueError, match=field):
        check_run(root, DATA, 0)


def test_legacy_plausibility_handles_sign_grouping_and_nonfinite():
    from scripts.validate_synthetic_dataset import check_value_plausibility

    def warnings(value, low, high):
        return check_value_plausibility({"results": [{"loinc": "test", "value": value}]},
                                        {"test": {"min": low, "max": high}}, "test.json")

    assert not warnings("-7.5", -10, -5)
    assert not warnings("-75", -10, -5)
    assert warnings("-101", -10, -5)
    assert warnings("-0.1", -10, -5)
    assert warnings("1,234.50", 1, 10)
    assert warnings("1.234,50", 1, 10)
    assert not warnings("1.234", 1, 10)  # Historical normalized JSON uses decimal points.
    for value in ("NaN", "Infinity", "-inf", "1e400"):
        assert "finite" in warnings(value, 1, 10)[0]


def test_legacy_cli_reports_bad_row_fields_and_object_stream_errors(tmp_path, capsys):
    from scripts import validate_synthetic_dataset as validator

    (tmp_path / "ground_truth").mkdir()
    (tmp_path / "images").mkdir()
    shutil.copyfile(DATA.parents[3] / "examples/report.jpg", tmp_path / "images/RPT_000000.jpg")
    # Preserve counts even if the printed report only retains 50 details.
    rows = [{"test_name": "Test", "value": 1, "loinc": [], "panel": {}} for _ in range(40)]
    rows.append({"loinc": "718-7", "value": None, "test_name": " "})
    write(tmp_path / "ground_truth/RPT_000000.json", {"results": rows})
    (tmp_path / "sft_train.jsonl").write_text("null\n")
    assert validator.main(["--dataset", str(tmp_path)]) == 1
    report = read(tmp_path / "validation_report.json")
    assert not report["passed"] and report["total_files"] == 1
    assert report["schema_errors"] == 82 and len(report["schema_error_details"]) == 50
    assert report["jsonl_errors"] == 1
    assert report["training_data"]["sft_train.jsonl"] == {"lines": 1, "errors": 1}
    assert "RESULT: VALIDATION FAILED" in capsys.readouterr().out


@pytest.mark.parametrize("code", ["2085-9", "2089-1"])
@pytest.mark.parametrize("european", [True, False])
@pytest.mark.parametrize("policy", ["omitted", "simulated"])
def test_rebalanced_cholesterol_display(code, european, policy):
    from scripts.lib.dataset_integrity import check_projection

    ref = "100.0 - 200.0" if policy == "simulated" else ""
    printed_ref = "100 - 200" if policy == "simulated" else ""
    total = {"test_name": "Total", "loinc": "2093-3", "unit": "mg/dL", "value": "200.0",
             "reference_range": ref, "flag": None, "panel": ""}
    component = {**total, "test_name": "Component", "loinc": code, "value": "125.0"}
    printed_total = {**total, "value": "200", "reference_range": printed_ref}
    tenth = "125,0" if european else "125.0"
    printed_component = {**component, "value": tenth, "reference_range": printed_ref}

    def check(raw, visible):
        check_projection(visible, raw, 5, False, european, policy, 4)

    check([total, component], [printed_total, printed_component])
    check([total, component], [printed_total, {**printed_component, "value": "125"}])

    def reject(raw, visible):
        with pytest.raises(ValueError, match="Visible result"):
            check(raw, visible)

    reject([component], [printed_component])  # No compatible total for rebalancing.
    reject([{**total, "unit": "mmol/L"}, component], [{**printed_total, "unit": "mmol/L"}, printed_component])
    for changes in ({"loinc": "2571-8"}, {"unit": "mmol/L"}):
        reject([total, {**component, **changes}], [printed_total, {**printed_component, **changes}])
    reject([total, {**component, "value": "125.0000000000000001"}], [printed_total, printed_component])
    reject([total, component], [printed_total, {**printed_component, "value": "125.0" if european else "125,0"}])
    reject([total, component], [printed_total, {**printed_component, "value": "126,0" if european else "126.0"}])


def test_training_jsonl_rejects_nonstandard_numbers_and_keeps_scanning_legacy(tmp_path):
    from scripts.lib.dataset_integrity import json_lines
    from scripts.validate_synthetic_dataset import check_jsonl_integrity

    path = tmp_path / "sft_train.jsonl"
    path.write_bytes(b'{"ok": true}\n{"nested": [NaN]}\n{"bad": "\xff"}\n{"ok": true}\n')
    count, errors = check_jsonl_integrity(path)
    assert count == 4 and len(errors) == 2
    assert "sft_train.jsonl:2" in errors[0] and "sft_train.jsonl:3" in errors[1]
    with pytest.raises(ValueError, match="sft_train.jsonl:2: invalid JSON"):
        list(json_lines(path))


def test_legacy_orphan_media_without_format_metadata(tmp_path):
    from scripts.validate_synthetic_dataset import validate_dataset
    (tmp_path / "ground_truth").mkdir()
    (tmp_path / "images").mkdir()
    write(tmp_path / "ground_truth/report.json", {"results": [{"loinc": "718-7", "test_name": "Hb", "value": "7"}]})
    for name in ("report.jpg", "orphan.jpg"):
        (tmp_path / "images" / name).write_bytes(b"synthetic fixture")
    report = validate_dataset(tmp_path)
    assert not report["passed"] and any("no matching ground truth" in error for error in report["failures"])


@pytest.mark.parametrize("bad", [[], {"profiles": ["fake"]}, {"regions": {"demo": "1"}},
                                 {"templates": {"demo": True}}, {"profiles": {"demo": -1}}])
def test_legacy_malformed_coverage_returns_failed_receipt(tmp_path, bad):
    from scripts.validate_synthetic_dataset import validate_dataset
    (tmp_path / "ground_truth").mkdir()
    (tmp_path / "images").mkdir()
    write(tmp_path / "ground_truth/report.json", {"results": [{"loinc": "718-7", "test_name": "Hb", "value": "7"}]})
    (tmp_path / "images/report.jpg").write_bytes(b"synthetic fixture")
    write(tmp_path / "coverage_report.json", bad)
    report = validate_dataset(tmp_path)
    assert not report["passed"] and any("coverage_report.json" in error for error in report["failures"])
    assert not read(tmp_path / "validation_report.json")["passed"]


@pytest.mark.parametrize("field,invalid", [
    ("schema_version", 2.0), ("min_loinc_coverage", False), ("min_loinc_coverage", -1),
])
def test_rejects_invalid_numeric_metadata(tmp_path, clean, field, invalid):
    root = tmp_path / "run"
    shutil.copytree(clean, root)
    path = root / "metadata.json"
    metadata = read(path)
    metadata[field] = invalid
    write(path, metadata)
    with pytest.raises(ValueError, match="schema version|minimum coverage"):
        check_run(root, DATA, 0)


@pytest.mark.parametrize("field", ["total_reports", "failures", "loinc_counts", "profiles"])
def test_rejects_booleans_equivalent_to_recorded_counts(tmp_path, clean, field):
    root = tmp_path / "run"
    shutil.copytree(clean, root)
    path = root / "coverage_report.json"
    coverage = read(path)
    if isinstance(coverage[field], dict):
        code = next(k for k, n in coverage[field].items() if n in (0, 1))
        coverage[field][code] = bool(coverage[field][code])
    else:
        coverage[field] = bool(coverage[field])
    write(path, coverage)
    with pytest.raises(ValueError, match=field):
        check_run(root, DATA, 0)


def test_negative_coverage_has_a_cli_error_message(clean, capsys):
    from scripts.validate_synthetic_dataset import main
    assert main(["--dataset", str(clean), "--min-loinc", "-1"]) == 1
    assert "negative" in capsys.readouterr().err
