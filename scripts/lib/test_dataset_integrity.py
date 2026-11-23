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
