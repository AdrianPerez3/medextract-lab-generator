"""Reconcile coordinate receipts through the full on-disk dataset validator."""
import hashlib
import json
import shutil

import pytest

import generate_complete_synthetic as generator
from scripts.lib.dataset_integrity import check_run


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def manifest(root):
    return [json.loads(line) for line in (root / "reports.jsonl").read_bytes().splitlines()]


def save_manifest(root, records):
    (root / "reports.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    result = {}
    for fmt, count in (("both", 1), ("pdf", 8), ("jpg", 1)):
        root = tmp_path_factory.mktemp("coordinates") / fmt
        assert generator.main(["--output", str(root), "--seed", "42", "--count", str(count), "--workers", "1",
                               "--format", fmt, "--tiers", "1,0,0,0,0", "--dpi", "72", "--coordinates"]) == 0
        result[fmt] = root
    return result


@pytest.mark.parametrize("fmt", ["pdf", "jpg", "both"])
def test_coordinate_report_passes_full_validation(clean, fmt):
    root = clean[fmt]
    records = manifest(root)
    report = check_run(root, generator._DATA_DIR, 0, verify_source=True)
    assert report["reports"] == len(records)
    assert report["pages"] == sum(rec["pages"] for rec in records)
    if fmt == "pdf":
        assert any(rec["pages"] > 1 for rec in records)


@pytest.mark.parametrize(("change", "reason"), [
    ("label_digest", "ground truth hash"), ("report_id", "identity"), ("page_order", "page order"),
    ("extent", "page dimensions"), ("image", "image path or dimensions"),
    ("image_path", "image path or dimensions"), ("index", "result index"), ("loinc", "LOINC"),
    ("missing_row", "every visible result"), ("empty_box", "no area"), ("wrong_field", "cell fields"),
    ("split", "partition"), ("tiny_cell_overlap", "partition"), ("tiny_beyond_row", "inside the row"),
    ("tiny_row_overlap", "overlap"), ("nonfinite", "Invalid JSON number"), ("outside", "outside"), ("row_order", "overlap"), ("shift", "PDF text"),
])
def test_rehashed_geometry_errors_fail(tmp_path, clean, change, reason):
    root = tmp_path / "run"
    shutil.copytree(clean["both"], root)
    records = manifest(root)
    rec = records[0]
    path = root / rec["coordinates"]
    receipt = load(path)
    page = receipt["pages"][0]
    row = page["rows"][0]
    if change == "label_digest":
        receipt["ground_truth_sha256"] = "0" * 64
    elif change == "report_id":
        receipt["report_id"] = "RPT_999999"
    elif change == "page_order":
        page["page"] = True  # True is not page 1.
    elif change == "extent":
        page["width"] += 1
    elif change == "image":
        page["image"]["width"] += 1
    elif change == "image_path":
        page["image"]["path"] = "../images/other.jpg"
    elif change == "index":
        row["result_index"] = False
    elif change == "loinc":
        row["loinc"] = page["rows"][1]["loinc"]
    elif change == "missing_row":
        page["rows"].pop()
    elif change == "empty_box":
        row["bbox"][2] = row["bbox"][0]
    elif change == "wrong_field":
        row["cells"]["unused"] = row["cells"].pop("unit")
    elif change == "split":
        row["cells"]["value"][0] += 1
    elif change == "tiny_cell_overlap":
        row["cells"]["value"][0] = row["cells"]["test_name"][2] - 1e-6
    elif change == "tiny_beyond_row":
        row["cells"]["value"][1] = row["bbox"][1] - 1e-6
    elif change == "tiny_row_overlap":
        previous, following = next((a, b) for a, b in zip(page["rows"], page["rows"][1:])
                                   if abs(a["bbox"][3] - b["bbox"][1]) < 1e-8)
        for box in [following["bbox"], *following["cells"].values()]:
            box[1] = previous["bbox"][3] - 1e-6
    elif change == "nonfinite":
        row["bbox"][0] = float("nan")
    elif change == "outside":
        row["bbox"][0] = -1
    elif change == "row_order":
        page["rows"][1]["bbox"][1] = row["bbox"][3] - 1
    elif change == "shift":
        # The row/cell partitions remain valid. They now point to the header.
        offset = row["bbox"][1] - 2
        for box in [row["bbox"], *row["cells"].values()]:
            box[1] -= offset
            box[3] -= offset
    save(path, receipt)
    rec["sha256"][rec["coordinates"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_manifest(root, records)
    with pytest.raises(ValueError, match=reason):
        check_run(root, generator._DATA_DIR, 0)


@pytest.mark.parametrize(("change", "reason"), [
    ("undeclared", "Undeclared coordinates"), ("missing_manifest", "coordinate path"),
    ("wrong_path", "coordinate path"), ("missing_hash", "Digest inventory"),
    ("bad_hash", "File digest"), ("extra", "entries in coordinates"), ("directory", "entries in coordinates"),
    ("symlink", "coordinate path or symlink"), ("declared_warp", "clean tier 0"),
    ("actual_warp", "clean tier 0"),
])
def test_coordinate_declarations_and_inventory(tmp_path, clean, change, reason):
    root = tmp_path / "run"
    shutil.copytree(clean["jpg"], root)
    meta = load(root / "metadata.json")
    records = manifest(root)
    rec = records[0]
    name = rec["coordinates"]
    if change == "undeclared":
        del meta["coordinates"]
        del rec["coordinates"]
        del rec["sha256"][name]
    elif change == "missing_manifest":
        del rec["coordinates"]
    elif change == "wrong_path":
        rec["coordinates"] = "../elsewhere.json"
    elif change == "missing_hash":
        del rec["sha256"][name]
    elif change == "bad_hash":
        rec["sha256"][name] = "0" * 64
    elif change == "extra":
        (root / "coordinates/orphan.json").write_text("{}")
    elif change == "directory":
        (root / "coordinates/untracked").mkdir()
    elif change == "symlink":
        copy = tmp_path / "external.json"
        (root / name).rename(copy)
        (root / name).symlink_to(copy)
    elif change == "declared_warp":
        meta["tiers"] = [0, 0, 1, 0, 0]
    elif change == "actual_warp":
        rec["tier"] = 3
    save(root / "metadata.json", meta)
    save_manifest(root, records)
    with pytest.raises(ValueError, match=reason):
        check_run(root, generator._DATA_DIR, 0)



def test_reassigned_row_fails_physical_pdf_verification(tmp_path, clean):
    root = tmp_path / "run"
    shutil.copytree(clean["pdf"], root)
    records = manifest(root)
    rec = next(record for record in records if record["pages"] > 1)
    path = root / rec["coordinates"]
    receipt = load(path)
    first, second = receipt["pages"][:2]
    row = first["rows"].pop()
    for box in [row["bbox"], *row["cells"].values()]:
        box[1], box[3] = 0.5, 2
    second["rows"].insert(0, row)
    save(path, receipt)
    rec["sha256"][rec["coordinates"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_manifest(root, records)
    with pytest.raises(ValueError, match="PDF text on page 2"):
        check_run(root, generator._DATA_DIR, 0)


def test_clean_weight_accepted_with_generators_rounding_tolerance(tmp_path, clean):
    root = tmp_path / "run"
    shutil.copytree(clean["jpg"], root)
    metadata = load(root / "metadata.json")
    metadata["tiers"] = [0.999999999, 0, 0, 0, 0]
    save(root / "metadata.json", metadata)
    assert check_run(root, generator._DATA_DIR, 0)["images"] == 1
