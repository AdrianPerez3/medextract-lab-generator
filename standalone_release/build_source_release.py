"""Build the allowlisted source package using only the official LOINC release and original demo fixtures."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.lib.build_terminology import TABLE, VARIANTS, build, display_key  # noqa: E402 (direct CLI)

SOURCE_FILES = (
    ".gitignore",
    ".github/workflows/test.yml",
    "CONTRIBUTING.md",
    "docs/usage.md",
    "docs/benchmark.md",
    "docs/benchmarks/ocr-punctuation.md",
    "docs/benchmarks/ocr-punctuation-crops.png",
    "docs/benchmarks/ocr-punctuation-results.json",
    "docs/benchmarks/paired-photocopy.md",
    "docs/benchmarks/paired-photocopy-crops.png",
    "docs/benchmarks/paired-photocopy-results.json",
    "docs/benchmarks/paired-ocr-terminology.json",
    "docs/benchmarks/paired-photocopy-replay.zip",
    "examples/report.pdf",
    "examples/report.jpg",
    "examples/visible.json",
    "examples/normalized.json",
    "LICENSE",
    "standalone_release/build_source_release.py",
    "generate_complete_synthetic.py",
    "scripts/package_training_data.py",
    "scripts/analyze_synthetic_dataset.py",
    "scripts/score_row_associations.py",
    "scripts/score_ocr_benchmark.py",
    "scripts/generate_paired_ocr_challenge.py",
    "scripts/make_paired_views.py",
    "scripts/score_paired_ocr.py",
    "scripts/validate_synthetic_dataset.py",
    "scripts/lib/__init__.py",
    "scripts/lib/build_terminology.py",
    "scripts/lib/correlation_engine.py",
    "scripts/lib/dataset_integrity.py",
    "scripts/lib/dataset_output.py",
    "scripts/lib/degradation_engine.py",
    "scripts/lib/ground_truth_writer.py",
    "scripts/lib/jsonl_split.py",
    "scripts/lib/report_composer.py",
    "scripts/lib/synonym_engine.py",
    "scripts/lib/template_engine.py",
    "scripts/lib/terminology.py",
    "scripts/lib/resources/routine_es_names.json",
    "scripts/lib/test_routine_es_names.py",
    "scripts/lib/value_generator.py",
    "scripts/lib/test_degradation_engine.py",
    "scripts/lib/test_ground_truth_writer.py",
    "scripts/lib/test_template_engine.py",
    "scripts/lib/test_render_synthetic.py",
    "scripts/lib/test_render_coordinates.py",
    "scripts/lib/test_coordinate_integrity.py",
    "scripts/lib/test_score_row_associations.py",
    "scripts/lib/test_score_ocr_benchmark.py",
    "scripts/lib/test_generate_paired_ocr_challenge.py",
    "scripts/lib/test_make_paired_views.py",
    "scripts/lib/test_score_paired_ocr.py",
    "scripts/lib/test_paired_score_replay.py",
    "scripts/lib/test_dataset_integrity.py",
    "scripts/lib/test_package_training_cli.py",
    "scripts/lib/test_analyze_synthetic_dataset.py",
    "scripts/lib/test_generate_complete_cli.py",
)
RELEASE_FILES = (
    "README.md", "TERMINOLOGY.md", "LOINC_short_license.txt",
    "requirements.txt", "requirements-test.txt",
)
# Only the reviewed public fixtures belong in this archive. Custom input packs
# may contain local terminology, provenance or other material meant to stay local.
DEMO_SHA256 = {
    "reference_ranges.json": "67692fb7e852aac3c8cc64369a5ee1713283cc4e82fc46079859f6ed6f890940",
    "loinc_metadata.json": "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    "panel_definitions.json": "10c3cf7cbe48e7b2c7f6e8cd4c69c41c311ce66d0bc90cd5fdab365b001200c8",
}
DATA_DIR = "src/paddle_ocr/dictionaries/data"
PREFIX = "medextract-lab-generator"
DEMO = ROOT / DATA_DIR
LICENSE_FILE = "licenses/LoincLicense_5.8.txt"
LICENSE_SHA256 = "05eb5069f8352b0513fe8c911d8a432e8ddc2d09f999b6f153617f2afd16ad85"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as src:
        yield from csv.DictReader(src)


def demo_catalog(release, work):
    """Check ambiguity against the official release before projecting the small demo."""
    ranges = DEMO / "reference_ranges.json"
    catalog_path = work / "official-index.json"
    build(release, ranges, catalog_path)
    catalog = json.loads(catalog_path.read_bytes())
    requested = json.loads(ranges.read_bytes())
    if set(catalog["terms"]) != set(requested):
        raise ValueError("The official terminology build did not resolve every demo code")
    terms = catalog["terms"]
    # A fully specified LOINC name uses the six official axes, including time
    # and scale. Spain's variant often supplies only these axes.
    axes = ("COMPONENT", "PROPERTY", "TIME_ASPCT", "SYSTEM", "SCALE_TYP", "METHOD_TYP")
    additions = {}
    for lang, filename in VARIANTS.items():
        for row in csv_rows(release / filename):
            code = row["LOINC_NUM"]
            if code in terms and all(row[k] for k in axes[:5]):
                label = ":".join(row[k] for k in axes)
                if label.isprintable():
                    additions[(code, lang)] = label
    # Validate candidate FQNs against *every* official row, before selecting
    # the demo subset. Exclude accent/case-equivalent labels shared by codes.
    owners = {display_key(s): set() for s in additions.values()}
    # Retain original display fields; drop constructed partial aliases.
    for lang, filename in {"en": TABLE, **VARIANTS}.items():
        display_column = "DisplayName" if lang == "en" else "LinguisticVariantDisplayName"
        for row in csv_rows(release / filename):
            code = row["LOINC_NUM"]
            originals = {row[k] for k in ("SHORTNAME", "LONG_COMMON_NAME", display_column)}
            fully_specified = ":".join(row[k] for k in axes)
            for label in originals | {fully_specified}:
                key = display_key(label)
                if key in owners:
                    owners[key].add(code)
            term = terms.get(code)
            if term is None or lang not in term["names"]:
                continue
            term["names"][lang] = [s for s in term["names"][lang] if s in originals]
            if not term["names"][lang]:
                del term["names"][lang]
    for (code, lang), label in additions.items():
        if owners[display_key(label)] == {code}:
            labels = terms[code]["names"].setdefault(lang, [])
            if label not in labels:
                labels.append(label)
    for code, term in terms.items():
        if not term["names"].get("en") or requested[code]["unit"] not in term["units"]:
            raise ValueError(f"Could not verify an official display name and matching unit for {code}")
    catalog["terms"] = terms
    return (json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def source_file(name):
    path = ROOT / name
    if path.resolve() != path or not path.is_file():
        raise ValueError(f"Expected an ordinary release file: {name}")
    return path


def archive(release, destination):
    names = (*SOURCE_FILES, *RELEASE_FILES, "scripts/lib/test_release_demo.py",
             LICENSE_FILE)
    files = {name: source_file(name).read_bytes() for name in names}
    for name, expected in DEMO_SHA256.items():
        relative = f"{DATA_DIR}/{name}"
        content = source_file(relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Public demo fixture differs from its reviewed version: {name}")
        files[relative] = content
    if hashlib.sha256(files[LICENSE_FILE]).hexdigest() != LICENSE_SHA256:
        raise ValueError("Bundled LOINC 2.82 license checksum mismatch")
    with tempfile.TemporaryDirectory() as work:
        files["scripts/lib/resources/loinc_2_82.json"] = demo_catalog(release, Path(work))
    # Only carry policy entries which apply to the packaged demo catalog.
    catalog = json.loads(files["scripts/lib/resources/loinc_2_82.json"])["terms"]
    policy_file = "scripts/lib/resources/simulation_policy.json"
    policy = json.loads(source_file(policy_file).read_bytes())
    policy["binary_answer_lists"] = {code: value for code, value in policy["binary_answer_lists"].items() if code in catalog}
    holds = {}
    for code, labels in policy["excluded_display_names"].items():
        if code in catalog:
            available = {s for names in catalog[code]["names"].values() for s in names}
            selected = [s for s in labels if s in available]
            if selected:
                holds[code] = selected
    policy["excluded_display_names"] = holds
    files[policy_file] = (json.dumps(policy, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(blob).hexdigest()}  {name}\n" for name, blob in sorted(files.items())
    ).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staged = Path(tmp) / "source.zip"
        with zipfile.ZipFile(staged, "w") as z:
            for name, blob in sorted(files.items()):
                entry = zipfile.ZipInfo(f"{PREFIX}/{name}", (1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                z.writestr(entry, blob, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(staged, destination)
    print(f"{len(files)} files; SHA-256 {hashlib.sha256(destination.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loinc-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.loinc_release, args.output)
