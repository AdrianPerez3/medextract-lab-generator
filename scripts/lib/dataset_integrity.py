"""Reconcile a schema-v2 generator run from artifacts on disk."""

import hashlib
import json
import math
import re
from collections import Counter
from contextlib import closing
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path


def _reject_constant(value: str):
    raise ValueError(f"Invalid JSON number: {value}")


# The standard library accepts NaN and Infinity unless explicitly disabled.
decode_json = json.JSONDecoder(parse_constant=_reject_constant).decode


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return decode_json(stream.read())


def json_lines(path):
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = decode_json(line.decode("utf-8"))
            except ValueError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{number}: expected a record")
            yield record


def check_training_record(kind, row, images, truth):
    if row["images"] != images:
        raise ValueError(f"{kind} page references differ from the report manifest")
    messages = row["messages"]
    if [m["role"] for m in messages] != (["system", "user"] if kind == "grpo" else ["system", "user", "assistant"]):
        raise ValueError(f"{kind} message roles are invalid")
    if messages[1]["content"].count("<image>") != len(images):
        raise ValueError(f"{kind} image token count differs from page count")
    target = row["solution"] if kind == "grpo" else messages[-1]["content"]
    if decode_json(target) != truth:
        raise ValueError(f"{kind} target differs from the visible ground truth")
    if kind == "dpo":
        rejected = decode_json(row["rejected_response"])
        if not isinstance(rejected, dict) or not isinstance(rejected.get("results"), list) or rejected == truth:
            raise ValueError("DPO needs a distinct, structured negative example")


def check_projection(visible, normalized, columns, sections, european, reference_policy="simulated", simulation_revision=1):
    from .value_generator import ValueGenerator
    from .correlation_engine import _AUTO_DIFFERENTIAL
    differential = {}
    if type(columns) is not int or columns not in (3, 4, 5, 6) or type(sections) is not bool or type(european) is not bool:
        raise ValueError("Invalid report layout")
    def identity(row):
        return (row["loinc"], " ".join(row["test_name"].split()), row["unit"])
    raw = {identity(row): row for row in normalized}
    if len(raw) != len(normalized) or len({row["loinc"] for row in normalized}) != len(raw):
        raise ValueError("Duplicate analyte in a synthetic report")
    has_total_cholesterol = any(row["loinc"] == "2093-3" and row["unit"] == "mg/dL" for row in normalized)
    for row in visible:
        original = raw[identity(row)]
        value, reference = original["value"], original["reference_range"]
        if original["flag"] not in (None, "L", "H", "A"):
            raise ValueError("Unknown normalized flag")
        if reference_policy == "omitted" and (reference != "" or original["flag"] is not None):
            raise ValueError("Omitted reference policy forbids normalized reference intervals and flags")
        if row["unit"]:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("Invalid finite numeric value")
            if reference_policy == "simulated":
                bounds = [float(x) for x in reference.split(" - ")]
                if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                    raise ValueError("Invalid finite numeric reference interval")
                if ValueGenerator._compute_flag(numeric, *bounds) != original["flag"]:
                    raise ValueError("Flag disagrees with normalized value and reference interval")
                reference = ValueGenerator._format_ref_range(*bounds, european)
            if row["loinc"] in _AUTO_DIFFERENTIAL and row["unit"] == "%":
                differential[row["loinc"]] = Decimal(value)
                value = ValueGenerator._format_number_dp(numeric, 2, european)
            else:
                value = ValueGenerator._format_number(numeric, european)
                # Rebalancing prints exact tenths for these two components only
                # when the same report contains total cholesterol in mg/dL.
                if (row["value"] != value and simulation_revision >= 4 and has_total_cholesterol
                        and row["loinc"] in {"2085-9", "2089-1"} and row["unit"] == "mg/dL"
                        and Decimal(original["value"]) == Decimal(f"{numeric:.1f}")):
                    value = ValueGenerator._format_number_dp(numeric, 1, european)
        elif simulation_revision >= 4 and reference_policy == "simulated" and value in {"Negativo", "Positivo"}:
            expected_flag = "A" if value == "Positivo" else None
            if reference != "Negativo" or original["flag"] != expected_flag:
                raise ValueError("Simulated binary reference or flag disagrees with the answer")
        elif simulation_revision >= 2 and reference_policy == "simulated" and (value.startswith("1:") or reference.startswith("< 1:")):
            dilution = re.fullmatch(r"1:([1-9][0-9]*)", value)
            threshold = re.fullmatch(r"< 1:([1-9][0-9]*)", reference)
            if not dilution or (not threshold and reference != value):
                raise ValueError("Invalid simulated dilution or reference")
            flag = "H" if threshold and int(dilution[1]) >= int(threshold[1]) else None
            if original["flag"] != flag:
                raise ValueError("Simulated titre flag disagrees with the printed reference")
        expected = (value, reference if columns >= 4 else "", original["flag"] if columns >= 5 else None,
                    " ".join(original["panel"].split()) if sections else "")
        if (row["value"], row["reference_range"], row["flag"], row["panel"]) != expected:
            raise ValueError(f"Visible result differs from the normalized value or layout for {row['loinc']}")
    total = sum(differential.values())
    if (any(v < 0 or v > 100 for v in differential.values()) or total > 100
            or (len(differential) == len(_AUTO_DIFFERENTIAL) and total != 100)):
        raise ValueError("Automated leukocyte differential percentages are inconsistent")


def check_run(root: Path, data_dir: Path, minimum: int, verify_source: bool = False):
    import pypdfium2
    from PIL import Image
    from .report_composer import ReportComposer, _VETERINARY_KEYWORDS
    from .terminology import CATALOG_PATH, admitted_range, load_terms
    from .value_generator import ERRONEOUS_PLASMA_TERMS, NONACTIVE_283_ADDITIONS, PROJECT_RANGE_SOURCES, UNSUPPORTED_REPORTING_METHODS

    meta = read_json(root / "metadata.json")
    count, fmt = meta["count"], meta["format"]
    if meta["schema_version"] != 2 or type(count) is not int or count <= 0 or fmt not in {"jpg", "pdf", "both"}:
        raise ValueError("Invalid schema version, report count or format")
    policy_keys = {"dpi", "reference_policy", "numeric_vocabulary", "profile_kind"}
    modern = bool(policy_keys.intersection(meta))
    if modern and (not policy_keys.issubset(meta) or meta["reference_policy"] not in {"omitted", "simulated"}
                   or meta["numeric_vocabulary"] not in {"guarded", "extended_simulation"}
                   or meta["profile_kind"] != "synthetic_generation_scenario" or type(meta["dpi"]) is not int
                   or not 72 <= meta["dpi"] <= 600):
        raise ValueError("Invalid render or simulation policy metadata")
    reference_policy = meta.get("reference_policy", "simulated")
    revision = meta.get("simulation_revision", 1)
    if type(revision) is not int or revision not in {1, 2, 3, 4} or (revision >= 2 and not modern):
        raise ValueError("Invalid simulation revision")
    extended = meta.get("numeric_vocabulary") != "guarded"
    try:
        composer = ReportComposer(str(data_dir), include_heuristic_numerics=extended)
    except ValueError as exc:
        # An archived custom input may consist exclusively of a now-excluded
        # term. Its artifact vocabulary is still reconstructed below.
        if revision >= 4 or str(exc) != "No eligible LOINC terms in supplied synthetic ranges":
            raise
        input_paths = {key: data_dir / f"{key}.json" for key in ("reference_ranges", "loinc_metadata", "panel_definitions")}
        input_paths["loinc_2_82"] = CATALOG_PATH
        units, name_keys = {}, {}
    else:
        input_paths = composer.input_paths
        name_keys = {code: {" ".join(label.split()) for label in composer.allowed_names(code)}
                     for code in composer.eligible_loincs}
        units = {code: set(composer.allowed_units(code)) for code in composer.eligible_loincs}
    terms = load_terms()
    if revision < 4:
        # Reconstruct only for reading historical artifacts. No historical
        # policy is exposed to generation or to caller-supplied range files.
        input_paths = {key:path for key,path in input_paths.items() if key != "simulation_policy"}
        ranges = read_json(input_paths["reference_ranges"])
        metadata = read_json(input_paths["loinc_metadata"])
        former = {code for code in terms if terms[code]["scale"] == "Ord"} | UNSUPPORTED_REPORTING_METHODS
        if revision < 3:
            former |= NONACTIVE_283_ADDITIONS
        else:
            former -= NONACTIVE_283_ADDITIONS
        if revision == 1:
            former |= ERRONEOUS_PLASMA_TERMS
        for code in former.intersection(terms, ranges):
            details = metadata.get(code, {})
            description = f"{details.get('COMPONENT','')} {details.get('LONG_COMMON_NAME','')}".lower()
            if any(word in description for word in _VETERINARY_KEYWORDS):
                continue
            entry = admitted_range(terms[code], ranges[code])
            if entry and (not entry["unit"] or extended or ranges[code].get("source") not in PROJECT_RANGE_SOURCES
                          or code in {"5803-2", "5811-5"}):
                units[code] = {entry["unit"]}
        name_keys = {code: {" ".join(name.split()) for labels in terms[code]["names"].values() for name in labels}
                     for code in units}
    eligible = set(units)
    digest = hashlib.sha256("\n".join(sorted(eligible)).encode()).hexdigest()
    if not eligible or meta["eligible_loincs_sha256"] != digest:
        raise ValueError("Eligible code set differs from the recorded input")
    if set(input_paths) != set(meta["input_sha256"]):
        raise ValueError("Input hash inventory differs")
    sources = meta["source_sha256"]
    from generate_complete_synthetic import GENERATOR_SOURCES
    if set(sources) != set(GENERATOR_SOURCES) or any(not re.fullmatch("[0-9a-f]{64}", digest) for digest in sources.values()):
        raise ValueError("Invalid generator source inventory")
    current_value_source = GENERATOR_SOURCES["scripts/lib/value_generator.py"]
    if (modern and revision < 4 and
            hashlib.sha256(current_value_source.read_bytes()).hexdigest() == sources["scripts/lib/value_generator.py"]):
        raise ValueError("Current generator source requires simulation revision 4")
    if verify_source:
        for name, path in GENERATOR_SOURCES.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sources[name]:
                raise ValueError(f"Generator source differs for {name}")
    for key, path in input_paths.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != meta["input_sha256"].get(key):
            raise ValueError(f"Generation input digest differs for {key}")
    streams = {}
    for kind in ("sft", "dpo", "grpo"):
        path = root / f"{kind}_train.jsonl"
        if type(meta[kind]) is not bool or path.exists() != meta[kind]:
            raise ValueError(f"Enabled {kind} flag and file presence disagree")
        if meta[kind]:
            if fmt == "pdf":
                raise ValueError("An image-conditioned export was declared for PDF-only output")
            streams[kind] = json_lines(path)

    code_counts = Counter()
    profiles = Counter()
    regions = Counter()
    seeds = set()
    names = set()
    n_pages = n_images = n_reports = 0
    manifests = json_lines(root / "reports.jsonl")
    for records in zip_longest(manifests, *streams.values()):
        rec = records[0]
        report_id = f"RPT_{n_reports:06d}"
        try:
            if any(item is None for item in records) or rec["report_id"] != report_id:
                raise ValueError("Out-of-order, duplicate or missing record")
            n_reports += 1
            pages = rec["pages"]
            if type(pages) is not int or pages < 1:
                raise ValueError("Invalid page count")
            seed = rec["seed"]
            if type(seed) is not int or seed in seeds or (revision >= 4 and not 0 <= seed < 2**53):
                raise ValueError("Invalid or duplicate report seed")
            seeds.add(seed)
            if revision >= 4:
                tier = rec["tier"]
                if (fmt == "pdf" and tier is not None) or (fmt != "pdf" and (type(tier) is not int or tier not in range(5))):
                    raise ValueError("Invalid degradation tier for output format")
            if type(rec["result_count"]) is not int or rec["result_count"] <= 0:
                raise ValueError("Invalid result_count")
            if ("reference_policy" in rec) != modern or (modern and rec["reference_policy"] != reference_policy):
                raise ValueError("Manifest reference policy disagrees with metadata")
            images = []
            if fmt in ("jpg", "both"):
                for index in range(1, pages + 1):
                    suffix = "" if index == 1 else f"_p{index:03d}"
                    images.append(f"images/{report_id}{suffix}.jpg")
            pdf = f"pdfs/{report_id}.pdf" if fmt in ("both", "pdf") else None
            if rec["pdf"] != pdf or rec["images"] != images:
                raise ValueError("Unexpected PDF or page-image paths")
            if pdf:
                with closing(pypdfium2.PdfDocument(root / pdf)) as doc:
                    if len(doc) != pages:
                        raise ValueError("PDF page count disagrees with manifest")
                    if modern:
                        for index in range(pages):
                            with closing(doc[index]) as page, closing(page.get_textpage()) as textpage:
                                text = textpage.get_text_bounded()
                            if ("DOCUMENTO SINTÉTICO DE ENTRENAMIENTO" not in text or
                                    ("INTERVALOS Y ALERTAS SIMULADOS" in text) != (reference_policy == "simulated")):
                                raise ValueError(f"Incorrect synthetic training notice on PDF page {index + 1}")
            for image in images:
                with Image.open(root / image) as page:
                    if page.format != "JPEG" or min(page.size) < 1:
                        raise ValueError(f"Invalid JPEG {image}")
                    if modern:
                        # PDFium rounds the A4 point dimensions upwards, subject
                        # to floating-point precision at exact pixel boundaries.
                        for dimension, millimeters in zip(page.size, (210, 297)):
                            exact = millimeters * meta["dpi"] / 25.4
                            if not math.ceil(exact - 0.001) <= dimension <= math.ceil(exact + 0.001):
                                raise ValueError(f"JPEG dimensions disagree with DPI for {image}")
                    page.load()  # Decode pixels as well as validating the JPEG header.
            artifacts = images + ([pdf] if pdf else []) + [
                f"ground_truth/{report_id}.json", f"normalized_ground_truth/{report_id}.json"
            ]
            if set(rec["sha256"]) != set(artifacts):
                raise ValueError("Digest inventory differs from expected files")
            for relative in artifacts:
                with (root / relative).open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(file.read()).hexdigest()
                if digest != rec["sha256"][relative]:
                    raise ValueError(f"File digest differs for {relative}")
            targets = {}
            for field in ("ground_truth", "normalized_ground_truth"):
                expected = f"{field}/{report_id}.json"
                if rec[field] != expected:
                    raise ValueError(f"Unexpected {field} path")
                target = read_json(root / expected)
                results = target["results"]
                if not isinstance(results, list) or not results or len(results) != rec["result_count"]:
                    raise ValueError(f"Invalid {field} result count")
                triples = Counter()
                for row in results:
                    code = row["loinc"]
                    if code not in eligible:
                        raise ValueError(f"Ineligible LOINC {code}")
                    if set(row) != {"loinc", "unit", "test_name", "value", "reference_range", "flag", "panel"}:
                        raise ValueError("Invalid result fields")
                    if not all(isinstance(row[key], str) for key in ("unit", "test_name", "reference_range", "panel", "value")) or not row["value"].strip():
                        raise ValueError("Missing or invalid result text")
                    if " ".join(row["test_name"].split()) not in name_keys[code] or row["unit"] not in units[code]:
                        raise ValueError(f"Name or unit is inconsistent with LOINC {code}")
                    if revision >= 4 and terms[code]["scale"] == "Ord" and row["value"] not in {"Positivo", "Negativo"}:
                        raise ValueError(f"Unexpected literal binary answer for {code}")
                    if revision >= 4 and terms[code]["property"] == "Titr" and not re.fullmatch(r"1:[1-9][0-9]*", row["value"]):
                        raise ValueError(f"Invalid synthetic dilution format for {code}")
                    triples[(code, row["unit"], " ".join(row["test_name"].split()))] += 1
                    if field == "ground_truth":
                        code_counts[code] += 1
                        names.add(row["test_name"])
                targets[field] = (target, triples)
            if targets["ground_truth"][1] != targets["normalized_ground_truth"][1]:
                raise ValueError("Visible and normalized analyte identities differ")
            check_projection(targets["ground_truth"][0]["results"], targets["normalized_ground_truth"][0]["results"],
                             rec["column_count"], rec["show_sections"], meta["european"], reference_policy, revision)
            for kind, row in zip(streams, records[1:]):
                check_training_record(kind, row, images, targets["ground_truth"][0])
            profiles[rec["profile"]] += 1
            regions[rec["region"]] += 1
            n_pages += pages
            n_images += len(images)
        except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as exc:
            raise ValueError(f"{report_id}: {exc}") from exc
    if n_reports != count:
        raise ValueError(f"Expected {count} reports, found {n_reports}")
    for folder, expected in {"ground_truth": count, "normalized_ground_truth": count,
                             "pdfs": count if fmt != "jpg" else 0, "images": n_images}.items():
        directory = root / folder
        if directory.exists() and sum(1 for path in directory.iterdir() if path.is_file()) != expected:
            raise ValueError(f"Unexpected files in {folder}")
    minimum = max(minimum, meta["min_loinc_coverage"])
    if any(code_counts[code] < minimum for code in eligible):
        raise ValueError(f"Minimum coverage of {minimum} not reached for every eligible code")
    coverage = read_json(root / "coverage_report.json")
    expected = {"total_reports": count, "failures": 0, "total_pages": n_pages, "total_images": n_images,
                "loinc_counts": {code: code_counts[code] for code in eligible},
                "total_loincs_in_dict": len(eligible), "unique_loincs_covered": len(code_counts),
                "unique_synonyms_used": len(names), "min_loinc_count": min(code_counts[code] for code in eligible),
                "max_loinc_count": max(code_counts.values()),
                "mean_loinc_count": round(sum(code_counts.values()) / len(eligible), 2),
                "loinc_coverage_pct": round(100 * len(code_counts) / len(eligible), 2), "profiles": dict(profiles), "regions": dict(regions)}
    for field, value in expected.items():
        if coverage.get(field) != value:
            raise ValueError(f"coverage_report.json disagrees with observed {field}")
    return {"passed": True, "reports": count, "pages": n_pages, "images": n_images,
            "results": sum(code_counts.values()), "unique_loincs": len(code_counts), "exports": list(streams)}
