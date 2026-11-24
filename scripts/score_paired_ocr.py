"""Compare OCR decisions on the same printed result in clean and photocopied reports."""

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

if __package__:
    from .score_ocr_benchmark import report_records
    from .score_row_associations import _prepare_pages, locate_prediction
else:
    from score_ocr_benchmark import report_records
    from score_row_associations import _prepare_pages, locate_prediction


CONDITIONS = ("clean", "photocopy")
STATES = ("exact", "wrong", "review", "missing", "ambiguous")
NUMERIC = re.compile(r"[+\-−]?[0-9]+(?:[.,][0-9]+)?")


def text_error(expected, read):
    """Classify exact display text first, then plain signed decimal values."""
    if expected == read:
        return "exact"
    if not all(NUMERIC.fullmatch(s) for s in (expected, read)):
        return "other_text"
    a, b = (Decimal(s.replace(",", ".").replace("−", "-")) for s in (expected, read))
    if a == b:
        return (
            "decimal_mark"
            if expected.replace(",", ".") == read.replace(",", ".")
            else "numeric_format"
        )
    if not a.is_zero() and not b.is_zero():
        # Decimal tuples preserve all input digits, including long strings beyond
        # the active decimal context. Equal significant digits at different
        # magnitudes imply an absolute ratio of 10^k for a nonzero integer k.
        scale_error = a.copy_abs() != b.copy_abs() and significant_digits(
            a
        ) == significant_digits(b)
        if a.is_signed() != b.is_signed():
            return "sign_and_power_of_ten" if scale_error else "sign"
        if scale_error:
            return "power_of_ten"
    return "other_numeric"


def significant_digits(number):
    digits = number.as_tuple().digits
    end = len(digits)
    while digits[end - 1] == 0:
        end -= 1
    return digits[:end]


def read_records(path):
    with path.open("rb") as source:
        return report_records(source, path.name)


def labels(root, report):
    def sidecar(field):
        name = report.get(field)
        if not isinstance(name, str) or not name:
            raise ValueError(f"missing {field} path")
        path = Path(name)
        resolved = (root / path).resolve()
        if path.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError(f"{field} path must be inside the manifest directory")
        data = resolved.read_bytes()
        digests = report.get("sha256")
        if digests is not None and (
            not isinstance(digests, dict)
            or digests.get(name) != sha256(data).hexdigest()
        ):
            raise ValueError(f"{name}: manifest SHA-256 mismatch")
        return data

    coords = json.loads(sidecar("coordinates"))
    visible = sidecar("ground_truth")
    if not isinstance(coords, dict) or coords.get("report_id") != report["report_id"]:
        raise ValueError("coordinate report_id mismatch")
    pages, values = _prepare_pages(coords, visible, "image-pixels")
    images = [page["image"] for page in coords["pages"]]
    if "pages" in report and (
        type(report["pages"]) is not int or report["pages"] != len(images)
    ):
        raise ValueError("manifest page count differs from the coordinate sidecar")
    if "images" in report and report["images"] != [image["path"] for image in images]:
        raise ValueError(
            "manifest image order or path differs from the coordinate sidecar"
        )
    if report.get("sha256") is not None:
        for image in images:
            digest = report["sha256"].get(image["path"])
            if (
                not isinstance(digest, str)
                or not re.fullmatch("[0-9a-f]{64}", digest)
                or image.get("sha256") != digest
            ):
                raise ValueError(
                    f"{image['path']}: manifest and sidecar image SHA-256 mismatch"
                )
    # Match every labeled cell, page size and raster dimension. Each view has
    # different image bytes/paths, even though its printed layout is identical.
    layout = [
        (
            p["page"],
            p["width"],
            p["height"],
            p["rows"],
            p["image"]["width"],
            p["image"]["height"],
        )
        for p in coords["pages"]
    ]
    return (json.loads(visible)["results"], layout), pages, values


def review_anchor(pages, decision):
    """Only assign reviews with one unambiguous original test/value cell pair."""
    sources = {}
    for name in ("baseline", "alternate", "before_punctuation_audit"):
        if decision.get(name) is not None:
            sources[name] = [decision[name]]
    for name in ("candidate_baselines", "candidate_alternates"):
        if decision.get(name):
            sources[name] = decision[name]
    # For a withheld accepted value, its own anchors identify the prior decision.
    locations = set()
    for name, candidates in sources.items():
        if not isinstance(candidates, list):
            raise ValueError(f"{name} must contain a list of predictions")
        for candidate in candidates:
            location = locate_prediction(pages, candidate)
            if (
                name == "before_punctuation_audit"
                or "before_punctuation_audit" not in sources
            ):
                locations.add(location)
    row = None
    if len(locations) == 1:
        test, _, geometry = locations.pop()
        if geometry == "same_row":
            row = test
    strings = {
        name: [p["value_text"] for p in items] for name, items in sources.items()
    }
    return row, {"reason": decision.get("reason"), "sources": strings}


def evaluate(pages, values, document):
    records = [{"accepted": [], "reviews": []} for _ in values]
    decisions = document.get("decisions")
    if not isinstance(decisions, list) or not all(
        isinstance(d, dict) for d in decisions
    ):
        raise ValueError("expected a decisions array of objects")
    submitted = [d.get("selected") for d in decisions if d.get("status") == "accepted"]
    predictions = document.get("predictions")
    if predictions != submitted:
        raise ValueError(
            "predictions must equal the accepted selections in decision order"
        )
    counts = Counter(accepted_predictions=len(submitted))
    geometry = Counter()
    for d in decisions:
        if d.get("status") == "accepted":
            row, _, location = locate_prediction(pages, d["selected"])
            geometry[location] += 1
            if location == "same_row":
                records[row]["accepted"].append(d["selected"]["value_text"])
        elif d.get("status") == "review" and d.get("selected") is None:
            counts["review_proposals"] += 1
            row, proposal = review_anchor(pages, d)
            if row is None:
                counts["unassigned_review_proposals"] += 1
            else:
                records[row]["reviews"].append(proposal)
        else:
            raise ValueError(
                "expected an accepted selection or a review with no selection"
            )
    for record, expected in zip(records, values, strict=True):
        choices = set(record["accepted"])
        if len(choices) > 1:
            state = "ambiguous"
        elif choices:
            record["text_class"] = text_error(expected, choices.pop())
            state = "exact" if record["text_class"] == "exact" else "wrong"
        else:
            state = "review" if record["reviews"] else "missing"
        record["status"] = state
    return records, counts, geometry


def group_pairs(manifest):
    pairs = {}
    group_splits = {}
    for row in manifest.values():
        pair, condition = row.get("pair_id"), row.get("condition")
        if not isinstance(pair, str) or not pair.strip() or condition not in CONDITIONS:
            raise ValueError("expected a pair_id and clean or photocopy condition")
        group, split = row.get("group_id", pair), row.get("split")
        if not all(isinstance(x, str) and x.strip() for x in (group, split)):
            raise ValueError(f"{pair}: expected nonempty group_id and split")
        if group_splits.setdefault(group, split) != split:
            raise ValueError(f"{group}: group appears in different splits")
        views = pairs.setdefault(pair, {})
        if condition in views:
            raise ValueError(f"{pair}: duplicate {condition} view")
        views[condition] = row
    for pair, views in pairs.items():
        if set(views) != set(CONDITIONS):
            raise ValueError(f"{pair}: expected both views")
        if views["clean"].get("group_id", pair) != views["photocopy"].get(
            "group_id", pair
        ):
            raise ValueError(f"{pair}: views in different groups")
    return pairs


def compare(manifest_path, decision_path, baseline_path=None, *, include_rows=True):
    manifest, docs = read_records(manifest_path), read_records(decision_path)
    baseline = read_records(baseline_path) if baseline_path is not None else None
    for source in [docs, baseline]:
        if source is not None and source.keys() != manifest.keys():
            raise ValueError(
                "decision streams must cover every manifest report exactly once"
            )
    root = manifest_path.resolve().parent
    pairs = group_pairs(manifest)
    counts = {
        c: Counter(
            accepted_predictions=0, review_proposals=0, unassigned_review_proposals=0
        )
        for c in CONDITIONS
    }
    geometry = {c: Counter() for c in CONDITIONS}
    states = {c: Counter(dict.fromkeys(STATES, 0)) for c in CONDITIONS}
    errors = {c: Counter() for c in CONDITIONS}
    withheld_errors = {c: Counter() for c in CONDITIONS}
    transitions = {s: dict.fromkeys(STATES, 0) for s in STATES}
    details = []
    paired_rows = 0
    for pair, views in sorted(pairs.items()):
        clean_spec = None
        rows = []
        for condition in CONDITIONS:
            view = views[condition]
            rid = view["report_id"]
            spec, pages, values = labels(root, view)
            if condition == "clean":
                clean_spec = spec
                rows = [
                    {
                        "pair_id": pair,
                        "result_index": i,
                        "loinc": label["loinc"],
                        "test_name": label.get("test_name"),
                        "expected": label["value"],
                    }
                    for i, label in enumerate(spec[0])
                ]
                for page_number, (cells, *_) in enumerate(pages, 1):
                    for index, _, _ in cells:
                        rows[index]["page"] = page_number
            elif clean_spec != spec:
                raise ValueError(
                    f"{pair}: printed labels or layout differ between views"
                )
            records, totals, locations = evaluate(pages, values, docs[rid])
            counts[condition].update(totals)
            geometry[condition].update(locations)
            before = (
                evaluate(pages, values, baseline[rid])[0]
                if baseline is not None
                else None
            )
            for i, record in enumerate(records):
                record["report_id"] = rid
                rows[i][condition] = record
                state = record["status"]
                states[condition][state] += 1
                if state == "wrong":
                    errors[condition][record["text_class"]] += 1
                if before is not None and state == "review":
                    counts[condition][
                        f"baseline_{before[i]['status']}_to_review_rows"
                    ] += 1
                    if before[i]["status"] == "wrong":
                        withheld_errors[condition][before[i]["text_class"]] += 1
        for row in rows:
            transitions[row["clean"]["status"]][row["photocopy"]["status"]] += 1
        paired_rows += len(rows)
        if include_rows:
            details.extend(rows)
    views = {}
    for c in CONDITIONS:
        total = counts[c]
        views[c] = dict(
            total,
            printed_rows=paired_rows,
            states=dict(states[c]),
            accepted_geometry=dict(geometry[c]),
            wrong_text_classes=dict(errors[c]),
            power_of_ten_magnitude_rows=errors[c]["power_of_ten"]
            + errors[c]["sign_and_power_of_ten"],
            opposite_sign_rows=errors[c]["sign"] + errors[c]["sign_and_power_of_ten"],
            review_proposals_per_100_rows=100 * total["review_proposals"] / paired_rows,
        )
        if baseline is not None:
            views[c]["baseline_wrong_classes_to_review"] = dict(withheld_errors[c])
            for state in STATES:
                key = f"baseline_{state}_to_review_rows"
                views[c][key] = total[key]
                views[c][key + "_per_100"] = 100 * total[key] / paired_rows
    result = {
        "report_pairs": len(pairs),
        "paired_rows": paired_rows,
        "views": views,
        "transitions": transitions,
    }
    if include_rows:
        result["rows"] = details
    return result


def main():
    cli = argparse.ArgumentParser(
        description=__doc__,
        epilog="Outputs JSON, including exact source strings for each paired row. Points use original image pixels.",
    )
    cli.add_argument(
        "--manifest", required=True, type=Path, help="Paired reports.jsonl"
    )
    cli.add_argument(
        "--decisions",
        required=True,
        type=Path,
        help="One prediction and decision record per view (JSONL)",
    )
    cli.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline decision stream to count exact or wrong rows withheld for review",
    )
    cli.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate counts and transitions only",
    )
    args = cli.parse_args()
    try:
        result = compare(
            args.manifest, args.decisions, args.baseline, include_rows=not args.summary
        )
    except (OSError, ValueError, KeyError, OverflowError) as exc:
        cli.exit(2, f"{cli.prog}: {exc}\n")
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
