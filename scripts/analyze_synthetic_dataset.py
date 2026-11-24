"""Measure coverage and report composition from a schema-v2 generator dataset."""

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import sys

if __package__:
    from .lib.dataset_integrity import decode_json, read_json
else:  # Direct invocation as python scripts/analyze_synthetic_dataset.py.
    from lib.dataset_integrity import decode_json, read_json


def read_object(path):
    if path.is_symlink():
        raise ValueError(f"Expected a regular dataset file: {path.name}")
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return obj


def natural(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    return value


def analyze(dataset, rare_at_most=5):
    """Count actual visible labels, retaining one report and aggregate counters."""
    natural(rare_at_most, "--rare-at-most")
    meta = read_object(dataset / "metadata.json")
    schema = meta.get("schema_version")
    if (
        type(schema) is not int
        or schema != 2
        or meta.get("format") not in ("pdf", "jpg", "both")
    ):
        raise ValueError(
            "Expected a schema-v2 generator dataset and a known media format"
        )
    coverage = read_object(dataset / "coverage_report.json")
    expected = coverage.get("loinc_counts")
    if not isinstance(expected, dict) or not expected:
        raise ValueError(
            "Expected a nonempty eligible LOINC vocabulary in coverage_report.json"
        )
    for code, value in expected.items():
        if not re.fullmatch(r"[0-9]+-[0-9]", code):
            raise ValueError("Invalid code in the recorded LOINC vocabulary")
        natural(value, "LOINC count")
    universe = sha256("\n".join(sorted(expected)).encode()).hexdigest()
    if universe != meta.get("eligible_loincs_sha256"):
        raise ValueError("Eligible LOINC vocabulary differs from metadata fingerprint")

    counts = dict.fromkeys(sorted(expected), 0)
    sizes, lengths, columns, tiers = Counter(), Counter(), Counter(), Counter()
    reports = pages = images = observations = 0
    has_images = meta["format"] != "pdf"
    manifest = dataset / "reports.jsonl"
    if manifest.is_symlink() or (dataset / "ground_truth").is_symlink():
        raise ValueError("Expected the manifest and visible labels inside the dataset")
    with manifest.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            rid = f"RPT_{index:06d}"
            row = decode_json(line)
            label = f"ground_truth/{rid}.json"
            if (
                not isinstance(row, dict)
                or row.get("report_id") != rid
                or row.get("ground_truth") != label
            ):
                raise ValueError(
                    f"Invalid report ID or label path at manifest line {index + 1}"
                )
            page_count = natural(row.get("pages"), f"{rid} pages", 1)
            page_images = row.get("images")
            if not isinstance(page_images, list) or len(page_images) != (
                page_count if has_images else 0
            ):
                raise ValueError(
                    f"{rid}: image path count differs from the declared format or pages"
                )
            for page, image in enumerate(page_images, 1):
                suffix = "" if page == 1 else f"_p{page:03d}"
                if image != f"images/{rid}{suffix}.jpg":
                    raise ValueError(
                        f"{rid}: unexpected or out-of-order manifest image reference"
                    )
            width = natural(row.get("column_count"), f"{rid} column count")
            if width not in (3, 4, 5, 6):
                raise ValueError(f"{rid}: invalid layout column count")
            tier = row.get("tier")
            if (has_images and (type(tier) is not int or tier not in range(5))) or (
                not has_images and tier is not None
            ):
                raise ValueError(f"{rid}: invalid image tier for the dataset format")
            path = dataset / label
            if path.is_symlink():
                raise ValueError(f"{rid}: expected a regular label file")
            raw = path.read_bytes()
            digests = row.get("sha256")
            if (
                not isinstance(digests, dict)
                or digests.get(label) != sha256(raw).hexdigest()
            ):
                raise ValueError(f"{rid}: visible label SHA-256 mismatch")
            target = decode_json(raw.decode("utf-8"))
            results = target.get("results") if isinstance(target, dict) else None
            if not isinstance(results, list) or natural(
                row.get("result_count"), f"{rid} result_count", 1
            ) != len(results):
                raise ValueError(
                    f"{rid}: result count differs from actual visible labels"
                )
            for item in results:
                code = item.get("loinc") if isinstance(item, dict) else None
                if not isinstance(code, str) or code not in counts:
                    raise ValueError(
                        f"{rid}: unknown or missing LOINC in visible labels"
                    )
                counts[code] += 1
            reports += 1
            pages += page_count
            images += len(page_images)
            observations += len(results)
            sizes[page_count] += 1
            lengths[len(results)] += 1
            columns[width] += 1
            if has_images:
                tiers[tier] += 1

    observed = sum(n > 0 for n in counts.values())
    if reports != natural(meta.get("count"), "metadata count", 1) or counts != expected:
        raise ValueError(
            "Measured report or LOINC counts differ from the recorded counts"
        )
    measured = {
        "total_reports": reports,
        "total_pages": pages,
        "total_images": images,
        "total_loincs_in_dict": len(counts),
        "unique_loincs_covered": observed,
    }
    for field, value in measured.items():
        if natural(coverage.get(field), field) != value:
            raise ValueError(f"{field} differs from the measured dataset")
    most_frequent = sorted(
        ((code, n) for code, n in counts.items() if n),
        key=lambda pair: (-pair[1], pair[0]),
    )[:10]
    top_observations = sum(n for _, n in most_frequent)
    return {
        "reports": reports,
        "pages": pages,
        "images": images,
        "observations": observations,
        "page_count_source": "manifest.pages",
        "observation_source": "visible labels",
        "eligible_codes": len(counts),
        "eligible_loincs_sha256": universe,
        "observed_codes": observed,
        "missing_codes": [c for c, n in counts.items() if n == 0],
        "rare_at_most": rare_at_most,
        "rare_codes": [c for c, n in counts.items() if 0 < n <= rare_at_most],
        "loinc_counts": counts,
        "top_10_concentration": {
            "counts": dict(most_frequent),
            "observations": top_observations,
            "total_observations": observations,
            "share_percent": round(100 * top_observations / observations, 2),
        },
        "pages_per_report": dict(sorted(sizes.items())),
        "observations_per_report": dict(sorted(lengths.items())),
        "reports_by_column_count": dict(sorted(columns.items())),
        "reports_by_image_tier": dict(sorted(tiers.items())),
    }


def bar_table(title, denominator, bins, peak=None):
    """Show exact counts alongside bars with an optional shared maximum."""
    peak = peak or max((n for _, n in bins), default=0) or 1
    items = "".join(
        f'<tr><th scope="row">{escape(str(label))}</th><td>{n:,}</td>'
        '<td><svg aria-hidden="true" viewBox="0 0 100 8" preserveAspectRatio="none">'
        '<rect width="100" height="8" fill="#e3ebee"/>'
        f'<rect width="{100 * n / peak:.3f}" height="8" fill="#22626c"/></svg></td></tr>'
        for label, n in bins
    )
    return (
        f"<section><h2>{escape(title)}</h2><p>{escape(denominator)}</p><table>"
        f'<thead><tr><th scope="col">Group</th><th scope="col">Count</th><th scope="col">Distribution</th></tr></thead>'
        f"<tbody>{items}</tbody></table></section>"
    )


def html_report(data):
    total = data["eligible_codes"]
    missing = len(data["missing_codes"])
    rare = len(data["rare_codes"])
    threshold = data["rare_at_most"]
    groups = [("Unobserved", missing)]
    if threshold:
        groups.append((f"1–{threshold} observations", rare))
    groups.append((f"More than {threshold} observations", total - missing - rare))
    top = data["top_10_concentration"]
    sections = [
        bar_table(
            "Most frequent LOINC codes",
            f"Top {len(top['counts'])} codes account for {top['observations']:,} of "
            f"{top['total_observations']:,} observations ({top['share_percent']:.2f}%).",
            list(top["counts"].items()),
        ),
        bar_table(
            "Coverage of eligible codes", f"{total:,} eligible LOINC codes.", groups
        ),
        bar_table(
            "Least frequent LOINC codes",
            "Up to ten codes, including zeros, on the same scale as the most frequent codes. Ties use code order.",
            sorted(data["loinc_counts"].items(), key=lambda pair: (pair[1], pair[0]))[
                :10
            ],
            peak=max(top["counts"].values()),
        ),
    ]
    for key, title in [
        ("pages_per_report", "Pages per report"),
        ("observations_per_report", "Observations per report"),
        ("reports_by_column_count", "Table columns per report"),
        ("reports_by_image_tier", "Image tier by report"),
    ]:
        if data[key]:
            sections.append(
                bar_table(
                    title, f"{data['reports']:,} reports.", list(data[key].items())
                )
            )
    style = (
        "body{font:16px/1.5 Inter,Arial,sans-serif;color:#18303b;max-width:70rem;margin:auto;padding:2rem}"
        "main{display:grid;grid-template-columns:1fr 1fr;gap:2rem}section{min-width:0}"
        "@media(max-width:50rem){main{grid-template-columns:1fr}}@media print{section{break-inside:avoid}}"
        "h2{font-size:1.25rem;margin-bottom:0}p{margin:.4rem 0 1rem}table{width:100%;border-collapse:collapse}"
        "th,td{text-align:left;padding:.25rem;border-bottom:1px solid #dce4e7}tbody th{font-weight:400}"
        "svg{width:100%;min-width:4rem;height:.65rem}td:nth-child(2){font-variant-numeric:tabular-nums}"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Synthetic dataset coverage</title><style>{style}</style></head><body><h1>Synthetic dataset coverage</h1>"
        f"<p>{data['observations']:,} visible observations · {data['reports']:,} reports · {data['pages']:,} manifest pages</p>"
        "<p>Counts describe generated labels and manifest settings. Media files and OCR quality are outside this analysis.</p>"
        f"<main>{''.join(sections)}</main></body></html>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Generated schema-v2 dataset directory",
    )
    parser.add_argument(
        "--rare-at-most",
        type=int,
        default=5,
        help="Maximum observations for a rare, observed code (default 5)",
    )
    parser.add_argument(
        "--html", type=Path, help="Also write a standalone chart to a new HTML file"
    )
    args = parser.parse_args()
    try:
        summary = analyze(args.dataset, args.rare_at_most)
        if args.html:
            chart = html_report(summary)
            with args.html.open("x", encoding="utf-8") as stream:
                stream.write(chart)
        json.dump(summary, sys.stdout, indent=2)
        print()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
