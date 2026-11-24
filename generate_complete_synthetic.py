#!/usr/bin/env python3
"""Generate synthetic Spanish-language laboratory documents and matched training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

from scripts.lib.report_composer import ReportComposer, ReportContent, ReportTest, CoverageTracker
from scripts.lib.template_engine import TemplateEngine
from scripts.lib.ground_truth_writer import GroundTruthWriter
from scripts.lib.dataset_output import file_sha256, staged_output


# ---------------------------------------------------------------------------
# Data directory (contains reference_ranges.json, loinc_metadata.json, etc.)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _PROJECT_ROOT / "src" / "paddle_ocr" / "dictionaries" / "data"


GENERATOR_SOURCES = {
    "generate_complete_synthetic.py": Path(__file__),
    **{f"scripts/lib/{name}.py": _PROJECT_ROOT / "scripts" / "lib" / f"{name}.py"
       for name in ("report_composer", "template_engine", "value_generator",
                    "correlation_engine", "synonym_engine", "degradation_engine",
                    "ground_truth_writer", "dataset_output", "terminology")},
}


# ---------------------------------------------------------------------------
# PDF renderer -- standalone ReportLab renderer reading from ReportContent
# ---------------------------------------------------------------------------

def _bold_font(family: str) -> str:
    """Return the bold variant name for a ReportLab built-in font."""
    _BOLD_MAP = {
        "Helvetica": "Helvetica-Bold",
        "Times-Roman": "Times-Bold",
        "Courier": "Courier-Bold",
    }
    return _BOLD_MAP.get(family, f"{family}-Bold")


def _table_columns(template: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    from scripts.lib.template_engine import _cols_5

    return tuple(template.get("columns") or _cols_5())


def _printed_row(test: ReportTest, panel: str, column_count: int) -> dict[str, Any]:
    """LOINC is the semantic target. All other fields transcribe visible cells."""
    def text(value: str) -> str:
        return " ".join(value.split())

    return {
        "test_name": text(test.display_name),
        "value": text(test.value),
        "unit": text(test.unit),
        "reference_range": text(test.ref_range) if column_count >= 4 else "",
        "flag": test.flag if column_count >= 5 else None,
        "loinc": test.loinc,
        "panel": text(panel),
    }


def visible_ground_truth(report: ReportContent, template_config: dict[str, Any]) -> dict[str, Any]:
    """Follow the table's panel/row order and the selected layout's visible fields."""
    column_count = len(_table_columns(template_config))
    sections = template_config.get("show_sections", True)
    return {"results": [
        _printed_row(test, panel if sections else "", column_count)
        for panel, tests in report.panels for test in tests
    ]}


def _wrap_cell(value: str, width: float, font_name: str, size: float) -> list[str]:
    if not value:
        return [""]
    from reportlab.lib.textsplit import wordSplit
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfbase import pdfmetrics

    # ReportLab can draw Greek through its Symbol fallback. Its final fallback
    # substitutes a square for unsupported characters, which would corrupt labels.
    if not value.isascii():
        font = pdfmetrics.getFont(font_name)
        for used_font, _ in pdfmetrics.unicode2T1(value, [font] + font.substitutionFonts):
            if used_font.fontName == "ZapfDingbats":
                raise ValueError(f"Unsupported glyph in table cell {value!r}")
    # Canonical cells that already fit need no word-by-word measuring.
    if value == " ".join(value.split()) and pdfmetrics.stringWidth(value, font_name, size) < width:
        return [value]
    lines = []
    for line in simpleSplit(value, font_name, size, width):
        if pdfmetrics.stringWidth(line, font_name, size) <= width:
            lines.append(line)
        else:
            pieces = wordSplit(line, width, font_name, size)
            if all(pdfmetrics.stringWidth(part, font_name, size) <= width + 1e-6 for _, part in pieces):
                lines.extend(part for _, part in pieces)
                continue
            # ReportLab permits hanging punctuation at a line end. In a table
            # that can place a literal colon from the assay in the value cell.
            # Reflow only the overlong token for which its normal wrapping fails.
            start, used = 0, 0.0
            for index, char in enumerate(line):
                advance = pdfmetrics.stringWidth(char, font_name, size)
                if advance > width + 1e-6:
                    raise ValueError(f"Table cell too narrow for glyph {char!r}")
                if used + advance > width + 1e-6:
                    lines.append(line[start:index])
                    start, used = index, 0.0
                used += advance
            lines.append(line[start:])
    return lines or [""]


def render_simple_pdf(
    report: ReportContent,
    template_config: dict[str, Any],
    output_path: Path,
    *,
    coordinate_pages: list[dict[str, Any]] | None = None,
) -> int:
    """Render a lab report PDF using ReportLab.

    Handles header with lab name/address/patient info, table with test results
    grouped by panel, alternating row backgrounds, column alignment, flag
    colouring, and footer with page numbers.

    Args:
        report: Complete report content dataclass.
        template_config: Template configuration dict from TemplateEngine.
        output_path: Destination path for the PDF file.
        coordinate_pages: Optional list receiving the table geometry for each page.

    Returns:
        Number of pages in the rendered PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas

    w, h = A4
    c = rl_canvas.Canvas(str(output_path), pagesize=A4, invariant=1)

    # -- Unpack template config --
    margins = template_config.get("margins", {"top": 60, "bottom": 50, "left": 60, "right": 50})
    left = margins["left"]
    right_edge = w - margins["right"]
    usable_w = right_edge - left
    top = h - margins["top"]
    bottom = margins["bottom"]

    cs = template_config.get("color_scheme", {})
    primary = cs.get("primary", (0.2, 0.2, 0.2))
    text_color = cs.get("text", (0.0, 0.0, 0.0))
    alt_row_color = cs.get("alt_row", (0.96, 0.96, 0.96))

    fc = template_config.get("font_config", {})
    font_family = fc.get("family", "Helvetica")
    bold_family = _bold_font(font_family)
    header_size = fc.get("header_size", 14)
    body_size = fc.get("body_size", 9)

    row_height = body_size * template_config.get("row_height_factor", 1.6)
    table_style = template_config.get("table_style", "alternating")
    show_sections = template_config.get("show_sections", True)

    page_notice = "DOCUMENTO SINTÉTICO DE ENTRENAMIENTO"
    if template_config.get("reference_policy") == "simulated":
        page_notice += " · INTERVALOS Y ALERTAS SIMULADOS"
    y = top
    page = 1
    result_index = 0
    if coordinate_pages is not None:
        from reportlab.pdfbase.pdfmetrics import getAscentDescent

        ascent, descent = getAscentDescent(font_family, body_size)
        # Distribute line leading around the body's ascent and descent. This
        # uses the actual baseline of the first painted line (after any heading).
        above_baseline = (row_height + ascent + descent) / 2
        coordinate_pages.append({"page": page, "width": w, "height": h, "rows": []})

    def new_page() -> None:
        nonlocal y, page
        # Footer before page break
        c.setFont(font_family, 7)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(left, bottom - 15, f"Pagina {page}")
        c.drawRightString(right_edge, bottom - 15, page_notice)
        c.showPage()
        page += 1
        if coordinate_pages is not None:
            coordinate_pages.append({"page": page, "width": w, "height": h, "rows": []})
        y = top

    # === HEADER ===
    c.setFont(bold_family, header_size)
    c.setFillColorRGB(*primary)
    c.drawString(left, y, report.lab_name)
    y -= header_size + 2

    c.setFont(font_family, max(body_size - 1, 6))
    c.setFillColorRGB(*text_color)
    c.drawString(left, y, report.lab_address)
    y -= body_size + 4

    # Separator line
    c.setStrokeColorRGB(*primary)
    c.setLineWidth(1)
    c.line(left, y, right_edge, y)
    y -= 14

    # Patient info block
    c.setFont(font_family, body_size)
    c.drawString(left, y, f"Paciente: {report.patient_name}")
    c.drawString(left + usable_w * 0.55, y, f"Edad: {report.patient_age} anos")
    y -= body_size + 2
    c.drawString(left, y, f"ID: {report.patient_id}")
    c.drawString(left + usable_w * 0.55, y, f"Sexo: {report.patient_sex}")
    y -= body_size + 2
    c.drawString(left, y, f"Fecha: {report.collection_date}")
    c.drawString(left + usable_w * 0.55, y, f"Dr.: {report.doctor_name}")
    y -= body_size + 10

    # Separator
    c.line(left, y, right_edge, y)
    y -= 12

    # === RESULTS TABLE ===
    # Resolve columns from template; fall back to standard 5-column layout.
    columns = _table_columns(template_config)

    # Compute absolute X positions
    col_widths = [col["width_pct"] * usable_w for col in columns]
    col_x = [left]
    for cw in col_widths[:-1]:
        col_x.append(col_x[-1] + cw)

    # Reserve two points on each side. Fit using the actual bold font,
    # including after template jitter narrows a column. Wrap at the 6pt floor.
    headers = []
    for col, width in zip(columns, col_widths):
        available = width - 4
        name = col["name"]
        measured = c.stringWidth(name, bold_family, body_size)
        size = max(6, min(body_size, body_size * available / measured)) if measured else body_size
        lines = [name] if c.stringWidth(name, bold_family, size) <= available + 1e-6 else _wrap_cell(name, available, bold_family, size)
        headers.append((size, lines))

    def _draw_column_headers() -> None:
        """Draw non-overlapping table headers, including on continuation pages."""
        nonlocal y
        c.setFillColorRGB(*primary)
        for x, (size, lines) in zip(col_x, headers):
            c.setFont(bold_family, size)
            for index, line in enumerate(lines):
                c.drawString(x + 2, y - index * row_height, line)
        y -= (max(len(lines) for _, lines in headers) - 1) * row_height + 2
        c.setStrokeColorRGB(*primary)
        c.line(left, y, right_edge, y)
        y -= row_height

    _draw_column_headers()

    row_idx = 0
    fields = ("test_name", "value", "unit", "reference_range", "flag", "method")
    for panel_name, tests in report.panels:
        panel = panel_name if show_sections else ""
        for panel_idx, test in enumerate(tests):
            row = _printed_row(test, panel, len(columns))
            cells = [
                _wrap_cell(row.get(field) or "", width - 4, font_family, body_size)
                for field, width in zip(fields, col_widths)
            ]
            lines = max(len(cell) for cell in cells)
            heading = bool(row["panel"]) and panel_idx == 0
            if y - (lines + heading) * row_height < bottom:
                new_page()
                _draw_column_headers()
                heading = bool(row["panel"])
            if y - (lines + heading) * row_height < bottom:
                raise ValueError("Table row exceeds the printable height of one page")
            if heading:
                c.setFont(bold_family, body_size + 1)
                c.setFillColorRGB(*primary)
                c.drawString(left + 2, y, row["panel"])
                y -= row_height
                row_idx = 0
            if table_style in ("alternating", "striped") and row_idx % 2 == 1:
                c.setFillColorRGB(*alt_row_color)
                c.rect(left, y - (lines - 0.7) * row_height, usable_w,
                       lines * row_height, fill=1, stroke=0)

            c.setFont(font_family, body_size)
            for i, cell in enumerate(cells):
                align = columns[i].get("align", "left")
                color = (0.8, 0.0, 0.0) if fields[i] == "flag" and row["flag"] else text_color
                c.setFillColorRGB(*color)
                for n, line in enumerate(cell):
                    baseline = y - n * row_height
                    if align == "right":
                        c.drawRightString(col_x[i] + col_widths[i] - 2, baseline, line)
                    elif align == "center":
                        c.drawCentredString(col_x[i] + col_widths[i] / 2, baseline, line)
                    else:
                        c.drawString(col_x[i] + 2, baseline, line)
            if coordinate_pages is not None:
                row_top = h - (y + above_baseline)
                row_bottom = row_top + lines * row_height
                cell_boxes = {
                    field: [x, row_top, x + width, row_bottom]
                    for field, x, width in zip(fields, col_x, col_widths)
                }
                coordinate_pages[-1]["rows"].append({
                    "result_index": result_index, "loinc": test.loinc,
                    "bbox": [col_x[0], row_top, col_x[-1] + col_widths[-1], row_bottom],
                    "cells": cell_boxes,
                })
                result_index += 1
            y -= lines * row_height
            row_idx += 1

    # === FOOTER (final page) ===
    c.setFont(font_family, 7)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(left, bottom - 15, f"Pagina {page}")
    c.drawRightString(right_edge, bottom - 15, page_notice)

    c.save()
    return page


# ---------------------------------------------------------------------------
# PDF to degraded image conversion
# ---------------------------------------------------------------------------

class _PageDegrader(Protocol):
    """Raster backends receive one BGR page and the seeded report RNG."""
    def degrade(self, image: Any, tier: int, rng: random.Random) -> Any: ...


def _pdf_to_degraded_image(
    pdf_path: Path,
    img_dir: Path,
    report_id: str,
    tier: int,
    rng: random.Random,
    degradation_engine: _PageDegrader,
    dpi: int = 300,
    *,
    coordinate_pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rasterize all pages in reading order, keeping the original first-page filename."""
    import cv2
    import pypdfium2 as pdfium

    if coordinate_pages is not None and tier != 0:
        raise ValueError("Coordinate annotations for page images require tier 0")
    image_paths = []
    with closing(pdfium.PdfDocument(str(pdf_path))) as pdf_doc:
        if coordinate_pages is not None and len(coordinate_pages) != len(pdf_doc):
            raise ValueError("Rendered PDF and coordinate page counts differ")
        for page_idx in range(len(pdf_doc)):
            with closing(pdf_doc[page_idx]) as page, closing(
                page.render(scale=dpi / 72, force_bitmap_format=pdfium.raw.FPDFBitmap_BGR)
            ) as bitmap:
                # Keep PDFium's buffer alive through encoding, even for a no-op engine.
                degraded = degradation_engine.degrade(bitmap.to_numpy(), tier, rng)
                suffix = "" if page_idx == 0 else f"_p{page_idx + 1:03d}"
                filename = f"{report_id}{suffix}.jpg"
                if not cv2.imwrite(str(img_dir / filename), degraded, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                    raise OSError(f"Could not write generated JPEG: {img_dir / filename}")
                relative = f"images/{filename}"
                image_paths.append(relative)
                if coordinate_pages is not None:
                    height, width = degraded.shape[:2]
                    if (width, height) != (bitmap.width, bitmap.height):
                        raise ValueError("Clean page coordinates require unchanged raster dimensions")
                    coordinate_pages[page_idx]["image"] = {"path": relative, "width": width, "height": height}
    if not image_paths:
        raise ValueError("Cannot generate training images from a PDF with no pages")
    return {"image_path": image_paths[0], "image_paths": image_paths, "tier": tier}


# ---------------------------------------------------------------------------
# Single-report generator (called per worker)
# ---------------------------------------------------------------------------

@dataclass
class RenderJob:
    report: ReportContent
    template: dict[str, Any]
    seed: int
    tier: int | None
    output: Path
    output_format: str
    dpi: int = 300
    coordinates: bool = False


def _write_json_artifact(path: Path, payload: dict, **options) -> str:
    """Write one report artifact exclusively and hash the exact bytes written."""
    text = json.dumps(payload, ensure_ascii=False, **options)
    # Match the text-mode newlines of datasets generated on this platform.
    if os.linesep != "\n":
        text = text.replace("\n", os.linesep)
    data = text.encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def generate_one(job: RenderJob) -> dict[str, Any]:
    """Render a single composed report into the private staging directory."""
    report = job.report
    report_id = report.report_id
    keep_pdf = job.output_format in ("pdf", "both")
    pdf_path = job.output / ("pdfs" if keep_pdf else ".render") / f"{report_id}.pdf"
    try:
        coordinate_pages = [] if job.coordinates else None
        pages = render_simple_pdf(report, job.template, pdf_path, coordinate_pages=coordinate_pages)
        if pages < 1 or not pdf_path.is_file() or not pdf_path.stat().st_size:
            raise ValueError("PDF renderer produced no pages")
        image_paths: list[str] = []
        if job.tier is not None:
            from scripts.lib.degradation_engine import DegradationEngine

            exported = _pdf_to_degraded_image(
                pdf_path, job.output / "images", report_id, job.tier,
                random.Random(job.seed), DegradationEngine(), dpi=job.dpi, coordinate_pages=coordinate_pages,
            )
            image_paths = exported["image_paths"]
            if len(image_paths) != pages or len(set(image_paths)) != pages:
                raise ValueError(f"Expected one distinct image per page ({pages})")
            for image in image_paths:
                path = job.output / image
                if not path.is_file() or not path.stat().st_size:
                    raise OSError(f"Missing or empty page image: {image}")
        if not keep_pdf:
            pdf_path.unlink()

        visible = visible_ground_truth(report, job.template)
        if not visible["results"]:
            raise ValueError("Report has no visible lab results")
        labels = f"ground_truth/{report_id}.json"
        normalized = f"normalized_ground_truth/{report_id}.json"
        digests = {
            labels: _write_json_artifact(job.output / labels, visible, indent=2),
            normalized: _write_json_artifact(job.output / normalized, report.ground_truth, indent=2),
        }
        for name in image_paths:
            digests[name] = file_sha256(job.output / name)
        pdf_name = pdf_path.relative_to(job.output).as_posix() if keep_pdf else None
        if pdf_name:
            digests[pdf_name] = file_sha256(pdf_path)
        record = {
            "sha256": digests,
            "report_id": report_id,
            "seed": job.seed,
            "pages": pages,
            "tier": job.tier,
            "reference_policy": job.template["reference_policy"],
            "column_count": len(_table_columns(job.template)),
            "show_sections": bool(job.template.get("show_sections", True)),
            "images": image_paths,
            "pdf": pdf_name,
            "ground_truth": labels,
            "normalized_ground_truth": normalized,
            "profile": report.profile_name,
            "region": report.region,
            "result_count": len(visible["results"]),
        }
        if coordinate_pages is not None:
            name = f"coordinates/{report_id}.json"
            sidecar = {
                "schema_version": 1, "report_id": report_id, "coordinate_space": "pdf_points_top_left",
                "ground_truth_sha256": digests[labels], "pages": coordinate_pages,
            }
            digests[name] = _write_json_artifact(job.output / name, sidecar, separators=(",", ":"), allow_nan=False)
            record["coordinates"] = name
        return record
    except Exception as exc:
        raise RuntimeError(f"{report_id}: {exc}") from exc


def _work_items(args, output, composer, template_engine):
    coverage = CoverageTracker()
    # A unique seed per report within this run, regardless of corpus size or worker count.
    # Keep JSON seeds exactly representable by common JSON/JavaScript consumers.
    first_seed = random.Random(str(args.seed)).getrandbits(53)
    for index in range(args.count):
        seed = (first_seed + index) % 2**53
        rng = random.Random(seed)
        report = composer.compose(
            f"RPT_{index:06d}", rng, coverage,
            european_format=args.european,
            gap_filling=index >= int(args.count * 0.7),
            simulated_reference_ranges=args.simulated_reference_ranges,
        )
        template = template_engine.select_template_config(rng, region=report.region)
        template["reference_policy"] = "simulated" if args.simulated_reference_ranges else "omitted"
        tier = None if args.format == "pdf" else rng.choices(range(5), weights=args.tiers, k=1)[0]
        yield RenderJob(report, template, seed, tier, output, args.format, args.dpi, args.coordinates)


def _configure_image_worker():
    import cv2

    cv2.setNumThreads(1)


def _rendered_reports(jobs, workers, *, render_images):
    if workers == 1:
        if not render_images:
            yield from map(generate_one, jobs)
            return
        import cv2

        previous = cv2.getNumThreads()
        _configure_image_worker()
        try:
            yield from map(generate_one, jobs)
        finally:
            cv2.setNumThreads(previous)
        return
    # Bound queued reports and yield in report order. Composition owns one coverage
    # tracker for the entire run; workers can render documents in any order.
    # Spawn starts fresh native-library state and also works without Unix sockets.
    with ProcessPoolExecutor(max_workers=workers, initializer=_configure_image_worker if render_images else None,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = iter(jobs)
        pending = deque(pool.submit(generate_one, job) for job in islice(jobs, 2 * workers))
        while pending:
            item = pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(generate_one, job))
            yield item


def _generate(args: argparse.Namespace, output: Path) -> None:
    composer = ReportComposer(str(args.data_dir), include_heuristic_numerics=args.include_heuristic_numerics,
                              name_style=args.name_style)
    eligible = composer.eligible_loincs
    if not eligible:
        raise ValueError("The input catalog contains no eligible LOINCs")
    workers = min(args.count, args.workers)
    directories = ["ground_truth", "normalized_ground_truth"]
    if args.coordinates:
        directories.append("coordinates")
    directories.append(".render" if args.format == "jpg" else "pdfs")
    if args.format != "pdf":
        directories.append("images")
    for name in directories:
        (output / name).mkdir()

    data_hashes = {name: file_sha256(path) for name, path in composer.input_paths.items()}
    source_hashes = {name: file_sha256(path) for name, path in GENERATOR_SOURCES.items()}
    loincs: Counter[str] = Counter()
    synonyms: set[str] = set()
    profiles: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    n_pages = n_images = 0
    started = time.monotonic()
    jobs = _work_items(args, output, composer, TemplateEngine())
    with ExitStack() as stack:
        manifest = stack.enter_context((output / "reports.jsonl").open("x", encoding="utf-8"))
        writer = None
        if args.sft or args.dpo or args.grpo:
            writer = stack.enter_context(GroundTruthWriter(output))
        for index, item in enumerate(_rendered_reports(jobs, workers, render_images=args.format != "pdf"), 1):
            with (output / item["ground_truth"]).open(encoding="utf-8") as stream:
                gt = json.load(stream)
            for row in gt["results"]:
                loincs[row["loinc"]] += 1
                synonyms.add(row["test_name"])
            profiles[item["profile"]] += 1
            regions[item["region"]] += 1
            n_pages += item["pages"]
            n_images += len(item["images"])
            if writer is not None:
                seed = item["seed"]
                negative = writer.generate_dpo_rejected(gt, random.Random(seed)) if args.dpo else None
                writer.write_examples(item["images"], gt, seed, sft=args.sft, dpo_rejected=negative, grpo=args.grpo)
            manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
            if index % max(1, args.count // 20) == 0 or index == args.count:
                print(f"  [{index}/{args.count}]")
    if args.format == "jpg":
        (output / ".render").rmdir()

    unexpected = sorted(loincs.keys() - eligible)
    if unexpected:
        raise ValueError(f"Generated {len(unexpected)} codes outside the eligible registry: {unexpected[:5]}")
    under = sorted(code for code in eligible if loincs[code] < args.min_loinc_coverage)
    if under:
        raise ValueError(
            f"{len(under)} of {len(eligible)} eligible LOINCs appeared fewer than "
            f"{args.min_loinc_coverage} times (examples: {', '.join(under[:5])}). "
            "Increase --count or lower --min-loinc-coverage."
        )
    elapsed = time.monotonic() - started
    coverage = {
        "total_reports": sum(profiles.values()), "failures": 0,
        "total_pages": n_pages, "total_images": n_images,
        "total_loincs_in_dict": len(eligible), "unique_loincs_covered": len(loincs),
        "loinc_coverage_pct": round(len(loincs) * 100 / len(eligible), 2),
        "min_loinc_count": min(loincs[code] for code in eligible),
        "max_loinc_count": max(loincs.values(), default=0),
        "mean_loinc_count": round(sum(loincs.values()) / len(eligible), 2),
        "loinc_counts": {code: loincs[code] for code in sorted(eligible)},
        "unique_synonyms_used": len(synonyms),
        "profiles": dict(profiles), "regions": dict(regions),
        "elapsed_seconds": round(elapsed, 2),
        "reports_per_second": round(args.count / elapsed, 2) if elapsed else 0,
    }
    runtime_versions = {"reportlab": version("reportlab"), "numpy": version("numpy")}
    if args.format != "pdf":
        import cv2
        import pypdfium2 as pdfium

        runtime_versions.update({
            "opencv": cv2.__version__, "Pillow": version("Pillow"),
            "pypdfium2": version("pypdfium2"), "PDFium": str(pdfium.PDFIUM_INFO),
        })
    metadata = {
        "schema_version": 2, "simulation_revision": 4, "count": args.count, "seed": args.seed,
        "format": args.format, "tiers": args.tiers, "dpi": args.dpi, "name_style": args.name_style,
        "reference_policy": "simulated" if args.simulated_reference_ranges else "omitted",
        "profile_kind": "synthetic_generation_scenario",
        "numeric_vocabulary": "extended_simulation" if args.include_heuristic_numerics else "guarded",
        "workers": workers, "requested_workers": args.workers, "european": args.european,
        "sft": args.sft, "dpo": args.dpo, "grpo": args.grpo,
        "min_loinc_coverage": args.min_loinc_coverage, "python_version": sys.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": data_hashes, "source_sha256": source_hashes,
        "runtime_versions": runtime_versions,
        "eligible_loincs_sha256": hashlib.sha256(
            "\n".join(sorted(eligible)).encode("utf-8")
        ).hexdigest(),
    }
    if args.coordinates:
        metadata["coordinates"] = True
    for filename, payload in (("coverage_report.json", coverage), ("metadata.json", metadata)):
        with (output / filename).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(f"  {args.count} reports, {n_pages} pages, {n_images} JPEGs; "
          f"{len(loincs)}/{len(eligible)} eligible LOINCs covered")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=1000, help="Number of reports (default: 1000)")
    parser.add_argument("--output", type=Path, default=Path("analitiques/complete_synthetic"),
                        help="New or empty destination directory")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR,
                        help="Directory containing generator dictionary inputs")
    parser.add_argument("--workers", type=int, help="Parallel render processes. By default use up to 4, or 1 for images above 300 DPI.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300,
                        help="Page-image rasterization resolution in DPI (72–600, default 300)")
    parser.add_argument("--include-heuristic-numerics", action="store_true",
                        help="Include historical heuristic numeric envelopes for synthetic stress testing")
    parser.add_argument("--simulated-reference-ranges", action="store_true",
                        help="Print and export explicitly simulated reference intervals and flags")
    parser.add_argument("--name-style", choices=("routine-es", "catalog"), default="routine-es",
                        help="Print short Spanish lab names for mapped codes (default), "
                             "or rotate same-code terminology labels with catalog")
    parser.add_argument("--format", choices=["pdf", "jpg", "both"], default="both")
    parser.add_argument("--coordinates", action="store_true",
                        help="Write PDF-point row and cell boxes (page images require --tiers 1,0,0,0,0)")
    parser.add_argument("--tiers", default="0.05,0.20,0.35,0.25,0.15",
                        help="Five comma-separated degradation probabilities totaling 1")
    parser.add_argument("--min-loinc-coverage", type=int, default=0,
                        help="Required observations per eligible code; 0 reports coverage only")
    parser.add_argument("--european", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a decimal comma when rendering numbers")
    for mode in ("sft", "dpo", "grpo"):
        parser.add_argument(f"--{mode}", action=argparse.BooleanOptionalAction,
                            help=f"Emit {mode.upper()} JSONL (default: enabled for JPEG output)")
    args = parser.parse_args(argv)
    if args.workers is None:
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        limit = 1 if args.format != "pdf" and args.dpi > 300 else 4
        args.workers = max(1, min(available - 2, limit))
    if args.count < 1 or args.workers < 1 or args.min_loinc_coverage < 0:
        parser.error("--count and --workers must be positive; --min-loinc-coverage must be nonnegative")
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    if args.count > 2**53:
        parser.error("--count exceeds the number of distinct 53-bit report seeds")
    try:
        args.tiers = [float(x) for x in args.tiers.split(",")]
    except ValueError:
        parser.error("--tiers must contain five numeric probabilities")
    if (len(args.tiers) != 5 or any(not math.isfinite(v) or v < 0 for v in args.tiers)
            or not math.isclose(sum(args.tiers), 1.0, abs_tol=1e-8)):
        parser.error("--tiers must contain five finite nonnegative probabilities totaling 1")
    if args.coordinates and args.format != "pdf" and any(weight > 0 for weight in args.tiers[1:]):
        parser.error("--coordinates for JPEG images requires --tiers 1,0,0,0,0")
    for mode in ("sft", "dpo", "grpo"):
        value = getattr(args, mode)
        if value and args.format == "pdf":
            parser.error(f"--{mode} requires --format jpg or both")
        if value is None:
            setattr(args, mode, args.format != "pdf")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output.absolute()
    try:
        with staged_output(output_dir) as staged:
            _generate(args, staged)
        print(f"Saved dataset to {output_dir}")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
