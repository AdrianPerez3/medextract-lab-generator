#!/usr/bin/env python3
"""Generate the fixed Spanish punctuation challenge, with two views of each page."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from shutil import copyfile
from tempfile import TemporaryDirectory

import pypdfium2 as pdfium
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

if __package__:
    from .lib.dataset_output import file_sha256, staged_output
    from .make_paired_views import write_paired_views
else:
    from lib.dataset_output import file_sha256, staged_output
    from make_paired_views import write_paired_views


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 2026112405
LAYOUTS = (
    ("bulletin", "Boletín del laboratorio", "Helvetica", "Helvetica-Bold"),
    ("reference-card", "Hoja de resultados", "Times-Roman", "Times-Bold"),
    ("compact-log", "Registro de analítica", "Helvetica", "Helvetica-Bold"),
    ("monitoring", "Cuadro de seguimiento", "Courier", "Courier-Bold"),
)
COLUMNS = ("test_name", "value", "unit", "previous")


@dataclass(frozen=True)
class Analyte:
    code: str
    name: str
    unit: str
    values: tuple[str, ...]


# These are transcription strings, including deliberate signs and trailing zeroes.
# The codes identify the specified observations; no reference intervals are supplied.
POTASSIUM = Analyte("2823-3", "Potasio sérico", "mmol/L", ("4.4",))
HEMATOCRIT = Analyte("4544-3", "Hematocrito automatizado", "%", ("44",))
BASE_EXCESS = Analyte(
    "1925-7",
    "Exceso de base arterial calculado",
    "mmol/L",
    ("-1.8", "+1.5", "-0.8", "+2.0"),
)
CREATININE = Analyte(
    "2160-0", "Creatinina sérica", "mg/dL", ("0.80", "1.20", "1.00", "1.60")
)
VENOUS_SATURATION = Analyte(
    "2711-0", "Saturación venosa de oxígeno", "%", ("65", "68", "71", "73", "78")
)
EXTRA = (
    Analyte("2345-7", "Glucosa sérica", "mg/dL", ("71", "85", "90", "105")),
    Analyte("2951-2", "Sodio sérico", "mmol/L", ("135", "138", "141", "144")),
    Analyte("1751-7", "Albúmina sérica", "g/dL", ("3.8", "4.0", "4.2", "4.8")),
    Analyte("718-7", "Hemoglobina en sangre", "g/dL", ("11.8", "12.0", "14.6", "16.2")),
    Analyte("3094-0", "Nitrógeno ureico sérico", "mg/dL", ("8", "12", "15", "21")),
)


def make_rows(rng, draw):
    """Balance dot/comma potassium and signed tokens within each ten-report layout."""
    targets = (
        (POTASSIUM, POTASSIUM.values[0], "4.1"),
        (HEMATOCRIT, HEMATOCRIT.values[0], "42"),
        (BASE_EXCESS, BASE_EXCESS.values[draw % len(BASE_EXCESS.values)], "-0.5"),
        (CREATININE, rng.choice(CREATININE.values), "0.90"),
        (VENOUS_SATURATION, rng.choice(VENOUS_SATURATION.values), "70"),
    )
    rows = []
    for index, (test, value, previous) in enumerate(targets):
        # Rotate the base-excess mark independently of its sign.
        comma = (draw // 2 if test == BASE_EXCESS else draw + index) % 2
        rows.append(_result(test, value, previous, comma))
    for index, test in enumerate(rng.sample(EXTRA, 2)):
        value, previous = rng.sample(test.values, 2)
        rows.append(_result(test, value, previous, (draw + index) % 2))
    rng.shuffle(rows)
    return rows


def _result(test, value, previous, comma):
    return {
        "loinc": test.code,
        "test_name": test.name,
        "unit": test.unit,
        "value": value.replace(".", ",") if comma else value,
        "previous": previous.replace(".", ",") if comma else previous,
    }


def draw_page(pair_id, layout, rng, rows):
    """Draw a vector page and collect its row and cell edges in top-left points."""
    family, heading, font, bold = LAYOUTS[layout]
    width, height = letter
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=letter, invariant=True)
    left = rng.randint(30, 52)
    value_left = rng.randint(235, 266)
    unit_left = value_left + rng.randint(65, 81)
    edges = (left, value_left, unit_left, unit_left + 74, width - left)
    top = rng.randint(135, 178)
    row_height = rng.choice((21, 24, 27, 30))
    body_size = rng.choice((8.5, 9.5, 10.5, 11))
    value_pad = rng.choice((2, 4, 8, 14))
    aligned_right = layout != 1

    canvas.setFont(bold, 20)
    canvas.drawString(left, height - 63, heading)
    canvas.setFont(font, 10)
    canvas.drawString(left, height - 87, f"Registro de ensayo  {pair_id[-8:]}")
    canvas.setFillGray(0.9)
    canvas.rect(left, height - top, edges[-1] - left, 23, fill=1, stroke=0)
    canvas.setFillGray(0.12)
    canvas.setFont(bold, 10)
    for x, label in zip(
        edges[:-1], ("Prueba", "Resultado", "Unidad", "Previo"), strict=True
    ):
        canvas.drawString(x + 3, height - top + 7, label)

    geometry = []
    for index, row in enumerate(rows):
        bottom = top + row_height
        if family == "monitoring" and index % 2:
            canvas.setFillGray(0.95)
            canvas.rect(
                left, height - bottom, edges[-1] - left, row_height, fill=1, stroke=0
            )
        canvas.setFillGray(0.12)
        cells = {}
        for field, x0, x1 in zip(COLUMNS, edges[:-1], edges[1:], strict=True):
            cells[field] = [x0, top, x1, bottom]
            size = body_size
            if field == "test_name":
                # Long printed names must stay inside their recorded name cell.
                size = min(
                    size, (x1 - x0 - 6) / canvas.stringWidth(row[field], font, 1)
                )
            canvas.setFont(font, size)
            baseline = height - (top + bottom) / 2 - size * 0.33
            if field == "value" and aligned_right:
                canvas.drawRightString(x1 - value_pad, baseline, row[field])
            else:
                canvas.drawString(x0 + 3, baseline, row[field])
        geometry.append(
            {
                "result_index": index,
                "loinc": row["loinc"],
                "bbox": [left, top, edges[-1], bottom],
                "cells": cells,
            }
        )
        if family in ("bulletin", "reference-card"):
            canvas.setStrokeGray(0.82)
            canvas.setLineWidth(0.4)
            canvas.line(left, height - bottom, edges[-1], height - bottom)
        top = bottom
    canvas.setFont(font, 9)
    canvas.drawString(left, 45, "FICHA SINTÉTICA PARA ENSAYO DE OCR")
    canvas.drawRightString(edges[-1], 45, "Página 1")
    canvas.save()
    return stream.getvalue(), geometry


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_clean_sources(folder, seed):
    for name in ("images", "ground_truth", "coordinates"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    with (folder / "reports.jsonl").open("x", encoding="utf-8") as manifest:
        for layout, (family, *_) in enumerate(LAYOUTS):
            for draw in range(10):
                digest = sha256(
                    f"paired-ocr-v1/{seed}/{layout}/{draw}".encode()
                ).hexdigest()
                rng = random.Random(digest)
                pair_id = "SYNPAIR_" + digest[:16]
                rows = make_rows(rng, draw)
                page_bytes, geometry = draw_page(pair_id, layout, rng, rows)
                image = f"images/{pair_id}.png"
                label = f"ground_truth/{pair_id}.json"
                coordinates = f"coordinates/{pair_id}.json"
                dpi = (108, 144, 180)[(draw + layout) % 3]
                with pdfium.PdfDocument(page_bytes) as document:
                    with (
                        closing(document[0]) as page,
                        closing(page.render(scale=dpi / 72)) as bitmap,
                    ):
                        with bitmap.to_pil() as decoded:
                            decoded.save(folder / image, format="PNG", dpi=(dpi, dpi))
                            raster_width, raster_height = decoded.size
                write_json(folder / label, {"results": rows})
                write_json(
                    folder / coordinates,
                    {
                        "schema_version": 1,
                        "coordinate_space": "pdf_points_top_left",
                        "report_id": pair_id,
                        "ground_truth_sha256": file_sha256(folder / label),
                        "pages": [
                            {
                                "page": 1,
                                "width": letter[0],
                                "height": letter[1],
                                "rows": geometry,
                                "image": {
                                    "path": image,
                                    "width": raster_width,
                                    "height": raster_height,
                                },
                            }
                        ],
                    },
                )
                record = {
                    "report_id": pair_id,
                    "group_id": pair_id,
                    "family": family,
                    "case_mix": "punctuation-challenge",
                    "decimal_mark": "mixed",
                    "split": "test",
                    "source_dpi": dpi,
                    "condition": "clean",
                    "pages": 1,
                    "ground_truth": label,
                    "coordinates": coordinates,
                    "images": [image],
                    "sha256": {
                        path: file_sha256(folder / path)
                        for path in (image, label, coordinates)
                    },
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")


def include_terminology(folder):
    """Keep the ten official names and upstream notices with the synthetic labels."""
    (folder / "licenses").mkdir()
    copyfile(
        PROJECT_ROOT / "docs/benchmarks/paired-ocr-terminology.json",
        folder / "paired-ocr-terminology.json",
    )
    for path in ("LOINC_short_license.txt", "licenses/LoincLicense_5.8.txt"):
        copyfile(PROJECT_ROOT / path, folder / path)


def generate(output, seed=DEFAULT_SEED):
    with staged_output(output) as stage, TemporaryDirectory() as directory:
        clean = Path(directory)
        write_clean_sources(clean, seed)
        result = write_paired_views(clean, stage, seed=seed)
        include_terminology(stage)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        help="New output directory for the 40 pairs. POSIX also permits an existing empty directory.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        result = generate(args.output, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
