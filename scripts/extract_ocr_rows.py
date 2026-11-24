"""Extract test/value point pairs from one page image or a saved Tesseract TSV."""

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .lib.ocr_rows import predict_page
else:
    from lib.ocr_rows import predict_page


TSV_COLUMNS = (
    "level page_num block_num par_num line_num word_num left top width height conf text"
).split()


def read_tsv(stream):
    """Read native, top-left pixel bounds from a single Tesseract page."""
    rows = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
    if next(rows, None) != TSV_COLUMNS:
        raise ValueError("expected the standard Tesseract TSV header")
    size = None
    words = []
    for fields in rows:
        if not fields:
            continue
        if len(fields) != len(TSV_COLUMNS):
            raise ValueError(f"TSV line {rows.line_num}: expected 12 columns")
        level, page_num = fields[:2]
        if page_num != "1" or (level == "1" and size is not None):
            raise ValueError("TSV must contain exactly one page")
        if level not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"TSV line {rows.line_num}: invalid level")
        if level == "1":
            left, top, width, height = map(int, fields[6:10])
            if left or top or min(width, height) <= 0:
                raise ValueError(
                    "TSV page record requires a positive size and top-left origin"
                )
            size = width, height
        elif size is None:
            raise ValueError("expected the TSV page record before OCR words")
        elif level == "5" and fields[11].strip():
            x, y, w, h = map(int, fields[6:10])
            if not (0 <= x < x + w <= size[0] and 0 <= y < y + h <= size[1]):
                raise ValueError(
                    f"TSV line {rows.line_num}: word box outside page or empty"
                )
            words.append((fields[11], [x, y, x + w, y + h]))
    if size is None:
        raise ValueError("missing TSV page record")
    return words, *size


def image_words(path, language):
    """Run Tesseract at native resolution and check that its page matches the image."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ValueError(
            "install Pillow for --image, or replay saved OCR with --tsv"
        ) from exc

    with path.open("rb", buffering=0) as source:
        try:
            with Image.open(source, formats=("JPEG", "PNG", "TIFF")) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("image must contain exactly one page")
                size = image.size
        except UnidentifiedImageError as exc:
            raise ValueError("expected a JPEG, PNG, or TIFF image") from exc
        except Image.DecompressionBombError as exc:
            raise ValueError(str(exc)) from exc
        source.seek(0)
        try:
            result = subprocess.run(
                [
                    "tesseract",
                    "stdin",
                    "stdout",
                    "-l",
                    language,
                    "--oem",
                    "1",
                    "--psm",
                    "6",
                    "tsv",
                ],
                stdin=source,
                capture_output=True,
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "install Tesseract for --image, or replay saved OCR with --tsv"
            ) from exc
    # Tesseract can exit successfully after silently falling back to one language.
    if result.returncode or "Failed loading language" in result.stderr:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise ValueError(f"Tesseract failed: {detail}")
    words, width, height = read_tsv(StringIO(result.stdout))
    if (width, height) != size:
        raise ValueError("Tesseract page dimensions differ from the source image")
    return words, width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single-page JPEG, PNG, or TIFF")
    source.add_argument(
        "--tsv", type=Path, help="Replay one unscaled page of Tesseract TSV"
    )
    parser.add_argument(
        "--report-id", required=True, help="Report identifier for scoring"
    )
    parser.add_argument(
        "--page", type=int, default=1, help="One-based page in the report (default 1)"
    )
    parser.add_argument(
        "--language", help="Tesseract language codes for --image (default spa+eng)"
    )
    args = parser.parse_args()
    if not args.report_id.strip() or args.page < 1:
        parser.error("--report-id must be nonempty and --page must be positive")
    if args.language is not None:
        if args.tsv is not None:
            parser.error("--language applies only to --image")
        if not all(code.strip() for code in args.language.split("+")):
            parser.error("--language requires a nonempty Tesseract language code")
    try:
        if args.tsv is not None:
            with args.tsv.open(encoding="utf-8-sig", newline="") as stream:
                words, width, height = read_tsv(stream)
        else:
            words, width, height = image_words(args.image, args.language or "spa+eng")
        predictions, review = predict_page(words, width, height, page=args.page)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    json.dump(
        {
            "report_id": args.report_id,
            "space": "image-pixels",
            "page": args.page,
            "width": width,
            "height": height,
            "predictions": predictions,
            "review": review,
        },
        sys.stdout,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    print()


if __name__ == "__main__":
    main()
