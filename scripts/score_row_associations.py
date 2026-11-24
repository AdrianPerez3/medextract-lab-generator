"""Score predicted test/value point pairs against one report's visible table cells."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys


def _object(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _array(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _integer(value, field, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _numbers(value, field, size):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must contain {size} coordinates")
    if not all(type(x) in (int, float) and isfinite(x) for x in value):
        raise ValueError(f"{field} coordinates must be finite numbers")
    return value


def _box(value, field, outer):
    left, top, right, bottom = _numbers(value, field, 4)
    if not (
        outer[0] <= left < right <= outer[2] and outer[1] <= top < bottom <= outer[3]
    ):
        raise ValueError(f"{field} must be a nonempty box inside its enclosing bounds")
    return left, top, right, bottom


def _contains(box, point):
    x, y = point
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _prepare_pages(sidecar, data, space):
    """Bind visible bytes and check the one-to-one mapping from cells to results."""
    if type(sidecar.get("schema_version")) is not int or sidecar["schema_version"] != 1:
        raise ValueError("Expected coordinate schema version 1")
    if sidecar.get("coordinate_space") != "pdf_points_top_left":
        raise ValueError("Expected pdf_points_top_left sidecar coordinates")
    report_id = sidecar.get("report_id")
    if not isinstance(report_id, str) or not report_id.strip():
        raise ValueError("Missing coordinate report_id")
    if sidecar.get("ground_truth_sha256") != sha256(data).hexdigest():
        raise ValueError("Visible ground truth SHA-256 does not match the sidecar")
    results = _array(
        _object(json.loads(data), "ground truth").get("results"), "results"
    )
    if not results:
        raise ValueError("Expected at least one visible result")
    if space not in ("pdf-points", "image-pixels"):
        raise ValueError("Unknown prediction coordinate space")
    pages = []
    values = []
    for number, page in enumerate(_array(sidecar.get("pages"), "pages"), 1):
        _object(page, "page")
        if _integer(page.get("page"), "page", 1) != number:
            raise ValueError("Page numbers must be consecutive, starting at 1")
        w, h = _numbers([page.get("width"), page.get("height")], "page dimensions", 2)
        if min(w, h) <= 0:
            raise ValueError("Page dimensions must be positive")
        input_width, input_height = w, h
        if space == "image-pixels":
            image = _object(page.get("image"), "clean page image")
            if not isinstance(image.get("path"), str) or not image["path"].strip():
                raise ValueError("Missing clean page image path")
            input_width = _integer(image.get("width"), "image width", 1)
            input_height = _integer(image.get("height"), "image height", 1)
        cells = []
        previous_bottom = 0
        for row in _array(page.get("rows"), "rows"):
            _object(row, "row")
            index = _integer(row.get("result_index"), "result_index", 0)
            if index != len(values) or index >= len(results):
                raise ValueError(
                    "Expected consecutive visible result indexes in reading order"
                )
            result = _object(results[index], "visible result")
            code = result.get("loinc")
            if (
                not isinstance(code, str)
                or not code.strip()
                or row.get("loinc") != code
            ):
                raise ValueError(f"LOINC mismatch for result {index}")
            value = result.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing visible value text for result {index}")
            bounds = _box(row.get("bbox"), "row bbox", (0, 0, w, h))
            # Separately computed boundaries can differ by a few floating-point ULPs.
            if previous_bottom - bounds[1] > 1e-8:
                raise ValueError("Row boxes must be vertically ordered and disjoint")
            previous_bottom = bounds[3]
            fields = _object(row.get("cells"), "cells")
            name = _box(fields.get("test_name"), "test_name", bounds)
            val = _box(fields.get("value"), "value", bounds)
            if name[2] > val[0]:
                raise ValueError("Expected disjoint name and value columns in reading order")
            cells.append((index, name, val))
            values.append(value)
        pages.append((cells, w, h, input_width, input_height))
    if len(values) != len(results):
        raise ValueError("Sidecar does not cover every visible result")
    return pages, values


def _locate(point, page, column):
    cells, width, height, input_width, input_height = page
    if not _contains((0, 0, input_width, input_height), point):
        return None, False
    # PDFium renders the unrotated page into its full integer bitmap rectangle.
    # Width and height can round up independently. Nominal DPI loses that detail.
    if (width, height) != (input_width, input_height):
        point = point[0] / input_width * width, point[1] / input_height * height
    # At a shared edge within floating-point roundoff, the lower row owns its top.
    for cell in reversed(cells):
        if _contains(cell[column], point):
            return cell[0], True
    return None, True


def locate_prediction(pages, item):
    """Return each anchor's result index and a geometry outcome for one prediction."""
    _object(item, "prediction")
    number = _integer(item.get("page"), "prediction page", 1)
    name = _numbers(item.get("test_point"), "test_point", 2)
    value = _numbers(item.get("value_point"), "value_point", 2)
    if not isinstance(item.get("value_text"), str):
        raise ValueError("value_text must be a string")
    if number > len(pages):
        return None, None, "unknown_page"
    test_row, name_in_page = _locate(name, pages[number - 1], 1)
    value_row, value_in_page = _locate(value, pages[number - 1], 2)
    if not (name_in_page and value_in_page):
        location = "outside_page"
    elif test_row is None or value_row is None:
        location = "outside_target_cell"
    elif test_row != value_row:
        location = "different_row"
    else:
        location = "same_row"
    return test_row, value_row, location


def _metrics(matched, total, submitted):
    count = len(matched)
    return {
        "matched_rows": count,
        "missing_rows": total - count,
        "precision": count / submitted if submitted else None,
        "recall": count / total,
    }


def _check_source_image(predictions, pages):
    """Check page metadata from a single-image extractor against the clean sidecar."""
    if not any(key in predictions for key in ("width", "height")):
        return None
    page = _integer(predictions.get("page"), "source image page", 1)
    if page > len(pages):
        raise ValueError("Source image page is missing from the coordinate sidecar")
    width = _integer(predictions.get("width"), "source image width", 1)
    height = _integer(predictions.get("height"), "source image height", 1)
    if (width, height) != pages[page - 1][-2:]:
        raise ValueError(
            "Source image dimensions differ from the clean coordinate sidecar"
        )
    return page


def score(coordinates, visible_bytes, predictions, space=None):
    """Award one credit per expected row. Every submitted pair counts for precision."""
    _object(predictions, "prediction document")
    declared_space = predictions.get("space")
    if space is None:
        space = predictions.get("space", "pdf-points")
    elif "space" in predictions and space != declared_space:
        raise ValueError(
            "Requested coordinate space conflicts with the prediction document"
        )
    pages, values = _prepare_pages(
        _object(coordinates, "coordinates"), visible_bytes, space
    )
    source_page = (
        _check_source_image(predictions, pages)
        if declared_space == "image-pixels"
        else None
    )
    report_id = coordinates["report_id"]
    if "report_id" in predictions and predictions["report_id"] != report_id:
        raise ValueError("Prediction report_id does not match coordinates")
    items = _array(predictions.get("predictions"), "predictions")
    geometry = dict.fromkeys(
        (
            "same_row",
            "different_row",
            "outside_target_cell",
            "outside_page",
            "unknown_page",
        ),
        0,
    )
    anchors = Counter()
    offsets = Counter()
    associated = set()
    exact = set()
    text_at_value = text_for_test = 0
    for item in items:
        test_row, value_row, location = locate_prediction(pages, item)
        if source_page is not None and item["page"] != source_page:
            raise ValueError(
                "Prediction page differs from its declared source image page"
            )
        text = item["value_text"]
        geometry[location] += 1
        if test_row is not None:
            anchors[test_row] += 1
            text_for_test += values[test_row] == text
        if value_row is not None:
            text_at_value += values[value_row] == text
        if location == "different_row":
            offsets[value_row - test_row] += 1
        elif location == "same_row":
            associated.add(test_row)
            if values[test_row] == text:
                exact.add(test_row)
    return {
        "report_id": report_id,
        "space": space,
        "expected_rows": len(values),
        "submitted_predictions": len(items),
        "association": _metrics(associated, len(values), len(items)),
        "association_and_exact_value": _metrics(exact, len(values), len(items)),
        "prediction_geometry": geometry,
        "wrong_row_offsets": {str(k): offsets[k] for k in sorted(offsets)},
        "duplicate_test_anchor_predictions": sum(anchors.values()) - len(anchors),
        "rows_without_test_anchor": len(values) - len(anchors),
        "exact_text_at_value_cell": text_at_value,
        "exact_text_for_test_row": text_for_test,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'Predictions JSON: {"predictions":[{"page":1,"test_point":[x,y],'
            '"value_point":[x,y],"value_text":"7,0"}]}. Points use a top-left origin. '
            "Use original clean image dimensions for pixels. Output is JSON."
        ),
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        required=True,
        help="Generator row-coordinate sidecar",
    )
    parser.add_argument(
        "--ground-truth", type=Path, required=True, help="Original visible results JSON"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predicted point pairs and visible values",
    )
    parser.add_argument(
        "--space",
        choices=("pdf-points", "image-pixels"),
        help="Override for unlabelled predictions (default: declared space or pdf-points)",
    )
    args = parser.parse_args()
    try:
        coords = json.loads(args.coordinates.read_bytes())
        predictions = json.loads(args.predictions.read_bytes())
        result = score(coords, args.ground_truth.read_bytes(), predictions, args.space)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
