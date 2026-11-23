"""Check spatial supervision against the actual PDF glyphs and the saved JPEGs."""
from contextlib import closing
from hashlib import sha256
import json

from PIL import Image
import pypdfium2 as pdfium
import pytest

import generate_complete_synthetic as generator
from scripts.lib.report_composer import ReportContent, ReportTest
from scripts.lib.template_engine import _cols_3, _cols_4, _cols_5, _cols_6


def read_json(path):
    return json.loads(path.read_bytes())


FIELD_NAMES = ("test_name", "value", "unit", "reference_range", "flag", "method")


@pytest.mark.parametrize("columns", [_cols_3, _cols_4, _cols_5, _cols_6])
def test_wrapped_rows_follow_painted_glyphs_across_pages(tmp_path, columns):
    rows = [
        ReportTest(f"TEST-{i}", f"Ácido α y β con descripción extensa de ensayo y marcador X{i:03d}",
                   f"{i:03d},52", "µg/dL", "0,20 – 0,80", "H" if i % 2 else None, "Panel SIM", True)
        for i in range(38)
    ]
    report = ReportContent("RPT_TEST", "SIM-CASO", 40, "F", "SIM-ID", "01/01/2026", "Centro SIM",
                           "Dirección SIM", "SIM-CLIN", [("Panel SIM", rows[:21]), ("Panel B", rows[21:])],
                           "normal_adult", "spain", {"results": []})
    config = {"columns": columns(), "font_config": {"family": "Times-Roman", "body_size": 10},
              "row_height_factor": 1.35}
    labels = generator.visible_ground_truth(report, config)["results"]
    pages = []
    pdf = tmp_path / "annotated.pdf"
    n = generator.render_simple_pdf(report, config, pdf, coordinate_pages=pages)
    assert n == len(pages) >= 2
    indices = []
    with closing(pdfium.PdfDocument(pdf)) as doc:
        for page_number, geometry in enumerate(pages, 1):
            assert geometry["page"] == page_number
            with closing(doc[page_number - 1]) as page, closing(page.get_textpage()) as text:
                assert page.get_size() == pytest.approx((geometry["width"], geometry["height"]), abs=0.001)
                glyphs = []
                for i in range(text.count_chars()):
                    char = chr(pdfium.raw.FPDFText_GetUnicode(text, i))
                    if not char.isspace():
                        x0, y0, x1, y1 = text.get_charbox(i)
                        glyphs.append(((x0 + x1) / 2, geometry["height"] - (y0 + y1) / 2, char))
            previous_bottom = 0
            for row in geometry["rows"]:
                idx = row["result_index"]
                indices.append(idx)
                assert row["loinc"] == labels[idx]["loinc"]
                left, top, right, bottom = row["bbox"]
                assert previous_bottom <= top + 1e-6 < bottom <= geometry["height"]
                assert list(row["cells"]) == list(FIELD_NAMES[:len(columns())])
                next_left = left
                for field, (x0, y0, x1, y1) in row["cells"].items():
                    assert (x0, y0, y1) == pytest.approx((next_left, top, bottom))
                    assert x0 < x1
                    seen = "".join(c for x, y, c in glyphs if x0 <= x < x1 and y0 <= y < y1)
                    assert seen == "".join((labels[idx].get(field) or "").split()), (idx, field, seen)
                    next_left = x1
                assert next_left == pytest.approx(right)
                previous_bottom = bottom
    assert indices == list(range(len(labels)))


@pytest.mark.parametrize("fmt", ["pdf", "both", "jpg"])
def test_cli_coordinates_are_opt_in_and_preserve_existing_artifacts(tmp_path, fmt):
    off, on = tmp_path / "off", tmp_path / "on"
    args = ["--format", fmt, "--count", "2", "--workers", "1", "--dpi", "137"]
    if fmt != "pdf":
        args += ["--tiers", "1,0,0,0,0"]
    assert generator.main([*args, "--output", str(off)]) == 0
    assert generator.main([*args, "--coordinates", "--workers", "2", "--output", str(on)]) == 0
    assert not (off / "coordinates").exists()
    assert "coordinates" not in read_json(off / "metadata.json")
    assert read_json(on / "metadata.json")["coordinates"] is True
    old = [json.loads(s) for s in (off / "reports.jsonl").read_text().splitlines()]
    new = [json.loads(s) for s in (on / "reports.jsonl").read_text().splitlines()]
    for a, b in zip(old, new, strict=True):
        path = b.pop("coordinates")
        assert path == f"coordinates/{a['report_id']}.json"
        assert b["sha256"].pop(path) == sha256((on / path).read_bytes()).hexdigest()
        assert a == b
        sidecar = read_json(on / path)
        assert set(sidecar) == {"schema_version", "report_id", "coordinate_space", "ground_truth_sha256", "pages"}
        assert sidecar["schema_version"] == 1 and sidecar["coordinate_space"] == "pdf_points_top_left"
        assert sidecar["report_id"] == a["report_id"]
        assert sidecar["ground_truth_sha256"] == sha256((on / a["ground_truth"]).read_bytes()).hexdigest()
        assert len(sidecar["pages"]) == a["pages"]
        for i, geometry in enumerate(sidecar["pages"]):
            if fmt == "pdf":
                assert "image" not in geometry
                continue
            info = geometry["image"]
            assert info["path"] == a["images"][i]
            with Image.open(on / info["path"]) as img:
                assert (info["width"], info["height"]) == img.size
            if fmt == "both":
                with closing(pdfium.PdfDocument(on / a["pdf"])) as doc, closing(doc[i]) as page, closing(
                    page.render(scale=137 / 72)
                ) as bitmap:
                    convert = bitmap.get_posconv(page)
                    for row in geometry["rows"]:
                        x, y = row["bbox"][:2]
                        fx, fy = x * info["width"] / geometry["width"], y * info["height"] / geometry["height"]
                        assert convert.to_bitmap(x, geometry["height"] - y) == pytest.approx((fx, fy), abs=0.51)
    for p in off.rglob("*"):
        if p.is_file() and p.name not in ("metadata.json", "coverage_report.json", "reports.jsonl"):
            assert p.read_bytes() == (on / p.relative_to(off)).read_bytes()


@pytest.mark.parametrize("fmt", ["jpg", "both"])
def test_coordinates_reject_image_tiers_that_move_content(tmp_path, fmt):
    dest = tmp_path / "unused"
    with pytest.raises(SystemExit):
        generator.main(["--coordinates", "--format", fmt, "--count", "1", "--tiers",
                        "0.999999,0,0,0.000001,0", "--output", str(dest)])
    assert not dest.exists()


def test_long_assay_code_keeps_hanging_punctuation_in_its_cell(tmp_path):
    from reportlab.lib.pagesizes import A4

    # Nine Courier letters use 54 pt. ReportLab's default CJK word splitter
    # also allows the following colon (6 pt) to hang into the next column.
    name = "AAAAAAAAA:BBBBBBBBB"
    assay = ReportTest("TEST-1", name, "98", "pg", "", None, "Panel SIM", True)
    report = ReportContent("RPT_T", "SIM-CASO", 30, "F", "SIM-ID", "01/01/2026", "Centro SIM", "",
                           "SIM-CLIN", [("Panel SIM", [assay])], "normal_adult", "spain", {"results": []})
    columns = _cols_3()
    columns[0]["width_pct"] = (54.25 + 4) / (A4[0] - 60 - 50)
    layout = {"columns": columns, "font_config": {"family": "Courier", "body_size": 10}}
    pages = []
    path = tmp_path / "test.pdf"
    generator.render_simple_pdf(report, layout, path, coordinate_pages=pages)
    geometry = pages[0]
    cell = geometry["rows"][0]["cells"]["test_name"]
    with closing(pdfium.PdfDocument(path)) as doc, closing(doc[0]) as page, closing(page.get_textpage()) as text:
        content = text.get_text_bounded(cell[0], geometry["height"] - cell[3], cell[2], geometry["height"] - cell[1])
    assert "".join(content.split()) == name
    from reportlab.pdfbase.pdfmetrics import stringWidth
    assert all(stringWidth(part, "Courier", 10) <= 54.25 for part in generator._wrap_cell(name, 54.25, "Courier", 10))
