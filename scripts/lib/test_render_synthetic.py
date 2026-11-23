"""Read the generated PDF and JPEGs to check the evidence supplied to training."""
import random

import cv2
import pypdfium2 as pdfium
import pytest
from reportlab.pdfgen.canvas import Canvas

import generate_complete_synthetic as generator
from scripts.lib.degradation_engine import DegradationEngine
from scripts.lib.report_composer import ReportContent, ReportTest
from scripts.lib.template_engine import _cols_3, _cols_4, _cols_5, _cols_6


def clinical_fixture():
    assay = ReportTest(
        "1234-5", "Determinación de α y β con anticuerpos γ y confirmación final centinela",
        "1.234,50", "µg/dL", "0,50 – 1,00", "H", "Panel Sintético", True,
    )
    return ReportContent(
        "RPT_TEST", "Paciente Sintético", 40, "F", "TEST-1", "01/01/2026",
        "Laboratorio Sintético", "Calle de ejemplo", "Doctora Ejemplo",
        [(assay.panel_name, [assay])], "normal_adult", "spain",
        {"results": [{"value": "1234.500", "reference_range": "0.5 - 1.0"}]},
    )


@pytest.mark.parametrize("columns", [_cols_3, _cols_4, _cols_5, _cols_6])
def test_labels_match_visible_fields(tmp_path, columns):
    report = clinical_fixture()
    config = {"columns": columns(), "show_sections": columns is not _cols_3}
    pdf = tmp_path / "report.pdf"
    generator.render_simple_pdf(report, config, pdf)
    doc = pdfium.PdfDocument(pdf)
    page = doc[0]
    textpage = page.get_textpage()
    try:
        text = textpage.get_text_bounded()
    finally:
        textpage.close()
        page.close()
        doc.close()
    assert "centinela" in text
    assert all(marker in text for marker in ("Determinación", "α", "β", "γ", "1.234,50", "µg/dL"))
    labels = generator.visible_ground_truth(report, config)["results"]
    row = labels[0]
    assert row["test_name"] == report.panels[0][1][0].display_name
    assert row["value"] == "1.234,50"
    assert row["unit"] == "µg/dL"
    assert row["loinc"] == "1234-5"
    assert row["reference_range"] == ("0,50 – 1,00" if len(columns()) >= 4 else "")
    assert ("0,50 – 1,00" in text) == bool(row["reference_range"])
    assert row["flag"] == ("H" if len(columns()) >= 5 else None)
    assert ("H" in text.split()) == bool(row["flag"])
    assert row["panel"] == ("Panel Sintético" if config["show_sections"] else "")
    assert ("Panel Sintético" in text) == bool(row["panel"])


def test_rendering_all_pages_in_order(tmp_path):
    pdf = tmp_path / "input.pdf"
    c = Canvas(str(pdf), pagesize=(144, 144))
    for x in (5, 55, 105):
        c.rect(x, 58, 30, 30, fill=1, stroke=0)
        c.showPage()
    c.save()
    images = tmp_path / "images"
    images.mkdir()
    out = generator._pdf_to_degraded_image(
        pdf, images, "RPT_TEST", 0, random.Random(42), DegradationEngine(),
    )
    assert out["image_path"] == "images/RPT_TEST.jpg"
    assert out["image_paths"] == [
        "images/RPT_TEST.jpg", "images/RPT_TEST_p002.jpg", "images/RPT_TEST_p003.jpg",
    ]
    for i, filename in enumerate(out["image_paths"]):
        image = cv2.imread(str(tmp_path / filename), cv2.IMREAD_GRAYSCALE)
        probes = [int(image[300, col]) for col in (80, 290, 500)]
        assert probes[i] < 10
        assert all(v > 240 for j, v in enumerate(probes) if j != i)


def test_jpeg_write_failure_is_reported(tmp_path):
    pdf = tmp_path / "one.pdf"
    c = Canvas(str(pdf), pagesize=(72, 72))
    c.drawString(4, 40, "visible")
    c.save()
    with pytest.raises(OSError):
        generator._pdf_to_degraded_image(
            pdf, tmp_path / "absent", "RPT_TEST", 0, random.Random(0), DegradationEngine(),
        )


def test_pdf_has_reproducible_bytes_and_creation_date(tmp_path, monkeypatch):
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    a, b = tmp_path / "one.pdf", tmp_path / "two.pdf"
    for path in (a, b):
        generator.render_simple_pdf(clinical_fixture(), {}, path)
    assert a.read_bytes() == b.read_bytes()
    assert b"D:20000101000000" in a.read_bytes()


def test_every_wrapped_row_and_panel_survives_page_breaks(tmp_path):
    from dataclasses import replace

    report = clinical_fixture()
    panel, (prototype,) = report.panels[0]
    tests = [
        replace(prototype, display_name=f"Determinación verificada en suero {idx:03d} FINAL{idx:03d}")
        for idx in range(120)
    ]
    tests[0] = replace(tests[0], display_name="Extremo" + "A" * 76 + "Z")
    report.panels = [(panel, tests)]
    config = {"columns": _cols_5(), "font_config": {"family": "Courier", "body_size": 10}}
    pdf = tmp_path / "long.pdf"
    count = generator.render_simple_pdf(report, config, pdf)
    assert count >= 3
    doc = pdfium.PdfDocument(pdf)
    try:
        all_text = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_bounded()
                assert panel in text
                # Read just the first column below the metadata. Hard-split words
                # must retain every glyph even when there is no wrapping space.
                left_column = textpage.get_text_bounded(left=59, right=224)
                all_text.append("".join(left_column.split()))
            finally:
                textpage.close()
                page.close()
    finally:
        doc.close()
    transcript = "".join(all_text)
    for test in tests:
        assert "".join(test.display_name.split()) in transcript


def test_unsupported_test_glyph_does_not_become_a_silent_square(tmp_path):
    report = clinical_fixture()
    report.panels[0][1][0].display_name = "Analito 🧬"
    with pytest.raises(ValueError, match="glyph"):
        generator.render_simple_pdf(report, {}, tmp_path / "unsupported.pdf")


def test_tight_six_column_headers_have_measurable_glyph_gaps(tmp_path):
    from contextlib import closing
    columns = _cols_6()
    # A 19-point Flag column occupied by four bold Courier glyphs at 10 pt
    # used to run into the following M in Metodo.
    columns[4]["width_pct"] = 0.04
    config = {"columns": columns, "font_config": {"family": "Courier", "body_size": 10}}
    path = tmp_path / "narrow_headers.pdf"
    generator.render_simple_pdf(clinical_fixture(), config, path)
    with closing(pdfium.PdfDocument(path)) as doc, closing(doc[0]) as page, closing(page.get_textpage()) as text:
        previous_right = None
        for column in columns:
            with closing(text.search(column["name"], match_case=True)) as search:
                match = search.get_next()
            assert match, column["name"]
            first, count = match
            boxes = [text.get_charbox(i) for i in range(first, first + count)]
            left = min(box[0] for box in boxes)
            if previous_right is not None:
                assert left - previous_right >= 3.75, (column["name"], left - previous_right)
            previous_right = max(box[2] for box in boxes)


def test_color_raster_with_padding_keeps_bgr_channels(tmp_path):
    # Odd pixel width exercises the native PDF bitmap's row stride.
    pdf = tmp_path / "colors.pdf"
    canvas = Canvas(str(pdf), pagesize=(73, 48))
    for x, rgb in ((0, (1, 0, 0)), (24, (0, 0, 1)), (48, (0, 1, 0))):
        canvas.setFillColorRGB(*rgb)
        canvas.rect(x, 0, 24, 48, fill=1, stroke=0)
    canvas.save()

    class IdentityEngine:
        def degrade(self, image, tier, rng):
            return image

    generator._pdf_to_degraded_image(pdf, tmp_path, "COLORS", 0, random.Random(42),
                                      IdentityEngine(), dpi=72)
    image = cv2.imread(str(tmp_path / "COLORS.jpg"))
    assert image.shape == (48, 73, 3)
    for x, channel in ((12, 2), (36, 0), (60, 1)):
        assert image[23, x, channel] > 245
        assert max(v for i, v in enumerate(image[23, x]) if i != channel) < 10
