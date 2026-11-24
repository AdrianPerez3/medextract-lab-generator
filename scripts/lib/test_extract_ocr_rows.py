"""Exercise raw Tesseract TSV and the one-page image adapter."""

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest


CLI = Path(__file__).resolve().parents[1] / "extract_ocr_rows.py"
HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
PAGE = "1\t1\t0\t0\t0\t0\t0\t0\t600\t800\t-1\t\n"


def word(x, y, width, text):
    return f"5\t1\t1\t1\t1\t1\t{x}\t{y}\t{width}\t10\t95\t{text}\n"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), "--report-id", "demo", *map(str, args)],
        capture_output=True,
        text=True,
    )


def test_replay_outputs_pixel_anchors_and_scoring_envelope(tmp_path):
    path = tmp_path / "saved words.tsv"
    path.write_text(
        HEADER
        + PAGE
        + word(20, 90, 50, "Prueba")
        + word(190, 90, 65, "Resultado")
        + word(325, 90, 40, "Unidad")
        + word(20, 130, 55, "Ejemplo")
        + word(254, 130, 11, "<")
        + word(270, 130, 30, "0,1")
        + word(310, 130, 42, "mg/dL"),
        encoding="utf-8",
    )
    result = run_cli("--tsv", path, "--page", 3)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert (output["report_id"], output["space"], output["page"]) == (
        "demo",
        "image-pixels",
        3,
    )
    assert (output["width"], output["height"]) == (600, 800)
    assert output["review"] == []
    (prediction,) = output["predictions"]
    assert (prediction["page"], prediction["value_text"]) == (3, "<0,1")
    tx, ty = prediction["test_point"]
    vx, vy = prediction["value_point"]
    assert 20 <= tx < 75 and 130 <= ty < 140
    assert 254 <= vx < 300 and 130 <= vy < 140


@pytest.mark.parametrize(
    "content,message",
    [
        ("value,text\n", "TSV header"),
        (HEADER, "page record"),
        (HEADER + PAGE + PAGE, "one page"),
        (HEADER + PAGE + word(599, 130, 4, "7"), "word box"),
        (HEADER + PAGE + word(20, 130, 0, "0"), "word box"),
        (HEADER + PAGE + word(20, 130, 9, "7").replace("5\t1", "5\t2", 1), "page"),
        (HEADER + PAGE + "broken\n", "columns"),
    ],
)
def test_invalid_tsv_is_a_cli_error_without_predictions(tmp_path, content, message):
    path = tmp_path / "bad.tsv"
    path.write_text(content, encoding="utf-8")
    result = run_cli("--tsv", path)
    assert result.returncode == 2
    assert message in result.stderr
    assert result.stdout == ""


def test_blank_page_replay_has_no_predictions(tmp_path):
    path = tmp_path / "blank.tsv"
    path.write_bytes((HEADER + PAGE).replace("\n", "\r\n").encode("utf-8-sig"))
    result = run_cli("--tsv", path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["predictions"] == []


def test_language_only_applies_to_image(tmp_path):
    result = run_cli("--tsv", tmp_path / "unused.tsv", "--language", "eng")
    assert result.returncode == 2
    assert "--language" in result.stderr


def test_image_rejects_multiple_frames(tmp_path):
    path = tmp_path / "two.tiff"
    frame = Image.new("L", (12, 12), 255)
    frame.save(path, save_all=True, append_images=[frame])
    result = run_cli("--image", path)
    assert result.returncode == 2
    assert "one page" in result.stderr
    assert result.stdout == ""


def test_image_adapter_preserves_source_pixels_and_model_options(tmp_path, monkeypatch):
    from scripts.extract_ocr_rows import image_words

    path = tmp_path / "source.png"
    Image.new("L", (600, 800), 255).save(path)

    def tesseract(command, *, stdin, **options):
        assert command == [
            "tesseract",
            "stdin",
            "stdout",
            "-l",
            "spa+eng",
            "--oem",
            "1",
            "--psm",
            "6",
            "tsv",
        ]
        assert stdin.read() == path.read_bytes()
        return subprocess.CompletedProcess(
            command, 0, HEADER + PAGE + word(20, 30, 10, '"'), ""
        )

    monkeypatch.setattr(subprocess, "run", tesseract)
    assert image_words(path, "spa+eng") == ([('"', [20, 30, 30, 40])], 600, 800)


@pytest.mark.parametrize(
    "code, stderr, message",
    [
        (1, "engine error", "engine error"),
        (0, "Failed loading language 'spa'", "Failed loading language"),
        (0, "", "dimensions differ"),
    ],
)
def test_image_errors_even_if_tesseract_exits_successfully(
    tmp_path, monkeypatch, code, stderr, message
):
    from scripts.extract_ocr_rows import image_words

    path = tmp_path / "input.jpg"
    Image.new("RGB", (60, 80), "white").save(path)
    result = subprocess.CompletedProcess([], code, HEADER + PAGE, stderr)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(ValueError, match=message):
        image_words(path, "spa+eng")


def test_missing_pillow_has_replay_guidance_and_does_not_block_tsv(tmp_path):
    # Keep the script directory importable, while omitting site packages.
    cmd = [sys.executable, "-S", str(CLI), "--report-id", "demo"]
    env = {"PYTHONPATH": "", "PYTHONUTF8": "1"}
    failed = subprocess.run(
        [*cmd, "--image", "unused.jpg"], env=env, capture_output=True, text=True
    )
    assert failed.returncode == 2 and failed.stdout == ""
    assert "Pillow" in failed.stderr and "--tsv" in failed.stderr
    assert "Traceback" not in failed.stderr
    path = tmp_path / "page.tsv"
    path.write_text(HEADER + PAGE, encoding="utf-8")
    replay = subprocess.run(
        [*cmd, "--tsv", str(path)], env=env, capture_output=True, text=True
    )
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout)["predictions"] == []
