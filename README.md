# MedExtract Lab Generator

Generate fictional laboratory reports with Spanish headings and Spanish or English test names. The bundled demo covers 32 quantitative LOINC codes and produces a vector PDF, one JPEG per page and two sets of JSON labels. Use them to test OCR, table extraction and terminology matching without patient records.

**Sample report** · [PDF](examples/report.pdf) · [JPEG](examples/report.jpg) · [Printed rows with LOINC annotations](examples/visible.json) · [Normalized rows](examples/normalized.json)

In the first row of the sample, the printed white blood cell count is `6,79` in `10*3/uL`. The normalized JSON stores the string `"6.79"`. Both targets identify the observation as LOINC `6690-2`, so you can score the transcription and the code assignment separately.

## Generate your first reports

Run these commands from the repository root with Python 3.11. The generator creates `demo-run` if needed and accepts an existing empty directory.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python generate_complete_synthetic.py --count 16 --seed 42 --workers 2 --tiers 1,0,0,0,0 --format both --dpi 144 --output demo-run
python scripts/validate_synthetic_dataset.py --dataset demo-run --expected-count 16 --verify-source
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`. Installing dependencies requires a package index or local wheels. Generation runs offline.

The checked run produced **16 reports, 20 page images and 291 result rows covering all 32 demo codes**. Open `demo-run/reports.jsonl` to find the files for each report. The sample above is `RPT_000000` from this command.

## Where to look

| If you want to… | Start here |
|---|---|
| Change the output format, simulate a camera capture or split training data | [Usage guide](docs/usage.md) |
| Trace how rows become pages and labels | [Generator entry point](generate_complete_synthetic.py) and [implementation and tests](scripts/lib/) |
| Inspect the bundled numeric inputs | [Demo data](src/paddle_ocr/dictionaries/data/) |
| Check which terminology is bundled or build a custom vocabulary | [Terminology and provenance](TERMINOLOGY.md) |
| Understand the separate 70,000-report evaluation | [Benchmark results and limitations](docs/benchmark.md) |
| Report a problem, run tests or rebuild the source archive | [Contributor guide](CONTRIBUTING.md) |

The demo's numeric values are simulated. Reference intervals and high/low flags are omitted by default. Its purpose is document-extraction testing.

## Licenses

The Python source uses the [MIT license](LICENSE). LOINC 2.82 content has [separate license terms](licenses/LoincLicense_5.8.txt) and [translation credits](TERMINOLOGY.md). Verify the files inside the extracted source package with `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux.

This material contains content from LOINC (http://loinc.org). LOINC is copyright © Regenstrief Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the license at http://loinc.org/license. LOINC® is a registered United States trademark of Regenstrief Institute, Inc.
