# MedExtract Lab Generator — synthetic lab reports for OCR

MedExtract Lab Generator creates synthetic laboratory reports in Python for evaluating OCR and table extraction. The default reports use concise Spanish test names and fictional patient and clinician identifiers. The bundled vocabulary covers 32 quantitative LOINC codes.

Generate a vector PDF and one JPEG per page, then compare an extractor’s output with two sets of JSON labels. The printed labels preserve decimal commas, units and display names for optical character recognition (OCR). Use the normalized labels to compare numeric values and score test-name mapping to LOINC. Optional row and cell bounding boxes let you score whether each extracted value belongs to the correct test.

## Download 1,000 annotated lab reports

Starter v0.2.0 contains 1,000 synthetic reports, 1,015 clean page images and 18,329 observations with row and cell coordinates. The pages use concise Spanish test names and accented headings. Each report stays in its original training, validation or test split. Stream a split from Hugging Face or download the PDFs, images and labels as Parquet files. [MedExtract Synthetic Lab Reports on Hugging Face](https://huggingface.co/datasets/AdrianPerez3/medextract-synthetic-lab-reports-starter)

For comparisons with earlier OCR results, v0.1.0 contains the same observations with catalog-style names across 1,417 pages. Read each version’s images and coordinates together. [Original starter revision](https://huggingface.co/datasets/AdrianPerez3/medextract-synthetic-lab-reports-starter/tree/2880762ee47ef0e9fa26ae9c27a914c6f7cfd0c9)

In the starter’s worked example, glucose reads `163,6` and creatinine reads `1,49`. Swapping the value locations along with their text leaves both numbers transcribed correctly. The scorer then credits zero correct test-to-value associations. [Run the row-association scoring example](https://huggingface.co/datasets/AdrianPerez3/medextract-synthetic-lab-reports-starter/blob/main/examples/README.md)

## Inspect a sample report and its ground truth

**Sample report** · [PDF](examples/report.pdf) · [JPEG](examples/report.jpg) · [Printed rows with LOINC annotations](examples/visible.json) · [Normalized rows](examples/normalized.json)

In the first row of the sample, the printed name is `Leucocitos` and the result is `6,79` in `10*3/uL`. The normalized JSON stores the string `"6.79"`. Both targets identify the observation as LOINC `6690-2`, so you can score the transcription and the code assignment separately.

Para evaluar el OCR de informes de laboratorio en español, compara la coma decimal y la unidad con el texto impreso. Usa las coordenadas de las celdas para comprobar que cada resultado corresponde a la prueba correcta.

## Generate your first reports

Run these commands from the repository root with Python 3.11. The generator creates `demo-run` if needed and accepts an existing empty directory.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python generate_complete_synthetic.py --count 16 --seed 42 --workers 2 --coordinates --tiers 1,0,0,0,0 --format both --dpi 144 --output demo-run
python scripts/validate_synthetic_dataset.py --dataset demo-run --expected-count 16 --verify-source
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`. Installing dependencies requires a package index or local wheels. Generation runs offline.

The checked run produced **16 reports, 16 page images and 291 result rows covering all 32 demo codes**. Open `demo-run/reports.jsonl` to find each report’s media, labels and coordinate file. The sample above is `RPT_000000` from this command.

The command uses clean images (tier 0) at 144 DPI. Row and cell bounds are recorded in PDF points from the top left. Image coordinates are supported for clean pages. The usage guide shows how to convert points to pixels and how to generate separate OCR stress cases with simulated blur or perspective. [Capture conditions and coordinate conventions](docs/usage.md)

## Where to look

| If you want to… | Start here |
|---|---|
| Measure what a simulated photocopy changes on the same 40 lab reports | [Paired OCR study, generator and score replay](docs/benchmarks/paired-photocopy.md) |
| Compare PaddleOCR, Tesseract and decimal-mark review on 80 synthetic reports | [OCR punctuation study, data and scoring instructions](docs/benchmarks/ocr-punctuation.md) |
| Check whether an extractor attached each value to the correct test | [Generate row and cell coordinates](docs/usage.md#check-which-test-a-value-belongs-to) |
| Plot code coverage, change the output format, simulate a camera capture or split training data | [Usage guide](docs/usage.md) |
| Trace how rows become pages and labels | [Generator entry point](generate_complete_synthetic.py) and [implementation and tests](scripts/lib/) |
| Inspect the bundled numeric inputs | [Demo data](src/paddle_ocr/dictionaries/data/) |
| Check which terminology is bundled or build a custom vocabulary | [Terminology and provenance](TERMINOLOGY.md) |
| Understand the separate 70,000-report evaluation | [Benchmark results and limitations](docs/benchmark.md) |
| Report a problem, run tests or rebuild the source archive | [Contributor guide](CONTRIBUTING.md) |

The demo's numeric values are simulated. Reference intervals and high/low flags are omitted by default. Its purpose is document-extraction testing.

## Licenses

The Python source uses the [MIT license](LICENSE). LOINC 2.82 content has [separate license terms](licenses/LoincLicense_5.8.txt) and [translation credits](TERMINOLOGY.md). Verify the files inside the extracted source package with `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux.

This material contains content from LOINC (http://loinc.org). LOINC is copyright © Regenstrief Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the license at http://loinc.org/license. LOINC® is a registered United States trademark of Regenstrief Institute, Inc.
