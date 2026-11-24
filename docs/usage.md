# Working with a generated dataset

Read `reports.jsonl` first. Each record identifies a report, its two label files, the requested media and their SHA-256 checksums. Image paths are in page order. Resolve manifest paths relative to the dataset directory. A training example can span several pages.

The report headings use simulation tokens for people. The patient label begins with `SIM-CASO-`, its identifier with `SIM-ID-`, and the clinician label with `SIM-CLIN-`. The manifest's `report_id` identifies the generated document.

| Output | How to use it |
|---|---|
| `pdfs/` and `images/` | Read the vector report or test OCR on its page JPEGs. |
| `ground_truth/` | Compare extracted rows with the display spellings, decimal punctuation and units. The LOINC field is an annotation for the observation. |
| `normalized_ground_truth/` | Compare the underlying structured rows, where numeric values are strings with a decimal point and `test_name` carries the selected catalog term. A layout can reorder the printed rows, so avoid matching the two files solely by array position. |
| `metadata.json` and `coverage_report.json` | Inspect run settings, source and input fingerprints, software versions and observed code counts. |

## Choose the printed test names

The default `--name-style routine-es` uses a short Spanish label for each of the 32 demo codes. For example, the report distinguishes `Hematíes`, `Hemoglobina corpuscular media (HCM)` and `Hemoglobina`. Use `ground_truth/` to score the literal text on the page, and join to `normalized_ground_truth/` by report ID and LOINC to inspect the corresponding catalog term.

Use `--name-style catalog` to exercise terminology extraction with the earlier mix of code-bound Spanish and English names. This is the naming style used in starter v0.1.0. In a build with a larger pinned LOINC catalog, codes outside the 32-name lookup retain their catalog labels. A shorter printed name can change wrapping, pagination and cell coordinates. Always take the sidecar and image from the same run.

## Choose the capture conditions

The quickstart uses clean pages at 144 DPI. Omit `--tiers` for mixed simulated capture conditions, including blur, perspective and occlusion. The five comma-separated weights must be nonnegative and total one. The generator selects one tier per report and writes one JPEG per page in that condition. The labels describe the typeset document. A word visible in the PDF can be unreadable in a degraded JPEG.

Use `--format pdf` for vector output, `--format jpg` for page images, or `--format both` for both. JPEG output defaults to 300 DPI and accepts integer resolutions from 72 to 600. Choose a new or empty output directory for each run. List all options with `python generate_complete_synthetic.py --help`.

The generator automatically uses up to four render processes. For page images above 300 DPI the default is one process. Higher resolutions increase the number of pixels on every page. Set `--workers 1` to minimize concurrent memory use, or choose an explicit worker count after measuring a representative run. A fixed seed produces the same ordered report artifacts across worker counts on the same software and platform.

## Plot dataset coverage

From the project root, run this after generating a schema-v2 dataset. Choose a new filename for the chart.

```sh
python scripts/analyze_synthetic_dataset.py --dataset demo-run --rare-at-most 5 --html coverage.html
```

Open `coverage.html` in a browser. The tool also writes the analysis as JSON to stdout. It reads one visible label file per report, verifies its recorded checksum, and counts the rows assigned to each eligible LOINC. The chart shows missing and rare codes, how many result rows the ten most common codes account for, and how reports vary in pages, row counts, columns and capture tier. Page counts and tiers come from the manifest. Use the validator to check media integrity and source fingerprints.

## Check which test a value belongs to

Generate the 16-report quickstart with coordinates and validate the resulting files.

```sh
python generate_complete_synthetic.py --coordinates --count 16 --seed 42 --workers 2 --format both --tiers 1,0,0,0,0 --dpi 144 --output spatial-run
python scripts/validate_synthetic_dataset.py --dataset spatial-run --expected-count 16 --verify-source
```

The manifest's `coordinates` entry points to a JSON file for that report. Each page has its own width, height and rows. Each row's zero-based `result_index` selects the corresponding entry in the **visible** label's `results` array. Its `loinc` identifies the test. The sidecar records the SHA-256 of that visible label, so predictions can be scored against the labels used for that rendering.

Rows and cells use rectangles `[left, top, right, bottom]` in PDF points, measured from the top-left corner of the page. Pages start at 1. Each field has a rectangle covering its allocated table cell, including whitespace and wrapped lines. Blank visible columns also have rectangles. When extracting from a clean JPEG, use the actual pixel dimensions in the page's `image` entry. To convert a PDF point to pixels, multiply x by `image.width / page.width` and y by `image.height / page.height`. The two ratios account for raster dimension rounding.

For vector-only extraction, use `--format pdf --coordinates`. Image coordinates currently support clean tier 0. The CLI requires `--tiers 1,0,0,0,0` for image runs with coordinates because other tiers can rotate or warp the printed cells.

To score an extraction, supply one point inside the printed test name and one inside the value paired with it. Use the center of each predicted word or field box. For example, save these two predictions from the first quickstart PDF as `predictions.json`.

```json
{
  "report_id": "RPT_000000",
  "predictions": [
    {"page": 1, "test_point": [55, 147], "value_point": [294, 147], "value_text": "6,79"},
    {"page": 1, "test_point": [53, 159], "value_point": [294, 159], "value_text": "4,87"}
  ]
}
```

```sh
python scripts/score_row_associations.py --coordinates spatial-run/coordinates/RPT_000000.json --ground-truth spatial-run/ground_truth/RPT_000000.json --predictions predictions.json
```

Both predictions pair the correct cells and match the printed values. The scorer credits two of 17 expected rows. If you exchange the two value points together with their text, both values still transcribe their selected cells correctly, but the scorer reports two wrong-row links and zero correct associations. For original clean JPEG pixel positions, add `--space image-pixels`.

`association` measures correct test-to-value links. `association_and_exact_value` also requires a character-for-character match to the printed value. Precision divides distinct matched rows by all submitted predictions, and recall divides them by all expected rows. Duplicate submissions can earn a row credit only once. The score evaluates the supplied locations and value text. It does not run OCR or check the extracted test-name spelling.

## Inspect results and flags

The ordinary command leaves reference intervals empty and flags null. For a fixture that exercises those fields, pass `--simulated-reference-ranges`. The page footer then identifies displayed intervals and flags as simulated. Values, intervals and patient profiles are software test fixtures. They have no clinical validation.

For custom input packs, choose numeric bounds that contain at least one value printable at the report's precision. If an ordinary sample would round outside the supplied bounds, the generator uses a printable value inside them when possible. If no value fits, generation stops with the affected LOINC code and bounds so you can correct the input.

## Split reports for training and evaluation

After generating the quickstart dataset, run the splitter from the project root.

```sh
python scripts/package_training_data.py --dataset demo-run --split 80,10,10 --seed 42
```

The default JPEG-enabled run writes three unsplit files. SFT contains the target for supervised fine-tuning, DPO pairs each target with an intentionally corrupted answer, and GRPO pairs a prompt with a solution for scoring. PDF-only runs omit these image-based streams.

Before publishing schema-v2 splits, the packager checks each conversation's role order and requires one `<image>` marker in the user message per referenced page. A multi-page report has one training example containing all its page images in order.

Use the files in `demo-run/splits/` for training and held-out evaluation. The top-level files contain every report, including those assigned to validation and test. Their image paths resolve from the dataset root. In the default split folder, image paths in split JSONL files resolve from that folder. If a custom output directory is on another Windows volume, the receipt's `image_path_base` identifies the original dataset as the base. The splitter keeps each report together across formats and groups reports that share an exact target or identical JPEG page bytes. On a small dataset the realized fractions may differ from the requested percentages.

`report_splits.json` lists report IDs by split. In `packaging_report.json`, each format's `source` is a filename relative to the dataset root, and each split's `file` is relative to the directory containing the receipt. The default receipt records the dataset as `..` and the output directory as `.`.
