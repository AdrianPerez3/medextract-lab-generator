# Working with a generated dataset

Read `reports.jsonl` first. Each record identifies a report, its two label files, the requested media and their SHA-256 checksums. Image paths are in page order. Resolve manifest paths relative to the dataset directory. A training example can span several pages.

The report headings use simulation tokens for people. The patient label begins with `SIM-CASO-`, its identifier with `SIM-ID-`, and the clinician label with `SIM-CLIN-`. The manifest's `report_id` identifies the generated document.

| Output | How to use it |
|---|---|
| `pdfs/` and `images/` | Read the vector report or test OCR on its page JPEGs. |
| `ground_truth/` | Compare extracted rows with the display spellings, decimal punctuation and units. The LOINC field is an annotation for the observation. |
| `normalized_ground_truth/` | Compare the underlying structured rows, where numeric values are strings with a decimal point. A layout can reorder the printed rows, so avoid matching the two files solely by array position. |
| `metadata.json` and `coverage_report.json` | Inspect run settings, source and input fingerprints, software versions and observed code counts. |

## Choose the capture conditions

The quickstart uses clean pages at 144 DPI. Omit `--tiers` for mixed simulated capture conditions, including blur, perspective and occlusion. The five comma-separated weights must be nonnegative and total one. The generator selects one tier per report and writes one JPEG per page in that condition. The labels describe the typeset document. A word visible in the PDF can be unreadable in a degraded JPEG.

Use `--format pdf` for vector output, `--format jpg` for page images, or `--format both` for both. JPEG output defaults to 300 DPI and accepts integer resolutions from 72 to 600. Choose a new or empty output directory for each run. List all options with `python generate_complete_synthetic.py --help`.

## Inspect results and flags

The ordinary command leaves reference intervals empty and flags null. For a fixture that exercises those fields, pass `--simulated-reference-ranges`. The page footer then identifies displayed intervals and flags as simulated. Values, intervals and patient profiles are software test fixtures. They have no clinical validation.

## Split reports for training and evaluation

After generating the quickstart dataset, run the splitter from the project root.

```sh
python scripts/package_training_data.py --dataset demo-run --split 80,10,10 --seed 42
```

The default JPEG-enabled run writes three unsplit files. SFT contains the target for supervised fine-tuning, DPO pairs each target with an intentionally corrupted answer, and GRPO pairs a prompt with a solution for scoring. PDF-only runs omit these image-based streams.

Use the files in `demo-run/splits/` for training and held-out evaluation. The top-level files contain every report, including those assigned to validation and test. Their image paths resolve from the dataset root. In the default split folder, image paths in split JSONL files resolve from that folder. If a custom output directory is on another Windows volume, the receipt's `image_path_base` identifies the original dataset as the base. The splitter keeps each report together across formats and groups reports that share an exact target or identical JPEG page bytes. On a small dataset the realized fractions may differ from the requested percentages.

`report_splits.json` lists report IDs by split. In `packaging_report.json`, each format's `source` is a filename relative to the dataset root, and each split's `file` is relative to the directory containing the receipt. The default receipt records the dataset as `..` and the output directory as `.`.
