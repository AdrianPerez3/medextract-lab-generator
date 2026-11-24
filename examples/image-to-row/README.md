# Extract and score rows from a lab report

This example follows the 17 printed rows of the repository's [synthetic report](../report.jpg). The OCR reads the words from the image. The row extractor pairs a point on a test name with a point on its result and returns the result text. The scorer uses the [visible labels](../visible.json) and [reference cell coordinates](coordinates.json) to check those pairs.

## Replay the saved OCR

Run these commands from the repository root. Replay uses the Python standard library and the included TSV file. It works offline without installing an OCR engine.

```sh
python scripts/extract_ocr_rows.py --tsv examples/image-to-row/report.tsv --report-id RPT_000000 > predictions.json
python scripts/score_row_associations.py --space image-pixels --coordinates examples/image-to-row/coordinates.json --ground-truth examples/visible.json --predictions predictions.json
```

All 17 predictions associate the test and value with the same row. Of those, 12 match the printed result exactly. For example, the saved OCR reads hemoglobin as `142` on the correct row, where the report prints `14,2`. The extractor preserves that reading so the scorer can expose the missing comma.

The JSON describes one page at its native width and height. Each accepted entry carries the one-based page, `test_point`, `value_point` and `value_text`. Points use pixels measured from the top-left corner. Their positions identify the observed words used for pairing. Test names and clinical codes are outside this example's output.

The baseline supports numeric readings with signs, decimal commas or points, and comparators such as `<0,1` and `≥5`. It joins nearby OCR fragments when their characters and positions support one reading. For example, separate `<` and `0,1` tokens become `<0,1`. Spaces between these fragments are omitted. If the OCR engine itself mistakes `≤` for `<`, the input to the row extractor already contains the wrong symbol. Located results that are unsupported or ambiguous go into the separate `review` array with their observed text, coordinates and reason. A missed row can be absent from both arrays. The scorer counts accepted predictions against every expected row. Use the example for extraction evaluation on synthetic reports.

## See what happens when the results trade rows

The first two rows read **Leucocitos 6,79** and **Hematíes 4,87**. The [swapped fixture](swapped.json) keeps the two test points and exchanges each result's text and point together. It submits only those two rows from the same 17-row page.

```sh
python scripts/score_row_associations.py --space image-pixels --coordinates examples/image-to-row/coordinates.json --ground-truth examples/visible.json --predictions examples/image-to-row/swapped.json
```

| Scorer output | Full saved replay | Two swapped predictions |
| --- | ---: | ---: |
| Submitted predictions | 17 | 2 |
| Exact text at the predicted value cell | 12 | 2 |
| Same-row pairs | 17 | 0 |
| Cross-row pairs | 0 | 2 |
| Correct row and exact printed value | 12 | 0 |

## Read the image again

Install the [repository's Python dependencies](../../requirements.txt) for Pillow, then install Tesseract and its Spanish and English language data. Check that `tesseract --list-langs` includes both `spa` and `eng`. This command reads the same single-page image at its native resolution. The results can vary by Tesseract build and language data. [Official installation instructions](https://tesseract-ocr.github.io/tessdoc/Installation.html)

```sh
python scripts/extract_ocr_rows.py --image examples/report.jpg --report-id RPT_000000 > fresh-predictions.json
python scripts/score_row_associations.py --space image-pixels --coordinates examples/image-to-row/coordinates.json --ground-truth examples/visible.json --predictions fresh-predictions.json
```

The saved TSV is the complete output of Tesseract 5.5.2 on the 1,191 × 1,684 pixel JPEG. To produce the same TSV format, use `tesseract examples/report.jpg stdout -l spa+eng --oem 1 --psm 6 tsv`. The Spanish and English traineddata SHA-256 digests for this run are `6f2e04d02774a18f01bed44b1111f2cd7f3ba7ac9dc4373cd3f898a40ea6b464` and `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`, respectively. The coordinate sidecar comes from the same generator quickstart as the JPEG and visible JSON, with its image path relative to the repository root.
