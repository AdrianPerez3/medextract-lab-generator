# What the larger benchmark measured

The 32-code demo exercises the generation and validation workflow. A separate evaluation generated 70,000 synthetic reports containing 1,295,495 results across 1,286 codes marked active in LOINC 2.83. Reproducing those figures requires the larger input pack and evaluation harness. This repository distributes the small demo and the aggregate findings below.

| Check on the larger run | Result |
|---|---|
| Clean media | 87,105 PDF pages and matching JPEGs. Each page image decoded and the displayed result rows were checked against the PDF. |
| Simulated captures | 87,105 paired pages across five tiers. The clean tier shared 4,293 original pages. |
| Report-level split | 56,000 training, 7,000 validation and 7,000 test reports. |

The large run contained 1,086,765 numeric results across 56 codes, 21,250 Spanish positive or negative results across 125 codes, and 187,480 synthetic dilution titres across 1,105 codes. Each binary result matched an answer on a published list for its LOINC. The titre check verified dilution syntax only. The run omitted reference intervals and flags. [LOINC answer lists](https://loinc.org/kb/accessory-files/answer-file/)

## What an ordinary split can hide

An exact lookup keyed by the normalized printed name and unit recovered the code for all 129,465 observations in the ordinary test split. Of those, 118,511 (91.54%) repeated a normalized name, unit, printed value and code already present in training.

A second experiment grouped all occurrences of a normalized test name into one of five folds. For each test row, the matcher saw only training names from the other folds. It compared character substrings among candidates with the same printed unit. It ranked the assigned LOINC first for 97,729 of 129,465 test observations (75.49%). Giving each of the 1,286 codes equal weight yielded 25.46%. These experiments scored terminology retrieval from typeset labels. Real-world extraction performance and diagnostic accuracy remain unmeasured.

## What camera degradation changed

On 20 held-out single-page reports from the most severe tier, Tesseract 5.5.2 with Spanish and English language data recovered 891 of 904 expected analyte words on the clean images and one of 904 on their paired degraded images. The check counted distinct normalized analyte words of at least four letters on each page. A match anywhere on that page counted, including a word read from the wrong row. This measures readability under these simulated conditions.
