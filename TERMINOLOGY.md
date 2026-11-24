# Terminology and data provenance

The demo contains 32 quantitative laboratory observations. For each LOINC code, the bundled catalog records its property, scale, official example UCUM units and the official names available to the generator. The terms and units come from *LOINC Complete 2.82*. All 32 identifiers were also checked against release 2.83 and were active. The numeric envelopes in the demo and the sampling rules in Python are simulation inputs. They establish no laboratory reference interval. [LOINC 2.82 release notes](https://loinc.org/kb/release-notes-archive/2.82) · [Bundled catalog](scripts/lib/resources/loinc_2_82.json) · [Numeric fixtures](src/paddle_ocr/dictionaries/data/reference_ranges.json)

## Names, units and credits

The default `routine-es` style uses a project-authored lookup of 32 Spanish print names. In the visible labels, `test_name` is the literal title on the page. In the normalized labels, the same code retains the catalog term selected for that observation. For example, `785-6` prints as `Hemoglobina corpuscular media (HCM)` while `718-7` prints as `Hemoglobina`. The local print names are distinct from the official LOINC names and translations. [Spanish print-name lookup](scripts/lib/resources/routine_es_names.json)

In the catalog, English labels are whole values from `SHORTNAME`, `LONG_COMMON_NAME` or `DisplayName` in `LoincTable/Loinc.csv`. Spanish labels are whole values from `SHORTNAME`, `LONG_COMMON_NAME` or `LinguisticVariantDisplayName`, or a fully specified name composed of the six official axes in order. Those axes are component, property, time aspect, system, scale and method. An empty method retains its position. The builder screens for names shared by different codes and skips entries with an external copyright notice or link. The catalog records a SHA-256 digest for each of the four input tables. [Release builder](standalone_release/build_source_release.py)

The 2.82 release's `LinguisticVariants.csv` credits the following producers for the three Spanish source files. [Official translation directory](https://loinc.org/international/)

| Source file | Translation producer |
|---|---|
| `esES12LinguisticVariant.csv` (Spain) | the Clinical Laboratory Committee of SERVICIO EXTREMEÑO DE SALUD, with the support of IQVIA |
| `esMX28LinguisticVariant.csv` (Mexico) | Manuel Aragonés. Deep Dive Data Science. |
| `esAR7LinguisticVariant.csv` (Argentina) | Conceptum Medical Terminology Center |

The Python source carries the MIT license. The original LOINC short notice and the complete license supplied with release 2.82 accompany this package. Keep codes with their corresponding official display names when redistributing the catalog. Make the notices accessible on the download page, including any separate release-asset page. The bundled license gives the conditions for incorporating or distributing LOINC content. [Code license](LICENSE) · [LOINC short notice](LOINC_short_license.txt) · [Bundled LOINC license](licenses/LoincLicense_5.8.txt) · [Upstream license](https://loinc.org/license/)

Unit strings come from the LOINC example UCUM field. Section G of the bundled license credits the UCUM table and codes to Regenstrief Institute, Inc. and the UCUM Organization. The upstream UCUM license covers the units specification and includes its warranty terms. [UCUM license](https://ucum.org/license)

## Build a custom vocabulary

The bundled demo runs offline as supplied. To rebuild the catalog, obtain *LOINC Complete 2.82* using your LOINC account. These commands assume you extracted it next to the project in `../Loinc_2.82`. The importer verifies all four source tables against the pinned digests. A new upstream version requires a reviewed importer update. [LOINC downloads and archive](https://loinc.org/downloads/)

Use a separate copy of the project, since the importer replaces that copy's runtime catalog. Create `custom-inputs` with `reference_ranges.json`, `loinc_metadata.json` and `panel_definitions.json`. The bundled data directory provides a working example. Each numeric entry needs its LOINC code, synthetic minimum and maximum, and a compatible unit from the official record. The metadata file can contain `{}` and the panel file can contain `{"panels": []}`. Check the current status of each added code. [Demo data directory](src/paddle_ocr/dictionaries/data/)

```sh
python -m scripts.lib.build_terminology --release ../Loinc_2.82 --ranges custom-inputs/reference_ranges.json
python generate_complete_synthetic.py --data-dir custom-inputs --count 8 --workers 1 --output custom-run
python scripts/validate_synthetic_dataset.py --dataset custom-run --data-dir custom-inputs --expected-count 8 --verify-source
```

The local importer can compose additional descriptive labels from translated axes. The distribution builder retains whole official display names or fully specified names for the demo. Review any added codes, labels, attribution and third-party terms before redistributing a custom index. For a custom binary result, the generator also requires a code-specific policy entry based on a published general answer list containing both Positive and Negative. The demo ships with an empty binary allowlist. [LOINC answer files](https://loinc.org/kb/accessory-files/answer-file/)

## Rebuild the demo archive

From the unmodified project with the licensed 2.82 release in the location above, run the allowlist-based builder. The ZIP contains the source, demo catalog, fixtures, examples, notices and a file-level checksum manifest.

```sh
python standalone_release/build_source_release.py --loinc-release ../Loinc_2.82 --output dist/medextract-lab-generator-source.zip
```
