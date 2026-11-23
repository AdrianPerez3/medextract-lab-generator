# Contributing

Start by running the [quickstart](README.md#generate-your-first-reports) and opening a generated page next to its JSON label. The generator, validator and training-data splitter are the main user-facing commands.

## Report a problem

Include the command, seed, Python version and the difference between the expected and actual output. For a layout issue, include the synthetic report ID and page number. For a terminology issue, include the printed name, LOINC code, specimen or method when relevant, and a link to the official term. Use a synthetic fixture that others can share and regenerate.

## Test a change

With the project's Python 3.11 environment active, run the tests from the repository root.

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The tests live beside the modules in `scripts/lib/`. For a rendering or label fix, add a regression case that exercises the printed output and its corresponding target. For a change to bundled terminology, follow the source checks and attribution instructions in [TERMINOLOGY.md](TERMINOLOGY.md).

## Source archives

The [release builder](standalone_release/build_source_release.py) verifies the exact bytes of `reference_ranges.json`, `loinc_metadata.json` and `panel_definitions.json` against the three SHA-256 hashes in `DEMO_SHA256`. A mismatch stops the build. For an intentional change to a public fixture, review its full contents and provenance before updating its matching hash. Use `--data-dir` to run the generator with your own local inputs.

When a packaged file changes, rebuild the archive in `dist/` with the pinned official LOINC release and copy the generated `SHA256SUMS` from the archive to the repository root. [Archive build instructions](TERMINOLOGY.md#rebuild-the-demo-archive)
