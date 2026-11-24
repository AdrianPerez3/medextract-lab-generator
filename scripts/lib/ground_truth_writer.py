"""Ground Truth Writer for ms-swift JSONL + optional GRPO/DPO formats.

Outputs training data in ms-swift conversation JSONL format for Qwen3-VL
SFT fine-tuning.  Also supports DPO preference pairs and GRPO prompt-only
format.  No WebDataset dependency -- just plain JSONL files.

Usage:
    writer = GroundTruthWriter(Path("/output"))
    writer.write_sft_example("RPT_000001", "images/RPT_000001.jpg", gt, seed=0)
    writer.close()

    split_dataset(Path("/output/sft_train.jsonl"), Path("/output/splits"))
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
import json
import random
from io import TextIOWrapper
from itertools import islice
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from stat import S_IMODE, S_ISREG
from typing import Any


# ---------------------------------------------------------------------------
# Prompt variants -- 6 rotations to prevent overfitting to one prompt
# ---------------------------------------------------------------------------

PROMPT_VARIANTS: list[str] = [
    "Extrae todos los resultados de laboratorio de este informe medico en formato JSON.",
    (
        "Analiza este informe de laboratorio y extrae los resultados con "
        "nombre del test, valor, unidad, rango de referencia y codigo LOINC."
    ),
    (
        "Identifica y extrae todos los analisis clinicos de este informe. "
        "Devuelve JSON con test_name, value, unit, reference_range, flag y loinc."
    ),
    (
        "Procesa este informe de analisis clinicos y devuelve los resultados "
        "estructurados en formato JSON."
    ),
    (
        "Extrae la informacion de cada prueba de laboratorio de esta imagen: "
        "nombre, resultado, unidades, valores de referencia y codigo LOINC."
    ),
    (
        "Lee este informe de laboratorio medico y genera un JSON con todos "
        "los resultados encontrados."
    ),
]

SYSTEM_PROMPT: str = (
    "Eres un experto en interpretacion de informes de laboratorio medico. "
    "Extrae todos los resultados de laboratorio del informe proporcionado "
    "y devuelve un JSON estructurado con los campos: test_name, value, unit, "
    "reference_range, flag (H/L/null) y loinc (codigo LOINC)."
)

# ---------------------------------------------------------------------------
# DPO rejection error patterns
# ---------------------------------------------------------------------------

# Common LOINC swaps (similar tests that are easy to confuse)
_LOINC_SWAPS: dict[str, str] = {
    "2345-7": "2339-0",   # Serum/plasma glucose -> blood glucose
    "2160-0": "38483-4",  # Serum/plasma creatinine -> blood creatinine
    "1742-6": "1743-4",   # ALT, method unspecified -> ALT with P-5'-P
    "1920-8": "1916-6",   # AST activity -> AST/ALT activity ratio
    "2093-3": "2089-1",   # Total cholesterol -> LDL cholesterol, method unspecified
    "2571-8": "13457-7",  # Triglycerides -> LDL-C (calculated)
    "3016-3": "3024-7",   # TSH -> T4 free
    "718-7": "4544-3",    # Hemoglobin -> Hematocrit
    "789-8": "26453-1",   # Blood erythrocytes, automated count -> method unspecified
    "6690-2": "26464-8",  # Blood leukocytes, automated count -> method unspecified
}

# Common unit confusions
_UNIT_SWAPS: dict[str, str] = {
    "mg/dL": "g/dL",
    "g/dL": "mg/dL",
    "mU/L": "U/L",
    "U/L": "mU/L",
    "mmol/L": "mEq/L",
    "mEq/L": "mmol/L",
    "ng/mL": "pg/mL",
    "pg/mL": "ng/mL",
    "uIU/mL": "mIU/L",
    "10^3/uL": "10^6/uL",
    "10^6/uL": "10^3/uL",
}


class GroundTruthWriter:
    """Writes training data in ms-swift JSONL format.

    Supports three output formats:
    - SFT (Supervised Fine-Tuning): full conversation with assistant response
    - DPO (Direct Preference Optimization): chosen + rejected response pairs
    - GRPO (Group Relative Policy Optimization): prompt-only with solution key
    """

    def __init__(self, output_dir: Path, images_subdir: str = "images") -> None:
        """Initialize the writer.

        Args:
            output_dir: Base output directory for JSONL files.
            images_subdir: Subdirectory name for images (relative to output_dir).
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / images_subdir
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._streams: dict[str, TextIOWrapper] = {}
        self._counts = {"sft": 0, "dpo": 0, "grpo": 0}

    @staticmethod
    def _image_prompt(image_path: str | list[str], seed: int) -> tuple[list[str], str]:
        images = [image_path] if isinstance(image_path, str) else image_path
        if not images or not all(isinstance(path, str) and path for path in images):
            raise ValueError("An image path is required for each report page")
        prompt = PROMPT_VARIANTS[seed % len(PROMPT_VARIANTS)]
        return images, "<image>" * len(images) + prompt

    def write_examples(
        self,
        image_path: str | list[str],
        ground_truth: dict[str, Any],
        seed: int,
        *,
        sft: bool = False,
        dpo_rejected: dict[str, Any] | None = None,
        grpo: bool = False,
    ) -> None:
        """Write the selected formats for one report, serializing the shared target once."""
        if not sft and not grpo and dpo_rejected is None:
            raise ValueError("Select at least one training export")
        images, prompt = self._image_prompt(image_path, seed)
        chosen = json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":"))
        # Validate both payloads before appending to any export. A caller can
        # correct a non-serializable rejection and retry the same report.
        negative = None
        if dpo_rejected is not None:
            negative = json.dumps(dpo_rejected, ensure_ascii=False, separators=(",", ":"))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        if sft or negative is not None:
            conversation = [*messages, {"role": "assistant", "content": chosen}]
            if sft:
                self._write_record("sft", {"messages": conversation, "images": images})
            if negative is not None:
                self._write_record("dpo", {"messages": conversation, "rejected_response": negative, "images": images})
        if grpo:
            self._write_record("grpo", {"messages": messages, "images": images, "solution": chosen})

    def _write_record(self, kind: str, record: dict[str, Any]) -> None:
        stream = self._streams.get(kind)
        if stream is None:
            stream = (self.output_dir / f"{kind}_train.jsonl").open("a", encoding="utf-8")
            self._streams[kind] = stream
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._counts[kind] += 1

    def write_sft_example(self, report_id: str, image_path: str | list[str],
                          ground_truth: dict[str, Any], seed: int) -> None:
        """Append a supervised conversation with the visible label as the assistant response."""
        self.write_examples(image_path, ground_truth, seed, sft=True)

    def write_dpo_example(self, report_id: str, image_path: str | list[str],
                          chosen_gt: dict[str, Any], rejected_gt: dict[str, Any], seed: int) -> None:
        """Append the chosen conversation and its explicitly supplied negative response."""
        self.write_examples(image_path, chosen_gt, seed, dpo_rejected=rejected_gt)

    def write_grpo_prompt(self, report_id: str, image_path: str | list[str],
                          ground_truth: dict[str, Any], seed: int) -> None:
        """Append the two-turn prompt and its separate solution."""
        self.write_examples(image_path, ground_truth, seed, grpo=True)

    def generate_dpo_rejected(
        self,
        ground_truth: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        """Generate a plausible-but-wrong variant of the ground truth for DPO.

        Applies 1-3 random error patterns drawn from:
        1. Wrong LOINC code (swap with a similar test)
        2. Value decimal shift (95 -> 9.5 or 950)
        3. Missing tests (drop 1-3 results)
        4. Unit confusion (mg/dL -> g/dL)
        5. Wrong flag (H instead of L, or vice versa)
        6. Hallucinated extra test

        Args:
            ground_truth: The correct ground truth dict.
            rng: Seeded Random instance for reproducibility.

        Returns:
            A modified copy of ground_truth containing realistic errors.
        """
        rejected = copy.deepcopy(ground_truth)
        results: list[dict[str, Any]] = rejected.get("results", [])

        if not results:
            return rejected

        # Pick 1-3 error types to apply
        error_types = [
            self._error_wrong_loinc,
            self._error_decimal_shift,
            self._error_missing_tests,
            self._error_unit_confusion,
            self._error_wrong_flag,
            self._error_hallucinated_test,
        ]
        n_errors = rng.randint(1, min(3, len(error_types)))
        chosen_errors = rng.sample(error_types, n_errors)

        for error_fn in chosen_errors:
            error_fn(results, rng)

        if rejected == ground_truth:
            self._error_wrong_flag(results, rng)
        rejected["results"] = results
        return rejected

    @staticmethod
    def _error_wrong_loinc(
        results: list[dict[str, Any]], rng: random.Random
    ) -> None:
        """Swap a LOINC code with a confusable one."""
        eligible = [r for r in results if r.get("loinc") in _LOINC_SWAPS]
        if not eligible:
            # Fallback: pick any result and deterministically shift the check digit
            if results:
                target = rng.choice(results)
                loinc = target.get("loinc", "")
                if loinc and "-" in loinc:
                    parts = loinc.rsplit("-", 1)
                    check = int(parts[1])
                    new_check = (check + 1) % 10  # Always different, deterministic
                    target["loinc"] = f"{parts[0]}-{new_check}"
            return
        target = rng.choice(eligible)
        target["loinc"] = _LOINC_SWAPS[target["loinc"]]

    @staticmethod
    def _error_decimal_shift(
        results: list[dict[str, Any]], rng: random.Random
    ) -> None:
        """Shift a numeric value by a factor of 10 or 0.1."""
        numeric = [r for r in results if _is_numeric_value(r.get("value", ""))]
        if not numeric:
            return
        target = rng.choice(numeric)
        val_str = str(target["value"])
        is_european = "," in val_str and "." not in val_str
        try:
            val = float(val_str.replace(",", "."))
            factor = rng.choice([0.1, 10.0])
            new_val = val * factor
            # Format: keep reasonable precision
            formatted = f"{new_val:.2f}".rstrip("0").rstrip(".")
            if is_european:
                formatted = formatted.replace(".", ",")
            target["value"] = formatted
        except (ValueError, TypeError):
            pass

    @staticmethod
    def _error_missing_tests(
        results: list[dict[str, Any]], rng: random.Random
    ) -> None:
        """Drop 1-3 tests from the results."""
        if len(results) <= 1:
            return
        n_drop = rng.randint(1, min(3, len(results) - 1))
        indices_to_drop = set(rng.sample(range(len(results)), n_drop))
        # Modify in-place by removing from end to preserve indices
        for idx in sorted(indices_to_drop, reverse=True):
            results.pop(idx)

    @staticmethod
    def _error_unit_confusion(
        results: list[dict[str, Any]], rng: random.Random
    ) -> None:
        """Swap a unit with a confusable one."""
        eligible = [r for r in results if r.get("unit") in _UNIT_SWAPS]
        if not eligible:
            return
        target = rng.choice(eligible)
        target["unit"] = _UNIT_SWAPS[target["unit"]]

    @staticmethod
    def _error_wrong_flag(
        results: list[dict[str, Any]], rng: random.Random
    ) -> None:
        """Swap H<->L or add a spurious flag."""
        flagged = [r for r in results if r.get("flag") in ("H", "L")]
        if flagged:
            target = rng.choice(flagged)
            target["flag"] = "L" if target["flag"] == "H" else "H"
        elif results:
            # Add a spurious flag to a normal result
            target = rng.choice(results)
            target["flag"] = rng.choice(["H", "L"])

    @staticmethod
    def _error_hallucinated_test(
        results: list[dict[str, Any]], rng: random.Random
    ) -> None:
        """Add a hallucinated test that does not exist in the report."""
        hallucinated_tests = [
            {
                "test_name": "Vitamina D",
                "value": "28.5",
                "unit": "ng/mL",
                "reference_range": "30 - 100",
                "flag": "L",
                "loinc": "1989-3",
            },
            {
                "test_name": "Ferritina",
                "value": "145",
                "unit": "ng/mL",
                "reference_range": "20 - 250",
                "flag": None,
                "loinc": "2276-4",
            },
            {
                "test_name": "PSA Total",
                "value": "2.8",
                "unit": "ng/mL",
                "reference_range": "0 - 4",
                "flag": None,
                "loinc": "2857-1",
            },
            {
                "test_name": "Acido Urico",
                "value": "7.2",
                "unit": "mg/dL",
                "reference_range": "3.4 - 7.0",
                "flag": "H",
                "loinc": "3084-1",
            },
        ]
        results.append(rng.choice(hallucinated_tests))

    def stats(self) -> dict[str, Any]:
        """Return write statistics.

        Returns:
            Dict with counts for each format and output file paths.
        """
        return {
            **{f"{kind}_count": count for kind, count in self._counts.items()},
            "total": sum(self._counts.values()),
            "output_dir": str(self.output_dir),
            "images_dir": str(self.images_dir),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open file handles."""
        with ExitStack() as handles:
            for stream in self._streams.values():
                handles.callback(stream.close)
            self._streams.clear()

    def __enter__(self) -> GroundTruthWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Module-level utilities
# ---------------------------------------------------------------------------


def _is_numeric_value(val: str) -> bool:
    """Check whether a string looks like a numeric lab value."""
    try:
        float(str(val).replace(",", "."))
        return True
    except (ValueError, TypeError):
        return False


def split_dataset(
    sft_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Split an SFT JSONL file into train/val/test splits.

    Creates:
    - ``output_dir/sft_train.jsonl``  (train_ratio, default 80%)
    - ``output_dir/sft_val.jsonl``    (val_ratio, default 10%)
    - ``output_dir/sft_test.jsonl``   (remainder, default 10%)

    Shuffles with a deterministic seed before splitting so that the split
    is reproducible. Concurrent writers to the same directory fail fast. An
    interrupted process can leave .sft-split-staging with its staged files
    and any originals moved aside during publication.

    Args:
        sft_path: Path to the source JSONL file.
        output_dir: Directory where the split files will be written.
        train_ratio: Fraction of data for training (0..1).
        val_ratio: Fraction of data for validation (0..1).
        seed: Random seed for deterministic shuffling.

    Returns:
        Dict with counts: ``{"train": N, "val": N, "test": N}``.

    Raises:
        ValueError: If ratios are invalid (sum > 1 or negative).
        FileNotFoundError: If sft_path does not exist.
    """
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("Ratios must be non-negative")
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be <= 1.0, "
            f"got {train_ratio} + {val_ratio} = {train_ratio + val_ratio}"
        )

    source = Path(sft_path)
    output_dir = Path(output_dir)
    # Direct callers accept blank/empty streams and may replace existing splits.
    # The packaging CLI separately validates JSON objects.
    from .jsonl_split import check_source_unchanged, shuffled_rows

    with source.open(encoding="utf-8", newline="") as stream, ExitStack() as cleanup:
        original = os.fstat(stream.fileno())
        output_dir.mkdir(parents=True, exist_ok=True)
        # The fixed directory reserves the three sibling names across processes.
        reservation = output_dir / ".sft-split-staging"
        try:
            reservation.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Cannot claim split staging directory (in use or left by an interrupted split): {reservation}"
            ) from exc
        cleanup.callback(reservation.rmdir)
        staging = Path(cleanup.enter_context(TemporaryDirectory(dir=reservation)))

        previous_modes = {}
        for split in ("train", "val", "test"):
            target = output_dir / f"sft_{split}.jsonl"
            try:
                status = target.lstat()
            except FileNotFoundError:
                continue
            if os.path.samestat(original, status):
                raise ValueError("Split output cannot overwrite its training source")
            if not S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise OSError(f"Split output must be a single-link regular file: {target}")
            previous_modes[split] = S_IMODE(status.st_mode)

        total, rows = shuffled_rows(stream, seed)
        n_train, n_val = int(total * train_ratio), int(total * val_ratio)
        counts = {"train": n_train, "val": n_val, "test": total - n_train - n_val}
        for split, count in counts.items():
            with (staging / split).open("xb") as target:
                target.writelines(islice(rows, count))
            if split in previous_modes:
                (staging / split).chmod(previous_modes[split])
        check_source_unchanged(source, original)

        # Keep the old entries until every replacement succeeds.
        with ExitStack() as rollback:
            for split in counts:
                target = output_dir / f"sft_{split}.jsonl"
                if split in previous_modes:
                    backup = staging / f"{split}.old"
                    target.replace(backup)
                    rollback.callback(backup.replace, target)
                (staging / split).replace(target)
                rollback.callback(target.unlink, missing_ok=True)
            rollback.pop_all()
    return counts
