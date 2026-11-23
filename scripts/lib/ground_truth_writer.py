"""Ground Truth Writer for ms-swift JSONL + optional GRPO/DPO formats.

Outputs training data in ms-swift conversation JSONL format for Qwen3-VL
SFT fine-tuning.  Also supports DPO preference pairs and GRPO prompt-only
format.  No WebDataset dependency -- just plain JSONL files.

Usage:
    writer = GroundTruthWriter(Path("/output"))
    writer.write_sft_example("RPT_000001", "images/RPT_000001.jpg", gt, seed=0)
    writer.close()

    split_dataset(Path("/output/sft_train.jsonl"), Path("/output"))
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
import json
import random
from io import TextIOWrapper
from pathlib import Path
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
    "2345-7": "2339-0",   # Glucose -> Glucose (urine)
    "2160-0": "38483-4",  # Creatinine -> Creatinine (urine)
    "1742-6": "1743-4",   # ALT -> ALT (no P5P)
    "1920-8": "1916-6",   # AST -> AST (no P5P)
    "2093-3": "2089-1",   # Cholesterol -> Cholesterol (calculated)
    "2571-8": "13457-7",  # Triglycerides -> LDL-C (calculated)
    "3016-3": "3024-7",   # TSH -> T4 free
    "718-7": "4544-3",    # Hemoglobin -> Hematocrit
    "789-8": "26453-1",   # RBC -> Erythrocytes (body fluid)
    "6690-2": "26464-8",  # WBC -> WBC (body fluid)
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

        self._sft_file: TextIOWrapper | None = None
        self._dpo_file: TextIOWrapper | None = None
        self._grpo_file: TextIOWrapper | None = None

        self._sft_count: int = 0
        self._dpo_count: int = 0
        self._grpo_count: int = 0

    @staticmethod
    def _image_prompt(image_path: str | list[str], seed: int) -> tuple[list[str], str]:
        images = [image_path] if isinstance(image_path, str) else image_path
        if not images or not all(isinstance(path, str) and path for path in images):
            raise ValueError("An image path is required for each report page")
        prompt = PROMPT_VARIANTS[seed % len(PROMPT_VARIANTS)]
        return images, "<image>" * len(images) + prompt

    # ------------------------------------------------------------------
    # SFT
    # ------------------------------------------------------------------

    def write_sft_example(
        self,
        report_id: str,
        image_path: str | list[str],
        ground_truth: dict[str, Any],
        seed: int,
    ) -> None:
        """Write one SFT training example in ms-swift conversation format.

        The output record has the structure expected by ms-swift:
        ``{"messages": [...], "images": ["images/RPT_000001.jpg"]}``

        Args:
            report_id: Unique report identifier (e.g. "RPT_000001").
            image_path: Relative path to the image from output_dir.
            ground_truth: Ground truth dict (typically ``{"results": [...]}``)
            seed: Integer seed used to rotate prompt variants.
        """
        images, prompt = self._image_prompt(image_path, seed)

        record: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        ground_truth, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "images": images,
        }

        self._ensure_sft_file()
        assert self._sft_file is not None
        self._sft_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._sft_count += 1

    # ------------------------------------------------------------------
    # DPO
    # ------------------------------------------------------------------

    def write_dpo_example(
        self,
        report_id: str,
        image_path: str | list[str],
        chosen_gt: dict[str, Any],
        rejected_gt: dict[str, Any],
        seed: int,
    ) -> None:
        """Write one DPO preference pair.

        The ``messages`` list contains the *chosen* response.  The
        ``rejected_response`` key contains the stringified rejected JSON.

        Args:
            report_id: Unique report identifier.
            image_path: Relative path to the image from output_dir.
            chosen_gt: Correct ground truth (chosen response).
            rejected_gt: Plausible-but-wrong variant (rejected response).
            seed: Integer seed for prompt rotation.
        """
        images, prompt = self._image_prompt(image_path, seed)

        record: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        chosen_gt, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "rejected_response": json.dumps(
                rejected_gt, ensure_ascii=False, separators=(",", ":")
            ),
            "images": images,
        }

        self._ensure_dpo_file()
        assert self._dpo_file is not None
        self._dpo_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._dpo_count += 1

    # ------------------------------------------------------------------
    # GRPO
    # ------------------------------------------------------------------

    def write_grpo_prompt(
        self,
        report_id: str,
        image_path: str | list[str],
        ground_truth: dict[str, Any],
        seed: int,
    ) -> None:
        """Write GRPO prompt-only format (no assistant response).

        The output record contains ``messages`` with only system + user turns,
        plus a ``solution`` key holding the ground truth for reward scoring.

        Args:
            report_id: Unique report identifier.
            image_path: Relative path to the image from output_dir.
            ground_truth: Ground truth dict for reward function scoring.
            seed: Integer seed for prompt rotation.
        """
        images, prompt = self._image_prompt(image_path, seed)

        record: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "images": images,
            "solution": json.dumps(
                ground_truth, ensure_ascii=False, separators=(",", ":")
            ),
        }

        self._ensure_grpo_file()
        assert self._grpo_file is not None
        self._grpo_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._grpo_count += 1

    # ------------------------------------------------------------------
    # DPO rejection generation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Dataset splitting
    # ------------------------------------------------------------------

    # (class-level utility, but also available as module-level function)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return write statistics.

        Returns:
            Dict with counts for each format and output file paths.
        """
        return {
            "sft_count": self._sft_count,
            "dpo_count": self._dpo_count,
            "grpo_count": self._grpo_count,
            "total": self._sft_count + self._dpo_count + self._grpo_count,
            "output_dir": str(self.output_dir),
            "images_dir": str(self.images_dir),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open file handles."""
        with ExitStack() as handles:
            for stream in (self._sft_file, self._dpo_file, self._grpo_file):
                if stream is not None:
                    handles.callback(stream.close)
            self._sft_file = self._dpo_file = self._grpo_file = None

    def __enter__(self) -> GroundTruthWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_sft_file(self) -> None:
        if self._sft_file is None:
            self._sft_file = open(
                self.output_dir / "sft_train.jsonl", "a", encoding="utf-8"
            )

    def _ensure_dpo_file(self) -> None:
        if self._dpo_file is None:
            self._dpo_file = open(
                self.output_dir / "dpo_train.jsonl", "a", encoding="utf-8"
            )

    def _ensure_grpo_file(self) -> None:
        if self._grpo_file is None:
            self._grpo_file = open(
                self.output_dir / "grpo_train.jsonl", "a", encoding="utf-8"
            )


# ---------------------------------------------------------------------------
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
    is reproducible.

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

    # Read all lines
    with open(sft_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(lines)

    total = len(lines)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    # Remainder goes to test
    n_test = total - n_train - n_val

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "sft_train.jsonl": lines[:n_train],
        "sft_val.jsonl": lines[n_train : n_train + n_val],
        "sft_test.jsonl": lines[n_train + n_val :],
    }

    for filename, split_lines in splits.items():
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for line in split_lines:
                if not line.endswith("\n"):
                    line += "\n"
                f.write(line)

    return {"train": n_train, "val": n_val, "test": n_test}
