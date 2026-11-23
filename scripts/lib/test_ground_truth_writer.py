"""Tests for ground_truth_writer module.

Covers:
1. SFT JSONL format (messages + images keys, valid JSON per line)
2. DPO JSONL format (rejected_response field)
3. GRPO JSONL format (solution field, no assistant turn)
4. Prompt rotation across 6 variants
5. generate_dpo_rejected produces different output
6. split_dataset creates 3 files with correct proportions
7. JSON validity for all output types
8. close/reopen without data loss
9. Context manager support
10. Edge cases (empty ground truth, single result, etc.)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from .ground_truth_writer import (
    PROMPT_VARIANTS,
    SYSTEM_PROMPT,
    GroundTruthWriter,
    split_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_ground_truth(n_results: int = 5) -> dict:
    """Build a sample ground truth dict with n results."""
    results = []
    test_defs = [
        ("Glucosa", "95", "mg/dL", "70 - 110", None, "2345-7"),
        ("Creatinina", "1.2", "mg/dL", "0.7 - 1.3", None, "2160-0"),
        ("ALT (GPT)", "45", "U/L", "7 - 56", None, "1742-6"),
        ("Colesterol Total", "220", "mg/dL", "0 - 200", "H", "2093-3"),
        ("Hemoglobina", "14.5", "g/dL", "12.0 - 17.5", None, "718-7"),
        ("TSH", "3.5", "mU/L", "0.27 - 4.2", None, "3016-3"),
        ("Leucocitos", "8500", "10^3/uL", "4000 - 11000", None, "6690-2"),
        ("Trigliceridos", "180", "mg/dL", "0 - 150", "H", "2571-8"),
    ]
    for i in range(min(n_results, len(test_defs))):
        name, val, unit, ref, flag, loinc = test_defs[i]
        results.append({
            "test_name": name,
            "value": val,
            "unit": unit,
            "reference_range": ref,
            "flag": flag,
            "loinc": loinc,
        })
    return {"results": results}


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary output directory."""
    d = tmp_path / "gt_output"
    d.mkdir()
    return d


@pytest.fixture()
def writer(output_dir: Path) -> GroundTruthWriter:
    """Provide a GroundTruthWriter instance (caller must close)."""
    return GroundTruthWriter(output_dir)


@pytest.fixture()
def sample_gt() -> dict:
    return _sample_ground_truth(5)


# ---------------------------------------------------------------------------
# 1. SFT format tests
# ---------------------------------------------------------------------------


class TestSFTFormat:
    """Test SFT JSONL output format."""

    def test_write_5_sft_examples_produces_valid_jsonl(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Write 5 SFT examples and verify each line is valid JSON."""
        for i in range(5):
            writer.write_sft_example(
                f"RPT_{i + 1:06d}",
                f"images/RPT_{i + 1:06d}.jpg",
                sample_gt,
                seed=i,
            )
        writer.close()

        sft_path = output_dir / "sft_train.jsonl"
        assert sft_path.exists()

        lines = sft_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

        for line in lines:
            record = json.loads(line)
            assert "messages" in record
            assert "images" in record
            assert isinstance(record["messages"], list)
            assert isinstance(record["images"], list)

    def test_sft_messages_structure(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Verify messages list has system, user, assistant roles."""
        writer.write_sft_example("RPT_000001", "images/RPT_000001.jpg", sample_gt, seed=0)
        writer.close()

        line = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        messages = record["messages"]

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert messages[1]["content"].startswith("<image>")
        assert messages[2]["role"] == "assistant"

        # Assistant content should be parseable JSON
        assistant_content = json.loads(messages[2]["content"])
        assert "results" in assistant_content

    def test_sft_images_field(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Verify images field contains the correct path."""
        writer.write_sft_example(
            "RPT_000001", "images/RPT_000001.jpg", sample_gt, seed=0
        )
        writer.close()

        line = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["images"] == ["images/RPT_000001.jpg"]

    def test_sft_compact_json_separators(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Verify assistant content uses compact JSON (no spaces)."""
        writer.write_sft_example("RPT_000001", "images/RPT_000001.jpg", sample_gt, seed=0)
        writer.close()

        line = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        content = record["messages"][2]["content"]
        # Compact separators: no space after , or :
        assert ": " not in content or content.count(": ") == 0
        # Double-check by re-serializing
        parsed = json.loads(content)
        re_compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        assert content == re_compact

    def test_sft_spanish_characters_preserved(
        self, writer: GroundTruthWriter, output_dir: Path
    ) -> None:
        """Verify Spanish characters are preserved (ensure_ascii=False)."""
        gt = {"results": [{"test_name": "Bilirrubina Conjugada", "value": "0,3", "unit": "mg/dL", "reference_range": "0 - 0.3", "flag": None, "loinc": "1968-7"}]}
        writer.write_sft_example("RPT_000001", "images/RPT_000001.jpg", gt, seed=0)
        writer.close()

        content = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8")
        assert "Bilirrubina Conjugada" in content
        assert "\\u" not in content  # No unicode escapes


# ---------------------------------------------------------------------------
# 2. DPO format tests
# ---------------------------------------------------------------------------


class TestDPOFormat:
    """Test DPO JSONL output format."""

    def test_write_3_dpo_examples(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Write 3 DPO examples and verify rejected_response field."""
        rng = random.Random(42)
        for i in range(3):
            rejected = writer.generate_dpo_rejected(sample_gt, rng)
            writer.write_dpo_example(
                f"RPT_{i + 1:06d}",
                f"images/RPT_{i + 1:06d}.jpg",
                sample_gt,
                rejected,
                seed=i,
            )
        writer.close()

        dpo_path = output_dir / "dpo_train.jsonl"
        assert dpo_path.exists()

        lines = dpo_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            record = json.loads(line)
            assert "messages" in record
            assert "rejected_response" in record
            assert "images" in record

            # rejected_response should be parseable JSON string
            rejected_parsed = json.loads(record["rejected_response"])
            assert "results" in rejected_parsed

    def test_dpo_chosen_matches_ground_truth(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Verify the chosen (assistant) response matches ground truth."""
        rng = random.Random(42)
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        writer.write_dpo_example(
            "RPT_000001", "images/RPT_000001.jpg", sample_gt, rejected, seed=0
        )
        writer.close()

        line = (output_dir / "dpo_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        chosen_content = json.loads(record["messages"][2]["content"])
        assert chosen_content == sample_gt

    def test_dpo_rejected_differs_from_chosen(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Verify rejected response differs from chosen."""
        rng = random.Random(42)
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        writer.write_dpo_example(
            "RPT_000001", "images/RPT_000001.jpg", sample_gt, rejected, seed=0
        )
        writer.close()

        line = (output_dir / "dpo_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        chosen = json.loads(record["messages"][2]["content"])
        rejected_parsed = json.loads(record["rejected_response"])
        assert chosen != rejected_parsed


# ---------------------------------------------------------------------------
# 3. GRPO format tests
# ---------------------------------------------------------------------------


class TestGRPOFormat:
    """Test GRPO JSONL output format."""

    def test_write_3_grpo_prompts(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Write 3 GRPO prompts and verify solution field."""
        for i in range(3):
            writer.write_grpo_prompt(
                f"RPT_{i + 1:06d}",
                f"images/RPT_{i + 1:06d}.jpg",
                sample_gt,
                seed=i,
            )
        writer.close()

        grpo_path = output_dir / "grpo_train.jsonl"
        assert grpo_path.exists()

        lines = grpo_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            record = json.loads(line)
            assert "messages" in record
            assert "solution" in record
            assert "images" in record

            # Solution should be parseable JSON
            solution = json.loads(record["solution"])
            assert "results" in solution

    def test_grpo_no_assistant_turn(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """GRPO messages should only have system + user (no assistant)."""
        writer.write_grpo_prompt("RPT_000001", "images/RPT_000001.jpg", sample_gt, seed=0)
        writer.close()

        line = (output_dir / "grpo_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        messages = record["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # No assistant role
        roles = [m["role"] for m in messages]
        assert "assistant" not in roles


# ---------------------------------------------------------------------------
# 4. Prompt rotation
# ---------------------------------------------------------------------------


class TestPromptRotation:
    """Test that prompt variants rotate correctly."""

    def test_6_prompts_across_6_seeds(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """6 consecutive seeds should produce 6 different prompts."""
        for i in range(6):
            writer.write_sft_example(
                f"RPT_{i + 1:06d}",
                f"images/RPT_{i + 1:06d}.jpg",
                sample_gt,
                seed=i,
            )
        writer.close()

        lines = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        prompts_used = set()
        for line in lines:
            record = json.loads(line)
            user_content = record["messages"][1]["content"]
            # Strip the <image> prefix
            prompt_text = user_content.replace("<image>", "")
            prompts_used.add(prompt_text)

        # All 6 prompts should be different
        assert len(prompts_used) == 6
        # And they should match our defined variants
        for prompt in prompts_used:
            assert prompt in PROMPT_VARIANTS

    def test_seed_modulo_wraps(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """seed=6 should wrap to same prompt as seed=0."""
        writer.write_sft_example("RPT_A", "images/A.jpg", sample_gt, seed=0)
        writer.write_sft_example("RPT_B", "images/B.jpg", sample_gt, seed=6)
        writer.close()

        lines = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        record_0 = json.loads(lines[0])
        record_6 = json.loads(lines[1])

        prompt_0 = record_0["messages"][1]["content"]
        prompt_6 = record_6["messages"][1]["content"]
        assert prompt_0 == prompt_6


# ---------------------------------------------------------------------------
# 5. DPO rejected generation
# ---------------------------------------------------------------------------


class TestDPORejectedGeneration:
    """Test generate_dpo_rejected produces different output."""

    def test_rejected_differs_from_original(self, sample_gt: dict) -> None:
        """Rejected output must differ from the original ground truth."""
        writer = GroundTruthWriter(Path("/tmp/unused"))
        rng = random.Random(42)
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        assert rejected != sample_gt

    def test_rejected_preserves_results_key(self, sample_gt: dict) -> None:
        """Rejected output still has a 'results' key."""
        writer = GroundTruthWriter(Path("/tmp/unused"))
        rng = random.Random(42)
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        assert "results" in rejected

    def test_rejected_is_valid_json_serializable(self, sample_gt: dict) -> None:
        """Rejected output can be serialized to JSON."""
        writer = GroundTruthWriter(Path("/tmp/unused"))
        rng = random.Random(42)
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        serialized = json.dumps(rejected, ensure_ascii=False)
        reparsed = json.loads(serialized)
        assert reparsed == rejected

    def test_rejected_does_not_mutate_original(self, sample_gt: dict) -> None:
        """Original ground truth must not be modified."""
        original_copy = json.loads(json.dumps(sample_gt))
        writer = GroundTruthWriter(Path("/tmp/unused"))
        rng = random.Random(42)
        _ = writer.generate_dpo_rejected(sample_gt, rng)
        assert sample_gt == original_copy

    def test_multiple_rejections_vary(self, sample_gt: dict) -> None:
        """Different seeds should produce different rejected variants."""
        writer = GroundTruthWriter(Path("/tmp/unused"))
        rejections = set()
        for seed in range(20):
            rng = random.Random(seed)
            rejected = writer.generate_dpo_rejected(sample_gt, rng)
            rejections.add(json.dumps(rejected, sort_keys=True))
        # With 20 different seeds we should get at least a few different variants
        assert len(rejections) >= 3

    def test_rejected_with_empty_results(self) -> None:
        """Empty results list should return unchanged."""
        writer = GroundTruthWriter(Path("/tmp/unused"))
        rng = random.Random(42)
        gt = {"results": []}
        rejected = writer.generate_dpo_rejected(gt, rng)
        assert rejected == gt

    def test_rejected_with_single_result(self) -> None:
        """Single result: errors still apply but drop cannot remove all."""
        writer = GroundTruthWriter(Path("/tmp/unused"))
        gt = {"results": [{"test_name": "Glucosa", "value": "95", "unit": "mg/dL", "reference_range": "70 - 110", "flag": None, "loinc": "2345-7"}]}
        # Run many times to exercise different error paths
        for seed in range(50):
            rng = random.Random(seed)
            rejected = writer.generate_dpo_rejected(gt, rng)
            assert "results" in rejected
            # With a single result, results list should have at least 1 item
            # (missing_tests cannot drop the only one)
            assert len(rejected["results"]) >= 1


# ---------------------------------------------------------------------------
# 6. split_dataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    """Test the split_dataset function."""

    def test_split_creates_3_files(self, output_dir: Path) -> None:
        """Split should create train, val, test JSONL files."""
        source = output_dir / "source.jsonl"
        records = []
        for i in range(100):
            record = {"messages": [{"role": "user", "content": f"test {i}"}], "images": [f"img_{i}.jpg"]}
            records.append(json.dumps(record))
        source.write_text("\n".join(records) + "\n", encoding="utf-8")

        split_dir = output_dir / "splits"
        counts = split_dataset(source, split_dir)

        assert (split_dir / "sft_train.jsonl").exists()
        assert (split_dir / "sft_val.jsonl").exists()
        assert (split_dir / "sft_test.jsonl").exists()

        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

    def test_split_proportions_correct(self, output_dir: Path) -> None:
        """Verify line counts match expected proportions."""
        source = output_dir / "source.jsonl"
        records = []
        for i in range(200):
            record = {"id": i}
            records.append(json.dumps(record))
        source.write_text("\n".join(records) + "\n", encoding="utf-8")

        split_dir = output_dir / "splits"
        counts = split_dataset(source, split_dir, train_ratio=0.7, val_ratio=0.15)

        # 200 * 0.7 = 140, 200 * 0.15 = 30, remainder = 30
        assert counts["train"] == 140
        assert counts["val"] == 30
        assert counts["test"] == 30

        # Verify actual file line counts
        train_lines = (split_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        val_lines = (split_dir / "sft_val.jsonl").read_text(encoding="utf-8").strip().split("\n")
        test_lines = (split_dir / "sft_test.jsonl").read_text(encoding="utf-8").strip().split("\n")

        assert len(train_lines) == 140
        assert len(val_lines) == 30
        assert len(test_lines) == 30

    def test_split_is_deterministic(self, output_dir: Path) -> None:
        """Same seed produces the same split."""
        source = output_dir / "source.jsonl"
        records = [json.dumps({"id": i}) for i in range(50)]
        source.write_text("\n".join(records) + "\n", encoding="utf-8")

        split_a = output_dir / "split_a"
        split_b = output_dir / "split_b"

        split_dataset(source, split_a, seed=123)
        split_dataset(source, split_b, seed=123)

        for fname in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
            content_a = (split_a / fname).read_text(encoding="utf-8")
            content_b = (split_b / fname).read_text(encoding="utf-8")
            assert content_a == content_b

    def test_split_no_data_loss(self, output_dir: Path) -> None:
        """All records should appear exactly once across the 3 splits."""
        source = output_dir / "source.jsonl"
        ids = list(range(100))
        records = [json.dumps({"id": i}) for i in ids]
        source.write_text("\n".join(records) + "\n", encoding="utf-8")

        split_dir = output_dir / "splits"
        split_dataset(source, split_dir)

        all_ids: set[int] = set()
        for fname in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
            for line in (split_dir / fname).read_text(encoding="utf-8").strip().split("\n"):
                record = json.loads(line)
                all_ids.add(record["id"])

        assert all_ids == set(ids)

    def test_split_invalid_ratios(self, output_dir: Path) -> None:
        """Invalid ratios should raise ValueError."""
        source = output_dir / "source.jsonl"
        source.write_text('{"id": 1}\n', encoding="utf-8")

        with pytest.raises(ValueError, match="<= 1.0"):
            split_dataset(source, output_dir / "out", train_ratio=0.8, val_ratio=0.3)

        with pytest.raises(ValueError, match="non-negative"):
            split_dataset(source, output_dir / "out", train_ratio=-0.1)

    def test_split_small_dataset(self, output_dir: Path) -> None:
        """Splitting a very small dataset (3 items) should not crash."""
        source = output_dir / "source.jsonl"
        records = [json.dumps({"id": i}) for i in range(3)]
        source.write_text("\n".join(records) + "\n", encoding="utf-8")

        split_dir = output_dir / "splits"
        counts = split_dataset(source, split_dir)

        total = counts["train"] + counts["val"] + counts["test"]
        assert total == 3

    def test_split_shuffles(self, output_dir: Path) -> None:
        """Different seeds produce different orderings."""
        source = output_dir / "source.jsonl"
        records = [json.dumps({"id": i}) for i in range(50)]
        source.write_text("\n".join(records) + "\n", encoding="utf-8")

        split_a = output_dir / "split_a"
        split_b = output_dir / "split_b"

        split_dataset(source, split_a, seed=1)
        split_dataset(source, split_b, seed=999)

        # At least one file should differ between different seeds
        any_different = False
        for fname in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
            a = (split_a / fname).read_text(encoding="utf-8")
            b = (split_b / fname).read_text(encoding="utf-8")
            if a != b:
                any_different = True
                break
        assert any_different


# ---------------------------------------------------------------------------
# 7. JSON validity across all formats
# ---------------------------------------------------------------------------


class TestJSONValidity:
    """Verify all output formats produce valid, parseable JSON."""

    def test_all_formats_valid_json(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Write examples of all formats and verify JSON validity."""
        rng = random.Random(42)

        # SFT
        writer.write_sft_example("RPT_001", "images/001.jpg", sample_gt, seed=0)
        # DPO
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        writer.write_dpo_example("RPT_002", "images/002.jpg", sample_gt, rejected, seed=1)
        # GRPO
        writer.write_grpo_prompt("RPT_003", "images/003.jpg", sample_gt, seed=2)

        writer.close()

        # Validate SFT
        for line in (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().split("\n"):
            record = json.loads(line)
            assert isinstance(record, dict)
            json.loads(record["messages"][2]["content"])  # assistant content is valid JSON

        # Validate DPO
        for line in (output_dir / "dpo_train.jsonl").read_text(encoding="utf-8").strip().split("\n"):
            record = json.loads(line)
            assert isinstance(record, dict)
            json.loads(record["messages"][2]["content"])
            json.loads(record["rejected_response"])

        # Validate GRPO
        for line in (output_dir / "grpo_train.jsonl").read_text(encoding="utf-8").strip().split("\n"):
            record = json.loads(line)
            assert isinstance(record, dict)
            json.loads(record["solution"])


# ---------------------------------------------------------------------------
# 8. Close/reopen without data loss
# ---------------------------------------------------------------------------


class TestCloseReopen:
    """Test that closing and reopening preserves all data."""

    def test_close_reopen_preserves_sft(
        self, output_dir: Path, sample_gt: dict
    ) -> None:
        """Write 3, close, reopen, write 2 more -- all 5 should be present."""
        writer = GroundTruthWriter(output_dir)
        for i in range(3):
            writer.write_sft_example(f"RPT_{i}", f"images/{i}.jpg", sample_gt, seed=i)
        writer.close()

        # Reopen (new instance, same directory)
        writer2 = GroundTruthWriter(output_dir)
        for i in range(3, 5):
            writer2.write_sft_example(f"RPT_{i}", f"images/{i}.jpg", sample_gt, seed=i)
        writer2.close()

        lines = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

        # All should be valid JSON
        for line in lines:
            json.loads(line)

    def test_close_reopen_preserves_dpo(
        self, output_dir: Path, sample_gt: dict
    ) -> None:
        """DPO file should also accumulate across close/reopen."""
        rng = random.Random(42)

        writer = GroundTruthWriter(output_dir)
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        writer.write_dpo_example("RPT_1", "images/1.jpg", sample_gt, rejected, seed=0)
        writer.close()

        writer2 = GroundTruthWriter(output_dir)
        rejected2 = writer2.generate_dpo_rejected(sample_gt, rng)
        writer2.write_dpo_example("RPT_2", "images/2.jpg", sample_gt, rejected2, seed=1)
        writer2.close()

        lines = (output_dir / "dpo_train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_close_reopen_preserves_grpo(
        self, output_dir: Path, sample_gt: dict
    ) -> None:
        """GRPO file should also accumulate across close/reopen."""
        writer = GroundTruthWriter(output_dir)
        writer.write_grpo_prompt("RPT_1", "images/1.jpg", sample_gt, seed=0)
        writer.close()

        writer2 = GroundTruthWriter(output_dir)
        writer2.write_grpo_prompt("RPT_2", "images/2.jpg", sample_gt, seed=1)
        writer2.close()

        lines = (output_dir / "grpo_train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# 9. Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Test the context manager interface."""

    def test_context_manager_closes_files(
        self, output_dir: Path, sample_gt: dict
    ) -> None:
        """Using 'with' should auto-close files."""
        with GroundTruthWriter(output_dir) as w:
            w.write_sft_example("RPT_1", "images/1.jpg", sample_gt, seed=0)

        # File should be closed and readable
        lines = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        json.loads(lines[0])

    def test_context_manager_returns_self(self, output_dir: Path) -> None:
        """__enter__ should return the writer instance."""
        writer = GroundTruthWriter(output_dir)
        with writer as w:
            assert w is writer


# ---------------------------------------------------------------------------
# 10. Stats
# ---------------------------------------------------------------------------


class TestStats:
    """Test stats() reporting."""

    def test_stats_counts(
        self, writer: GroundTruthWriter, output_dir: Path, sample_gt: dict
    ) -> None:
        """Stats should reflect the number of written examples."""
        rng = random.Random(42)

        writer.write_sft_example("RPT_1", "images/1.jpg", sample_gt, seed=0)
        writer.write_sft_example("RPT_2", "images/2.jpg", sample_gt, seed=1)
        rejected = writer.generate_dpo_rejected(sample_gt, rng)
        writer.write_dpo_example("RPT_3", "images/3.jpg", sample_gt, rejected, seed=2)
        writer.write_grpo_prompt("RPT_4", "images/4.jpg", sample_gt, seed=3)

        s = writer.stats()
        assert s["sft_count"] == 2
        assert s["dpo_count"] == 1
        assert s["grpo_count"] == 1
        assert s["total"] == 4
        assert s["output_dir"] == str(output_dir)

        writer.close()

    def test_stats_initial(self, writer: GroundTruthWriter) -> None:
        """Fresh writer should have zero counts."""
        s = writer.stats()
        assert s["sft_count"] == 0
        assert s["dpo_count"] == 0
        assert s["grpo_count"] == 0
        assert s["total"] == 0


# ---------------------------------------------------------------------------
# 11. Images directory creation
# ---------------------------------------------------------------------------


class TestImagesDirectory:
    """Test that images subdirectory is created correctly."""

    def test_default_images_subdir(self, tmp_path: Path) -> None:
        """Default images subdir should be 'images'."""
        writer = GroundTruthWriter(tmp_path / "out")
        assert (tmp_path / "out" / "images").is_dir()
        writer.close()

    def test_custom_images_subdir(self, tmp_path: Path) -> None:
        """Custom images subdir should be created."""
        writer = GroundTruthWriter(tmp_path / "out", images_subdir="my_images")
        assert (tmp_path / "out" / "my_images").is_dir()
        writer.close()


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_ground_truth(
        self, writer: GroundTruthWriter, output_dir: Path
    ) -> None:
        """Empty ground truth should still produce valid JSONL."""
        gt: dict[str, Any] = {"results": []}
        writer.write_sft_example("RPT_1", "images/1.jpg", gt, seed=0)
        writer.close()

        line = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        content = json.loads(record["messages"][2]["content"])
        assert content == {"results": []}

    def test_ground_truth_with_none_flags(
        self, writer: GroundTruthWriter, output_dir: Path
    ) -> None:
        """None flags should be serialized as JSON null."""
        gt = {"results": [{"test_name": "Test", "value": "1", "unit": "U", "reference_range": "0-2", "flag": None, "loinc": "0000-0"}]}
        writer.write_sft_example("RPT_1", "images/1.jpg", gt, seed=0)
        writer.close()

        line = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        content = json.loads(record["messages"][2]["content"])
        assert content["results"][0]["flag"] is None

    def test_double_close_is_safe(self, output_dir: Path, sample_gt: dict) -> None:
        """Calling close() twice should not raise."""
        writer = GroundTruthWriter(output_dir)
        writer.write_sft_example("RPT_1", "images/1.jpg", sample_gt, seed=0)
        writer.close()
        writer.close()  # Should not raise

    def test_large_ground_truth(
        self, writer: GroundTruthWriter, output_dir: Path
    ) -> None:
        """Large ground truth (30+ results) should serialize correctly."""
        gt = _sample_ground_truth(8)
        # Duplicate results to get 32
        gt["results"] = gt["results"] * 4
        assert len(gt["results"]) == 32

        writer.write_sft_example("RPT_1", "images/1.jpg", gt, seed=0)
        writer.close()

        line = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        content = json.loads(record["messages"][2]["content"])
        assert len(content["results"]) == 32
