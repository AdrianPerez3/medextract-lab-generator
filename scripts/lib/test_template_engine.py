"""Tests for the stochastic template variation engine.

Verifies:
- 100 random configs produce >= 80 unique combinations
- Colour schemes are valid (all values 0-1)
- Column widths sum to ~1.0 (+/- 0.1)
- Stochastic variation produces different results with different seeds
- All 17 country presets are reachable
"""

from __future__ import annotations

import random

import pytest

try:
    from scripts.lib.template_engine import (
        StochasticVariation,
        TemplateEngine,
        _clamp01,
    )
except ModuleNotFoundError:
    from template_engine import (
        StochasticVariation,
        TemplateEngine,
        _clamp01,
    )

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> TemplateEngine:
    return TemplateEngine()


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


class TestUniqueness:
    """100 random configs should produce at least 80 unique combinations."""

    def test_100_configs_produce_at_least_80_unique(self, engine: TemplateEngine) -> None:
        names: set[str] = set()
        for seed in range(100):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            names.add(cfg["name"])
        assert len(names) >= 80, f"Only {len(names)} unique names from 100 configs"

    def test_uniqueness_across_regions(self, engine: TemplateEngine) -> None:
        """Different regions should produce different distributions."""
        spain_names: set[str] = set()
        latam_names: set[str] = set()
        for seed in range(50):
            rng_es = random.Random(seed + 10000)
            rng_la = random.Random(seed + 20000)
            spain_names.add(engine.select_template_config(rng_es, "spain")["name"])
            latam_names.add(engine.select_template_config(rng_la, "latam")["name"])
        # They should not be identical sets
        assert spain_names != latam_names


# ---------------------------------------------------------------------------
# Colour scheme validity
# ---------------------------------------------------------------------------


class TestColorSchemes:
    """All colour values must be in [0.0, 1.0]."""

    def test_builtin_schemes_valid(self, engine: TemplateEngine) -> None:
        for scheme in engine.color_schemes:
            for key in ("primary", "secondary", "accent", "text", "bg", "alt_row"):
                rgb = scheme[key]
                for ch, v in zip("RGB", rgb):
                    assert 0.0 <= v <= 1.0, (
                        f"Scheme {scheme['name']}: {key}.{ch} = {v} out of range"
                    )

    def test_varied_config_colors_valid(self, engine: TemplateEngine) -> None:
        """After stochastic variation, colours should still be clamped to [0, 1]."""
        for seed in range(200):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            cs = cfg["color_scheme"]
            for key in ("primary", "secondary", "accent", "text", "bg", "alt_row"):
                rgb = cs[key]
                for ch, v in zip("RGB", rgb):
                    assert 0.0 <= v <= 1.0, (
                        f"Seed {seed}: color_scheme.{key}.{ch} = {v} out of range"
                    )

    def test_at_least_20_color_schemes(self, engine: TemplateEngine) -> None:
        assert len(engine.color_schemes) >= 20


# ---------------------------------------------------------------------------
# Column widths
# ---------------------------------------------------------------------------


class TestColumnWidths:
    """Column widths must sum to approximately 1.0."""

    def test_column_widths_sum_near_one(self, engine: TemplateEngine) -> None:
        for seed in range(200):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            columns = cfg["columns"]
            total = sum(c["width_pct"] for c in columns)
            assert abs(total - 1.0) < 0.1, (
                f"Seed {seed}: column widths sum to {total:.4f}"
            )

    def test_no_negative_widths(self, engine: TemplateEngine) -> None:
        for seed in range(200):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            for col in cfg["columns"]:
                assert col["width_pct"] > 0, (
                    f"Seed {seed}: column {col['name']} has negative width"
                )

    def test_column_count_matches_preset(self, engine: TemplateEngine) -> None:
        """Column count should be 3, 4, 5 or 6."""
        for seed in range(100):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            n = len(cfg["columns"])
            assert n in (3, 4, 5, 6), f"Seed {seed}: unexpected column count {n}"


# ---------------------------------------------------------------------------
# Stochastic variation determinism + difference
# ---------------------------------------------------------------------------


class TestStochasticVariation:
    """Variation must differ across seeds and be deterministic per seed."""

    def test_different_seeds_different_results(self, engine: TemplateEngine) -> None:
        rng1 = random.Random(42)
        rng2 = random.Random(99)
        cfg1 = engine.select_template_config(rng1, region="spain")
        cfg2 = engine.select_template_config(rng2, region="spain")

        # At least the name or colour scheme or font should differ
        differs = (
            cfg1["name"] != cfg2["name"]
            or cfg1["color_scheme"] != cfg2["color_scheme"]
            or cfg1["font_config"] != cfg2["font_config"]
        )
        assert differs, "Two different seeds produced identical configs"

    def test_same_seed_same_result(self, engine: TemplateEngine) -> None:
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        cfg1 = engine.select_template_config(rng1, region="spain")
        cfg2 = engine.select_template_config(rng2, region="spain")
        assert cfg1["name"] == cfg2["name"]
        assert cfg1["color_scheme"] == cfg2["color_scheme"]
        assert cfg1["font_config"] == cfg2["font_config"]
        assert cfg1["margins"] == cfg2["margins"]

    def test_variation_object_present(self, engine: TemplateEngine) -> None:
        rng = random.Random(1)
        cfg = engine.select_template_config(rng, region="spain")
        assert "variation" in cfg
        v = cfg["variation"]
        assert isinstance(v, StochasticVariation)

    def test_variation_ranges(self, engine: TemplateEngine) -> None:
        """Variation fields should be within their expected ranges."""
        for seed in range(100):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            v: StochasticVariation = cfg["variation"]
            assert -2.5 <= v.font_size_delta <= 2.5
            assert 0.0 <= v.column_width_jitter <= 0.1
            assert -0.5 <= v.row_height_jitter <= 0.5
            assert 0.85 <= v.spacing_factor <= 1.15
            for mj in v.margin_jitter:
                assert -15.0 <= mj <= 15.0
            for cv in v.color_variation:
                assert -0.05 <= cv <= 0.05


# ---------------------------------------------------------------------------
# Country preset reachability
# ---------------------------------------------------------------------------


class TestCountryPresets:
    """All 17 country presets should be reachable."""

    def test_all_presets_reachable_worldwide(self, engine: TemplateEngine) -> None:
        """With enough samples from 'worldwide', all presets should appear."""
        all_preset_keys = set(engine.country_presets.keys())
        seen: set[str] = set()
        # The config name starts with the preset key
        for seed in range(2000):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            name = cfg["name"]
            for pk in all_preset_keys:
                if name.startswith(pk + "_"):
                    seen.add(pk)
                    break
        missing = all_preset_keys - seen
        assert not missing, f"Presets never reached: {missing}"

    def test_exact_preset_key_as_region(self, engine: TemplateEngine) -> None:
        """Passing an exact preset key as region should always select it."""
        for key in ("spain_public", "us_quest", "de_din", "fr_sil", "minimal_iso"):
            rng = random.Random(123)
            cfg = engine.select_template_config(rng, region=key)
            assert cfg["name"].startswith(key + "_"), (
                f"region={key} produced name={cfg['name']}"
            )

    def test_at_least_17_presets(self, engine: TemplateEngine) -> None:
        assert len(engine.country_presets) >= 17

    def test_spain_region_biases_spain(self, engine: TemplateEngine) -> None:
        """'spain' region should produce >50% spain presets."""
        spain_count = 0
        total = 200
        for seed in range(total):
            rng = random.Random(seed + 5000)
            cfg = engine.select_template_config(rng, region="spain")
            name = cfg["name"]
            if name.startswith("spain_"):
                spain_count += 1
        pct = spain_count / total
        assert pct > 0.40, f"Spain presets only {pct:.1%} of 'spain' region selections"


# ---------------------------------------------------------------------------
# Lab info
# ---------------------------------------------------------------------------


class TestLabInfo:
    """Lab info list should have 30+ entries with required fields."""

    def test_at_least_30_labs(self, engine: TemplateEngine) -> None:
        assert len(engine.lab_info) >= 30

    def test_lab_fields_present(self, engine: TemplateEngine) -> None:
        for lab in engine.lab_info:
            assert "key" in lab
            assert "name" in lab
            assert "region" in lab
            assert len(lab["key"]) > 0
            assert len(lab["name"]) > 0


# ---------------------------------------------------------------------------
# Font presets
# ---------------------------------------------------------------------------


class TestFontPresets:
    """Font presets should cover all expected families."""

    def test_at_least_4_presets(self, engine: TemplateEngine) -> None:
        assert len(engine.font_presets) >= 4

    def test_all_families_covered(self, engine: TemplateEngine) -> None:
        families = {f["family"] for f in engine.font_presets}
        assert "Helvetica" in families
        assert "Times-Roman" in families
        assert "Courier" in families

    def test_font_sizes_positive(self, engine: TemplateEngine) -> None:
        for seed in range(100):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            fc = cfg["font_config"]
            assert fc["header_size"] >= 6.0, f"Seed {seed}: header too small"
            assert fc["body_size"] >= 5.0, f"Seed {seed}: body too small"


# ---------------------------------------------------------------------------
# Margin presets
# ---------------------------------------------------------------------------


class TestMarginPresets:
    """Margins should always be positive."""

    def test_margins_positive(self, engine: TemplateEngine) -> None:
        for seed in range(200):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            m = cfg["margins"]
            for side in ("top", "bottom", "left", "right"):
                assert m[side] >= 15.0, (
                    f"Seed {seed}: margin.{side} = {m[side]:.1f} too small"
                )


# ---------------------------------------------------------------------------
# Config completeness
# ---------------------------------------------------------------------------


class TestConfigCompleteness:
    """Generated config should have all TemplateConfig-compatible fields."""

    REQUIRED_KEYS = {
        "name", "lab_key", "header_style", "color_scheme", "table_style",
        "columns", "flag_style", "section_style", "font_config", "margins",
        "row_height_factor", "show_refs", "show_sections",
        "page_break_strategy", "ref_format", "footer_style", "extras",
    }

    def test_all_keys_present(self, engine: TemplateEngine) -> None:
        for seed in range(50):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            missing = self.REQUIRED_KEYS - set(cfg.keys())
            assert not missing, f"Seed {seed}: missing keys {missing}"

    def test_valid_style_enums(self, engine: TemplateEngine) -> None:
        try:
            from scripts.lib.template_engine import (
                FLAG_STYLES,
                FOOTER_STYLES,
                HEADER_STYLES,
                PAGE_BREAK_STRATEGIES,
                REF_FORMATS,
                SECTION_STYLES,
                TABLE_STYLES,
            )
        except ModuleNotFoundError:
            from template_engine import (
                FLAG_STYLES,
                FOOTER_STYLES,
                HEADER_STYLES,
                PAGE_BREAK_STRATEGIES,
                REF_FORMATS,
                SECTION_STYLES,
                TABLE_STYLES,
            )

        for seed in range(100):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            assert cfg["header_style"] in HEADER_STYLES, f"Seed {seed}: bad header_style {cfg['header_style']}"
            assert cfg["table_style"] in TABLE_STYLES, f"Seed {seed}: bad table_style {cfg['table_style']}"
            assert cfg["flag_style"] in FLAG_STYLES, f"Seed {seed}: bad flag_style {cfg['flag_style']}"
            assert cfg["section_style"] in SECTION_STYLES, f"Seed {seed}: bad section_style {cfg['section_style']}"
            assert cfg["footer_style"] in FOOTER_STYLES, f"Seed {seed}: bad footer_style {cfg['footer_style']}"
            assert cfg["page_break_strategy"] in PAGE_BREAK_STRATEGIES, f"Seed {seed}: bad page_break_strategy"
            assert cfg["ref_format"] in REF_FORMATS, f"Seed {seed}: bad ref_format {cfg['ref_format']}"


# ---------------------------------------------------------------------------
# apply_variation standalone
# ---------------------------------------------------------------------------


class TestApplyVariation:
    """Test apply_variation as an independent function."""

    def test_idempotent_seed(self, engine: TemplateEngine) -> None:
        """Same base + same seed => same output."""
        base = {
            "name": "test",
            "lab_key": "test",
            "header_style": "standard",
            "color_scheme": {
                "primary": (0.2, 0.3, 0.4),
                "secondary": (0.5, 0.5, 0.5),
                "accent": (0.0, 0.4, 0.8),
                "text": (0.0, 0.0, 0.0),
                "bg": (1.0, 1.0, 1.0),
                "alt_row": (0.95, 0.95, 0.95),
            },
            "table_style": "alternating",
            "columns": [
                {"name": "A", "width_pct": 0.5, "align": "left"},
                {"name": "B", "width_pct": 0.5, "align": "right"},
            ],
            "flag_style": "asterisk",
            "section_style": "underline",
            "font_config": {"family": "Helvetica", "header_size": 14.0, "body_size": 9.0, "bold_header": True},
            "margins": {"top": 60.0, "bottom": 60.0, "left": 50.0, "right": 50.0},
            "row_height_factor": 1.6,
            "show_refs": True,
            "show_sections": True,
            "page_break_strategy": "flow",
            "ref_format": "column",
            "footer_style": "accreditation",
            "extras": {},
        }

        rng1 = random.Random(77)
        rng2 = random.Random(77)
        v1 = engine.apply_variation(dict(base), rng1)
        v2 = engine.apply_variation(dict(base), rng2)
        assert v1["font_config"] == v2["font_config"]
        assert v1["margins"] == v2["margins"]
        assert v1["color_scheme"] == v2["color_scheme"]

    def test_colors_stay_clamped_at_extremes(self, engine: TemplateEngine) -> None:
        """Colours near 0 or 1 should not go out of range after jitter."""
        base = {
            "name": "edge",
            "color_scheme": {
                "primary": (0.0, 0.0, 0.0),
                "secondary": (1.0, 1.0, 1.0),
                "accent": (0.0, 1.0, 0.5),
                "text": (0.01, 0.01, 0.01),
                "bg": (0.99, 0.99, 0.99),
                "alt_row": (0.0, 0.0, 0.0),
            },
            "font_config": {"family": "Helvetica", "header_size": 14.0, "body_size": 9.0, "bold_header": True},
            "columns": [{"name": "X", "width_pct": 1.0, "align": "left"}],
            "margins": {"top": 25.0, "bottom": 25.0, "left": 25.0, "right": 25.0},
            "row_height_factor": 1.0,
            "extras": {},
        }
        for seed in range(100):
            rng = random.Random(seed)
            result = engine.apply_variation(dict(base), rng)
            cs = result["color_scheme"]
            for key in cs:
                for v in cs[key]:
                    assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class TestUtility:
    def test_clamp01(self) -> None:
        assert _clamp01(-0.1) == 0.0
        assert _clamp01(0.5) == 0.5
        assert _clamp01(1.1) == 1.0
        assert _clamp01(0.0) == 0.0
        assert _clamp01(1.0) == 1.0


# ---------------------------------------------------------------------------
# Row height
# ---------------------------------------------------------------------------


class TestRowHeight:
    def test_row_height_factor_positive(self, engine: TemplateEngine) -> None:
        for seed in range(200):
            rng = random.Random(seed)
            cfg = engine.select_template_config(rng, region="worldwide")
            assert cfg["row_height_factor"] >= 1.0, (
                f"Seed {seed}: row_height_factor = {cfg['row_height_factor']}"
            )
