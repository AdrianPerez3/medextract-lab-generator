"""Stochastic template variation engine for synthetic lab report generation.

Provides a parametric variation layer on top of the 70 existing ReportLab
templates in ``generate_30k_synthetic.py``.  Instead of replacing the existing
rendering code, this module generates ``TemplateConfig``-compatible dicts with
stochastic jitter applied to fonts, colours, margins, column widths, spacing
and row height.  The combinatorial explosion of 17+ country presets, 20+ colour
schemes, 30+ lab institutions, 4+ font presets, 4+ margin presets and continuous
stochastic jitter produces millions of unique visual combinations.

Usage
-----
    from scripts.lib.template_engine import TemplateEngine

    engine = TemplateEngine()
    rng = random.Random(42)
    cfg = engine.select_template_config(rng, region="spain")
    # cfg is a dict ready to be unpacked into TemplateConfig(...)

All public state is deterministic given the same ``random.Random`` instance,
so output is reproducible across runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Stochastic variation descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StochasticVariation:
    """Random variation axes applied on top of a base template.

    Each field specifies an additive or multiplicative jitter that the engine
    applies after choosing a base configuration.
    """

    font_size_delta: float
    """Additive delta (pts) applied to both header_size and body_size."""

    column_width_jitter: float
    """Max additive jitter (fraction) applied to each column's width_pct."""

    row_height_jitter: float
    """Additive delta applied to row_height_factor."""

    margin_jitter: tuple[float, float, float, float]
    """Per-margin additive jitter (top, bottom, left, right) in points."""

    color_variation: tuple[float, float, float]
    """Per-channel additive jitter applied to every colour in the scheme."""

    spacing_factor: float
    """Multiplicative factor (0.9 - 1.1) for vertical spacing."""


# ---------------------------------------------------------------------------
# Column definition helpers
# ---------------------------------------------------------------------------

def _cols_3() -> list[dict]:
    return [
        {"name": "Analisis", "width_pct": 0.45, "align": "left"},
        {"name": "Resultado", "width_pct": 0.25, "align": "right"},
        {"name": "Unidades", "width_pct": 0.30, "align": "left"},
    ]


def _cols_4() -> list[dict]:
    return [
        {"name": "Prueba", "width_pct": 0.38, "align": "left"},
        {"name": "Resultado", "width_pct": 0.18, "align": "right"},
        {"name": "Unidades", "width_pct": 0.16, "align": "left"},
        {"name": "Val. Ref.", "width_pct": 0.28, "align": "left"},
    ]


def _cols_5() -> list[dict]:
    return [
        {"name": "Prueba", "width_pct": 0.34, "align": "left"},
        {"name": "Resultado", "width_pct": 0.16, "align": "right"},
        {"name": "Unid.", "width_pct": 0.13, "align": "left"},
        {"name": "Val. Ref.", "width_pct": 0.24, "align": "left"},
        {"name": "Flag", "width_pct": 0.07, "align": "center"},
    ]


def _cols_6() -> list[dict]:
    return [
        {"name": "Prueba", "width_pct": 0.28, "align": "left"},
        {"name": "Result.", "width_pct": 0.13, "align": "right"},
        {"name": "Unid.", "width_pct": 0.11, "align": "left"},
        {"name": "Ref.", "width_pct": 0.20, "align": "left"},
        {"name": "Flag", "width_pct": 0.06, "align": "center"},
        {"name": "Metodo", "width_pct": 0.16, "align": "left"},
    ]


_COLUMN_FACTORIES: dict[int, Any] = {
    3: _cols_3,
    4: _cols_4,
    5: _cols_5,
    6: _cols_6,
}

# ---------------------------------------------------------------------------
# Valid enum values -- must match generate_30k_synthetic rendering logic
# ---------------------------------------------------------------------------

HEADER_STYLES = [
    "standard", "logo_left", "minimal", "banner",
    "dual_block", "centered_formal",
]

TABLE_STYLES = [
    "alternating", "bordered", "grid", "plain", "striped",
    "fixed_width", "cards", "form_fields",
]

FLAG_STYLES = [
    "asterisk", "bold_red", "colored_column", "hl_text",
    "hl_highlight", "traffic_light", "arrow", "background", "none",
]

SECTION_STYLES = [
    "underline", "colored_bar", "box", "rounded_box",
    "discipline_header", "side_tab",
]

FOOTER_STYLES = ["accreditation", "simple_page"]

PAGE_BREAK_STRATEGIES = ["flow", "every_n"]

REF_FORMATS = ["column", "inline"]

FONT_FAMILIES = ["Helvetica", "Times-Roman", "Courier"]


# ---------------------------------------------------------------------------
# Template Engine
# ---------------------------------------------------------------------------

# Fictional sites are used by both the report and layout selectors.
SYNTHETIC_LABS = [
    {
        "key": f"demo_{region}_{i:02d}",
        "name": f"Laboratorio de Simulación {location} {i:02d}",
        "dept": "Laboratorio de demostración",
        "addr": f"Sede ficticia {i:02d}, {location}",
        "phone": "",
        "region": region,
    }
    for region, location, count in (
        ("spain", "España", 10), ("mexico", "México", 6),
        ("argentina", "Argentina", 4), ("colombia", "Colombia", 4),
        ("chile", "Chile", 4), ("peru", "Perú", 4),
    )
    for i in range(1, count + 1)
]


class TemplateEngine:
    """Provides stochastic template selection and parametric variation.

    The engine combines discrete choices (country preset, colour scheme, font
    preset, margin preset, lab institution) with continuous stochastic jitter
    to produce an effectively unlimited number of unique template configs.
    """

    def __init__(self) -> None:
        self.color_schemes: list[dict[str, Any]] = self._build_color_schemes()
        self.country_presets: dict[str, dict[str, Any]] = self._build_country_presets()
        self.lab_info: list[dict[str, str]] = self._build_lab_info()
        self.font_presets: list[dict[str, Any]] = self._build_font_presets()
        self.margin_presets: list[dict[str, Any]] = self._build_margin_presets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_template_config(
        self,
        rng: random.Random,
        region: str = "spain",
    ) -> dict[str, Any]:
        """Select a full template configuration with stochastic variation.

        Parameters
        ----------
        rng:
            Seeded random instance -- makes output deterministic per seed.
        region:
            Bias country-preset selection towards this region.  Accepts
            ``"spain"``, ``"latam"``, ``"europe"``, ``"worldwide"`` or a
            specific country-preset key.

        Returns
        -------
        dict
            All fields required by ``TemplateConfig`` plus a ``"variation"``
            key containing the applied ``StochasticVariation``.  Column dicts
            are under ``"columns"`` as a tuple of dicts (name/width_pct/align).
        """
        # 1. Pick country preset (weighted by region)
        preset_key = self._pick_country_preset(rng, region)
        preset = self.country_presets[preset_key]

        # 2. Pick colour scheme
        scheme = rng.choice(self.color_schemes)

        # 3. Pick font config
        font = self._pick_font(rng, preset)

        # 4. Pick margins
        margin = rng.choice(self.margin_presets)

        # 5. Pick lab institution
        lab = rng.choice(self.lab_info)

        # 6. Build base config dict
        n_cols: int = preset.get("columns", 5)
        col_factory = _COLUMN_FACTORIES.get(n_cols, _cols_5)
        columns = col_factory()

        base: dict[str, Any] = {
            "name": f"{preset_key}_{scheme['name']}_{lab['key']}",
            "lab_key": lab["key"],
            "header_style": preset["header_style"],
            "color_scheme": {
                "primary": scheme["primary"],
                "secondary": scheme["secondary"],
                "accent": scheme["accent"],
                "text": scheme["text"],
                "bg": scheme["bg"],
                "alt_row": scheme["alt_row"],
            },
            "table_style": preset["table_style"],
            "columns": columns,
            "flag_style": preset["flag_style"],
            "section_style": preset["section_style"],
            "font_config": {
                "family": font["family"],
                "header_size": font["header_size"],
                "body_size": font["body_size"],
                "bold_header": font["bold_header"],
            },
            "margins": {
                "top": margin["top"],
                "bottom": margin["bottom"],
                "left": margin["left"],
                "right": margin["right"],
            },
            "row_height_factor": rng.uniform(1.3, 2.0),
            "show_refs": rng.random() < 0.92,
            "show_sections": rng.random() < 0.88,
            "page_break_strategy": rng.choice(PAGE_BREAK_STRATEGIES),
            "ref_format": preset.get("ref_format", "column"),
            "footer_style": rng.choice(FOOTER_STYLES),
            "extras": dict(preset.get("extras", {})),
        }

        # 7. Apply stochastic variation
        varied = self.apply_variation(base, rng)
        return varied

    def apply_variation(
        self,
        base_config: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        """Apply stochastic variation to a base template config.

        Randomises font sizes (+/-2 pt), column widths (+/-5 %), row height
        (+/-0.2), margins (+/-10 pt), colours (+/-0.03 per channel) and
        vertical spacing (+/-10 %).
        """
        cfg = dict(base_config)

        # -- Generate variation descriptor --
        variation = StochasticVariation(
            font_size_delta=rng.uniform(-2.0, 2.0),
            column_width_jitter=rng.uniform(0.02, 0.05),
            row_height_jitter=rng.uniform(-0.2, 0.2),
            margin_jitter=(
                rng.uniform(-10.0, 10.0),
                rng.uniform(-10.0, 10.0),
                rng.uniform(-10.0, 10.0),
                rng.uniform(-10.0, 10.0),
            ),
            color_variation=(
                rng.uniform(-0.03, 0.03),
                rng.uniform(-0.03, 0.03),
                rng.uniform(-0.03, 0.03),
            ),
            spacing_factor=rng.uniform(0.9, 1.1),
        )
        cfg["variation"] = variation

        # -- Font sizes --
        fc = dict(cfg["font_config"])
        fc["header_size"] = max(8.0, fc["header_size"] + variation.font_size_delta)
        fc["body_size"] = max(6.0, fc["body_size"] + variation.font_size_delta * 0.6)
        cfg["font_config"] = fc

        # -- Column widths --
        columns: list[dict[str, Any]] = [dict(c) for c in cfg["columns"]]
        for col in columns:
            jitter = rng.uniform(
                -variation.column_width_jitter,
                variation.column_width_jitter,
            )
            col["width_pct"] = max(0.04, col["width_pct"] + jitter)
        # Renormalise to sum to 1.0
        total_w = sum(c["width_pct"] for c in columns)
        if total_w > 0:
            for col in columns:
                col["width_pct"] = col["width_pct"] / total_w
        cfg["columns"] = tuple(columns)

        # -- Row height --
        cfg["row_height_factor"] = max(
            1.0,
            cfg["row_height_factor"] + variation.row_height_jitter,
        )

        # -- Margins --
        margins = dict(cfg["margins"])
        for key, jitter in zip(
            ("top", "bottom", "left", "right"),
            variation.margin_jitter,
        ):
            margins[key] = max(20.0, margins[key] + jitter)
        cfg["margins"] = margins

        # -- Colours --
        cs = dict(cfg["color_scheme"])
        cv = variation.color_variation
        for color_key in ("primary", "secondary", "accent", "text", "bg", "alt_row"):
            r, g, b = cs[color_key]
            cs[color_key] = (
                _clamp01(r + cv[0]),
                _clamp01(g + cv[1]),
                _clamp01(b + cv[2]),
            )
        cfg["color_scheme"] = cs

        # -- Spacing factor (stored in extras for the renderer) --
        extras = dict(cfg.get("extras", {}))
        extras["spacing_factor"] = variation.spacing_factor
        cfg["extras"] = extras

        return cfg

    # ------------------------------------------------------------------
    # Preset builders
    # ------------------------------------------------------------------

    def _build_color_schemes(self) -> list[dict[str, Any]]:
        """22 colour scheme presets spanning clinical, institutional and modern styles."""
        return [
            {"name": "clinical_blue", "primary": (0.13, 0.27, 0.53), "secondary": (0.4, 0.5, 0.65), "accent": (0.0, 0.44, 0.75), "text": (0.1, 0.1, 0.1), "bg": (1.0, 1.0, 1.0), "alt_row": (0.94, 0.96, 0.98)},
            {"name": "nhs_green", "primary": (0.0, 0.36, 0.28), "secondary": (0.3, 0.5, 0.4), "accent": (0.0, 0.53, 0.33), "text": (0.15, 0.15, 0.15), "bg": (1.0, 1.0, 1.0), "alt_row": (0.93, 0.97, 0.94)},
            {"name": "warm_earth", "primary": (0.45, 0.25, 0.1), "secondary": (0.6, 0.45, 0.3), "accent": (0.7, 0.35, 0.1), "text": (0.2, 0.15, 0.1), "bg": (1.0, 0.99, 0.97), "alt_row": (0.97, 0.95, 0.93)},
            {"name": "stark_bw", "primary": (0.0, 0.0, 0.0), "secondary": (0.4, 0.4, 0.4), "accent": (0.2, 0.2, 0.2), "text": (0.0, 0.0, 0.0), "bg": (1.0, 1.0, 1.0), "alt_row": (0.95, 0.95, 0.95)},
            {"name": "medical_red", "primary": (0.6, 0.1, 0.1), "secondary": (0.7, 0.3, 0.3), "accent": (0.8, 0.0, 0.0), "text": (0.1, 0.1, 0.1), "bg": (1.0, 1.0, 1.0), "alt_row": (0.98, 0.95, 0.95)},
            {"name": "teal_modern", "primary": (0.0, 0.4, 0.4), "secondary": (0.3, 0.5, 0.5), "accent": (0.0, 0.6, 0.6), "text": (0.1, 0.1, 0.1), "bg": (1.0, 1.0, 1.0), "alt_row": (0.93, 0.97, 0.97)},
            {"name": "purple_lab", "primary": (0.3, 0.1, 0.4), "secondary": (0.5, 0.3, 0.6), "accent": (0.45, 0.15, 0.55), "text": (0.15, 0.1, 0.2), "bg": (1.0, 1.0, 1.0), "alt_row": (0.96, 0.94, 0.97)},
            {"name": "dark_professional", "primary": (0.15, 0.15, 0.2), "secondary": (0.35, 0.35, 0.4), "accent": (0.25, 0.4, 0.6), "text": (0.1, 0.1, 0.1), "bg": (0.98, 0.98, 0.99), "alt_row": (0.94, 0.94, 0.96)},
            {"name": "sage_green", "primary": (0.25, 0.35, 0.25), "secondary": (0.45, 0.55, 0.45), "accent": (0.3, 0.5, 0.3), "text": (0.1, 0.1, 0.1), "bg": (1.0, 1.0, 1.0), "alt_row": (0.95, 0.97, 0.95)},
            {"name": "ocean_blue", "primary": (0.05, 0.2, 0.45), "secondary": (0.2, 0.4, 0.6), "accent": (0.1, 0.5, 0.8), "text": (0.05, 0.1, 0.2), "bg": (1.0, 1.0, 1.0), "alt_row": (0.93, 0.96, 0.99)},
            {"name": "imss_green", "primary": (0.0, 0.4, 0.2), "secondary": (0.2, 0.5, 0.3), "accent": (0.0, 0.55, 0.25), "text": (0.1, 0.1, 0.1), "bg": (1.0, 1.0, 1.0), "alt_row": (0.93, 0.97, 0.94)},
            {"name": "sermas_blue", "primary": (0.0, 0.25, 0.55), "secondary": (0.3, 0.45, 0.65), "accent": (0.0, 0.35, 0.7), "text": (0.1, 0.1, 0.1), "bg": (1.0, 1.0, 1.0), "alt_row": (0.94, 0.96, 0.98)},
            {"name": "essalud_orange", "primary": (0.7, 0.35, 0.0), "secondary": (0.8, 0.5, 0.2), "accent": (0.85, 0.4, 0.0), "text": (0.15, 0.1, 0.05), "bg": (1.0, 1.0, 1.0), "alt_row": (0.98, 0.96, 0.93)},
            {"name": "fonasa_blue", "primary": (0.1, 0.3, 0.6), "secondary": (0.3, 0.5, 0.7), "accent": (0.15, 0.4, 0.7), "text": (0.1, 0.1, 0.15), "bg": (1.0, 1.0, 1.0), "alt_row": (0.94, 0.96, 0.98)},
            {"name": "cream_warm", "primary": (0.5, 0.35, 0.2), "secondary": (0.65, 0.5, 0.35), "accent": (0.6, 0.4, 0.15), "text": (0.2, 0.15, 0.1), "bg": (1.0, 0.99, 0.96), "alt_row": (0.97, 0.95, 0.92)},
            {"name": "minimal_grey", "primary": (0.3, 0.3, 0.3), "secondary": (0.5, 0.5, 0.5), "accent": (0.4, 0.4, 0.4), "text": (0.15, 0.15, 0.15), "bg": (1.0, 1.0, 1.0), "alt_row": (0.96, 0.96, 0.96)},
            {"name": "royal_navy", "primary": (0.05, 0.1, 0.3), "secondary": (0.2, 0.25, 0.45), "accent": (0.1, 0.15, 0.5), "text": (0.05, 0.05, 0.15), "bg": (1.0, 1.0, 1.0), "alt_row": (0.94, 0.95, 0.97)},
            {"name": "rose_medical", "primary": (0.55, 0.15, 0.25), "secondary": (0.7, 0.35, 0.4), "accent": (0.65, 0.2, 0.3), "text": (0.15, 0.1, 0.1), "bg": (1.0, 1.0, 1.0), "alt_row": (0.98, 0.95, 0.96)},
            {"name": "olive_classic", "primary": (0.3, 0.35, 0.15), "secondary": (0.5, 0.55, 0.35), "accent": (0.4, 0.45, 0.2), "text": (0.15, 0.15, 0.1), "bg": (1.0, 1.0, 0.99), "alt_row": (0.96, 0.97, 0.94)},
            {"name": "slate_modern", "primary": (0.2, 0.25, 0.3), "secondary": (0.4, 0.45, 0.5), "accent": (0.3, 0.5, 0.65), "text": (0.1, 0.1, 0.15), "bg": (0.99, 0.99, 1.0), "alt_row": (0.95, 0.95, 0.97)},
            # 2 extra for variety
            {"name": "sunflower_gold", "primary": (0.65, 0.5, 0.05), "secondary": (0.75, 0.65, 0.25), "accent": (0.8, 0.6, 0.0), "text": (0.15, 0.12, 0.05), "bg": (1.0, 1.0, 0.98), "alt_row": (0.98, 0.97, 0.93)},
            {"name": "arctic_ice", "primary": (0.15, 0.35, 0.5), "secondary": (0.35, 0.55, 0.7), "accent": (0.2, 0.55, 0.75), "text": (0.08, 0.1, 0.15), "bg": (0.98, 0.99, 1.0), "alt_row": (0.93, 0.96, 0.99)},
        ]

    def _build_country_presets(self) -> dict[str, dict[str, Any]]:
        """17 country/institution presets with layout preferences."""
        return {
            "spain_public": {
                "header_style": "standard",
                "table_style": "alternating",
                "flag_style": "hl_text",
                "section_style": "colored_bar",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {},
            },
            "spain_private": {
                "header_style": "logo_left",
                "table_style": "bordered",
                "flag_style": "bold_red",
                "section_style": "underline",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {},
            },
            "mexico_imss": {
                "header_style": "standard",
                "table_style": "grid",
                "flag_style": "asterisk",
                "section_style": "colored_bar",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {},
            },
            "mexico_private": {
                "header_style": "logo_left",
                "table_style": "alternating",
                "flag_style": "hl_text",
                "section_style": "box",
                "font_family": "Helvetica",
                "columns": 4,
                "ref_format": "inline",
                "extras": {},
            },
            "argentina_public": {
                "header_style": "standard",
                "table_style": "plain",
                "flag_style": "asterisk",
                "section_style": "underline",
                "font_family": "Times-Roman",
                "columns": 4,
                "ref_format": "column",
                "extras": {},
            },
            "argentina_private": {
                "header_style": "logo_left",
                "table_style": "alternating",
                "flag_style": "bold_red",
                "section_style": "colored_bar",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {},
            },
            "colombia_eps": {
                "header_style": "standard",
                "table_style": "bordered",
                "flag_style": "hl_text",
                "section_style": "box",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {},
            },
            "chile_fonasa": {
                "header_style": "standard",
                "table_style": "grid",
                "flag_style": "asterisk",
                "section_style": "underline",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {},
            },
            "peru_minsa": {
                "header_style": "standard",
                "table_style": "plain",
                "flag_style": "asterisk",
                "section_style": "underline",
                "font_family": "Courier",
                "columns": 4,
                "ref_format": "column",
                "extras": {},
            },
            "us_quest": {
                "header_style": "logo_left",
                "table_style": "alternating",
                "flag_style": "hl_text",
                "section_style": "discipline_header",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {"code_column": True},
            },
            "us_labcorp": {
                "header_style": "logo_left",
                "table_style": "alternating",
                "flag_style": "hl_highlight",
                "section_style": "colored_bar",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {},
            },
            "uk_nhs": {
                "header_style": "standard",
                "table_style": "grid",
                "flag_style": "hl_text",
                "section_style": "discipline_header",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {"code_column": True},
            },
            "de_din": {
                "header_style": "standard",
                "table_style": "fixed_width",
                "flag_style": "none",
                "section_style": "underline",
                "font_family": "Courier",
                "columns": 6,
                "ref_format": "column",
                "extras": {"code_column": True},
            },
            "fr_sil": {
                "header_style": "logo_left",
                "table_style": "bordered",
                "flag_style": "bold_red",
                "section_style": "rounded_box",
                "font_family": "Times-Roman",
                "columns": 5,
                "ref_format": "column",
                "extras": {"method_column": True},
            },
            "br_anvisa": {
                "header_style": "standard",
                "table_style": "bordered",
                "flag_style": "hl_text",
                "section_style": "colored_bar",
                "font_family": "Helvetica",
                "columns": 5,
                "ref_format": "column",
                "extras": {"code_column": True},
            },
            "digital_portal": {
                "header_style": "banner",
                "table_style": "cards",
                "flag_style": "traffic_light",
                "section_style": "rounded_box",
                "font_family": "Helvetica",
                "columns": 4,
                "ref_format": "inline",
                "extras": {"qr": True},
            },
            "minimal_iso": {
                "header_style": "standard",
                "table_style": "plain",
                "flag_style": "none",
                "section_style": "underline",
                "font_family": "Times-Roman",
                "columns": 4,
                "ref_format": "column",
                "extras": {},
            },
        }

    def _build_lab_info(self) -> list[dict[str, str]]:
        """Return fresh configuration dictionaries for the fictional sites."""
        return [lab.copy() for lab in SYNTHETIC_LABS]

    def _build_font_presets(self) -> list[dict[str, Any]]:
        """Font configuration presets (family + base sizes)."""
        return [
            {"family": "Helvetica", "header_size": 14.0, "body_size": 9.0, "bold_header": True, "tag": "standard"},
            {"family": "Helvetica", "header_size": 12.0, "body_size": 8.0, "bold_header": True, "tag": "small"},
            {"family": "Helvetica", "header_size": 16.0, "body_size": 10.0, "bold_header": True, "tag": "large"},
            {"family": "Helvetica", "header_size": 11.0, "body_size": 7.0, "bold_header": True, "tag": "tiny"},
            {"family": "Times-Roman", "header_size": 14.0, "body_size": 9.0, "bold_header": True, "tag": "serif_standard"},
            {"family": "Times-Roman", "header_size": 12.0, "body_size": 8.0, "bold_header": True, "tag": "serif_small"},
            {"family": "Courier", "header_size": 12.0, "body_size": 8.0, "bold_header": True, "tag": "mono_standard"},
            {"family": "Courier", "header_size": 11.0, "body_size": 7.5, "bold_header": True, "tag": "mono_small"},
        ]

    def _build_margin_presets(self) -> list[dict[str, Any]]:
        """Margin presets in points."""
        return [
            {"top": 60.0, "bottom": 60.0, "left": 50.0, "right": 50.0, "tag": "standard"},
            {"top": 50.0, "bottom": 50.0, "left": 40.0, "right": 40.0, "tag": "narrow"},
            {"top": 70.0, "bottom": 70.0, "left": 60.0, "right": 60.0, "tag": "wide"},
            {"top": 45.0, "bottom": 45.0, "left": 35.0, "right": 35.0, "tag": "tight"},
        ]

    # ------------------------------------------------------------------
    # Internal selection helpers
    # ------------------------------------------------------------------

    def _pick_country_preset(
        self,
        rng: random.Random,
        region: str,
    ) -> str:
        """Select a country preset key, weighted by region affinity."""
        all_keys = list(self.country_presets.keys())

        # If caller specified an exact preset key, honour it
        if region in self.country_presets:
            return region

        # Build weight vector based on region affinity
        region_weights = _REGION_AFFINITY.get(region, _REGION_AFFINITY["worldwide"])
        weights: list[float] = []
        for key in all_keys:
            w = 1.0
            for prefix, mult in region_weights:
                if key.startswith(prefix):
                    w = mult
                    break
            weights.append(w)

        total = sum(weights)
        weights = [w / total for w in weights]

        chosen = rng.choices(all_keys, weights=weights, k=1)[0]
        return chosen

    def _pick_font(
        self,
        rng: random.Random,
        preset: dict[str, Any],
    ) -> dict[str, Any]:
        """Pick a font preset, preferring the family from the country preset."""
        preferred_family = preset.get("font_family", "Helvetica")
        matching = [f for f in self.font_presets if f["family"] == preferred_family]
        if matching and rng.random() < 0.7:
            # 70 % chance to respect the country preference
            return rng.choice(matching)
        return rng.choice(self.font_presets)


# ---------------------------------------------------------------------------
# Region affinity tables
# ---------------------------------------------------------------------------

_REGION_AFFINITY: dict[str, list[tuple[str, float]]] = {
    "spain": [
        ("spain_", 5.0),
        ("mexico_", 1.0),
        ("argentina_", 1.0),
        ("colombia_", 1.0),
        ("chile_", 1.0),
        ("peru_", 1.0),
    ],
    "latam": [
        ("mexico_", 4.0),
        ("argentina_", 3.0),
        ("colombia_", 3.0),
        ("chile_", 3.0),
        ("peru_", 3.0),
        ("spain_", 2.0),
        ("br_", 2.0),
    ],
    "europe": [
        ("spain_", 3.0),
        ("uk_", 3.0),
        ("de_", 3.0),
        ("fr_", 3.0),
    ],
    "worldwide": [
        # Uniform-ish weights
    ],
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    """Clamp a value to the [0.0, 1.0] range."""
    return max(0.0, min(1.0, v))
