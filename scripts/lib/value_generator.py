"""Generate synthetic observations for testing lab-document extraction.

Sampling envelopes, optional reference fields, and flags are simulator inputs.
European number formatting uses a decimal comma.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Optional

import random

from scripts.lib.terminology import admitted_range, load_simulation_policy, load_terms, placeholder_bounds, unit_spelling


# ---------------------------------------------------------------------------
# Clinical reference ranges for common LOINCs (ref_low / ref_high)
# These represent typical adult reference intervals used by Spanish labs.
# The reference_ranges.json only has plausibility min/max, so we maintain
# clinical normal ranges separately for the most common tests.
# ---------------------------------------------------------------------------
CLINICAL_RANGES: dict[str, dict] = {
    # Chemistry
    "2345-7":  {"ref_low": 70.0,  "ref_high": 110.0,  "unit": "mg/dL"},     # Glucose
    "2093-3":  {"ref_low": 100.0, "ref_high": 200.0,  "unit": "mg/dL"},     # Total Cholesterol
    "2571-8":  {"ref_low": 60.0,  "ref_high": 160.0,  "unit": "mg/dL"},     # Triglycerides
    "2085-9":  {"ref_low": 40.0,  "ref_high": 60.0,   "unit": "mg/dL"},     # HDL
    "13457-7": {"ref_low": 0.0,   "ref_high": 100.0,  "unit": "mg/dL"},     # LDL calc
    "1742-6":  {"ref_low": 7.0,   "ref_high": 56.0,   "unit": "U/L"},       # ALT
    "1920-8":  {"ref_low": 8.0,   "ref_high": 48.0,   "unit": "U/L"},       # AST
    "2324-2":  {"ref_low": 44.0,  "ref_high": 147.0,  "unit": "U/L"},       # GGT
    "6768-6":  {"ref_low": 44.0,  "ref_high": 147.0,  "unit": "U/L"},       # Alkaline Phosphatase
    "1975-2":  {"ref_low": 0.1,   "ref_high": 1.2,    "unit": "mg/dL"},     # Total Bilirubin
    "1968-7":  {"ref_low": 0.0,   "ref_high": 0.3,    "unit": "mg/dL"},     # Direct Bilirubin
    "2160-0":  {"ref_low": 0.7,   "ref_high": 1.3,    "unit": "mg/dL"},     # Creatinine
    "3094-0":  {"ref_low": 7.0,   "ref_high": 20.0,   "unit": "mg/dL"},     # BUN
    "2132-9":  {"ref_low": 200.0, "ref_high": 900.0,   "unit": "pg/mL"},     # Vitamin B12
    "2823-3":  {"ref_low": 3.5,   "ref_high": 5.1,    "unit": "mmol/L"},     # Potassium
    "2951-2":  {"ref_low": 136.0, "ref_high": 145.0,  "unit": "mmol/L"},    # Sodium
    "17861-6": {"ref_low": 8.5,   "ref_high": 10.5,   "unit": "mg/dL"},     # Calcium
    "19123-9":  {"ref_low": 1.7,   "ref_high": 2.2,    "unit": "mg/dL"},     # Magnesium
    "2777-1":  {"ref_low": 2.5,   "ref_high": 4.5,    "unit": "mg/dL"},     # Phosphorus
    "2276-4":  {"ref_low": 12.0,  "ref_high": 300.0,  "unit": "ng/mL"},     # Ferritin
    "2498-4":  {"ref_low": 33.0,  "ref_high": 193.0,  "unit": "ug/dL"},     # Iron
    "2500-7":  {"ref_low": 200.0, "ref_high": 360.0,  "unit": "ug/dL"},     # TIBC
    "17856-6": {"ref_low": 4.0,   "ref_high": 5.6,    "unit": "%"},         # HbA1c (LOINC 17856-6)
    "4548-4":  {"ref_low": 4.0,   "ref_high": 5.6,    "unit": "%"},         # HbA1c (LOINC 4548-4)
    "2885-2":  {"ref_low": 6.4,   "ref_high": 8.3,    "unit": "g/dL"},      # Total Protein
    "1751-7":  {"ref_low": 3.5,   "ref_high": 5.2,    "unit": "g/dL"},      # Albumin

    # Hormones
    "3016-3":  {"ref_low": 0.27,  "ref_high": 4.20,   "unit": "m[IU]/L"},     # TSH
    "3024-7":  {"ref_low": 0.93,  "ref_high": 1.70,   "unit": "ng/dL"},     # Free T4
    "3051-0":  {"ref_low": 2.0,   "ref_high": 4.4,    "unit": "pg/mL"},     # Free T3
    "3026-2":  {"ref_low": 4.6,   "ref_high": 12.0,   "unit": "ug/dL"},     # Total T4
    "3053-6": {"ref_low": 80.0,  "ref_high": 200.0,  "unit": "ng/dL"},     # Total T3
    "15067-2": {"ref_low": 1.5,   "ref_high": 9.0,    "unit": "m[IU]/mL"},    # FSH
    "2143-6":  {"ref_low": 5.0,   "ref_high": 25.0,   "unit": "ug/dL"},     # Cortisol

    # Hematology
    "718-7":   {"ref_low": 12.0,  "ref_high": 17.5,   "unit": "g/dL"},      # Hemoglobin
    "4544-3":  {"ref_low": 36.0,  "ref_high": 50.0,   "unit": "%"},         # Hematocrit
    "789-8":   {"ref_low": 4.0,   "ref_high": 5.5,    "unit": "10*6/uL"},   # RBC
    "787-2":   {"ref_low": 80.0,  "ref_high": 100.0,  "unit": "fL"},        # MCV
    "785-6":   {"ref_low": 27.0,  "ref_high": 33.0,   "unit": "pg"},        # MCH
    "786-4":   {"ref_low": 31.5,  "ref_high": 35.7,   "unit": "g/dL"},      # MCHC
    "788-0":   {"ref_low": 11.5,  "ref_high": 14.5,   "unit": "%"},         # RDW
    "777-3":   {"ref_low": 150.0, "ref_high": 400.0,  "unit": "10*3/uL"},   # Platelet count
    "32623-1": {"ref_low": 7.4,   "ref_high": 10.4,   "unit": "fL"},        # MPV
    "6690-2":  {"ref_low": 4.5,   "ref_high": 11.0,   "unit": "10*3/uL"},   # WBC

    # Coagulation
    "6301-6":  {"ref_low": 0.8,   "ref_high": 1.2,    "unit": "{INR}"},     # INR
    "5902-2":  {"ref_low": 11.0,  "ref_high": 13.5,   "unit": "s"},         # PT seconds
    "3173-2":  {"ref_low": 25.0,  "ref_high": 35.0,   "unit": "s"},         # aPTT seconds
    "3255-7":  {"ref_low": 200.0, "ref_high": 400.0,  "unit": "mg/dL"},     # Fibrinogen

    # Urinalysis
    "5811-5":  {"ref_low": 1.005, "ref_high": 1.030,  "unit": "{Spec grav}"}, # Specific gravity
    "5803-2":  {"ref_low": 4.5,   "ref_high": 8.0,    "unit": "[pH]"},      # pH urine

    # Inflammation
    "1988-5":  {"ref_low": 0.0,   "ref_high": 5.0,    "unit": "mg/L"},      # CRP
    "30341-2": {"ref_low": 0.0,   "ref_high": 20.0,   "unit": "mm/h"},      # ESR

    # Cardiac
    "2157-6":  {"ref_low": 30.0,  "ref_high": 200.0,  "unit": "U/L"},       # CK
    "13969-1": {"ref_low": 0.0,   "ref_high": 25.0,   "unit": "ng/mL"},     # CK-MB

    # Pancreatic
    "1798-8":  {"ref_low": 28.0,  "ref_high": 100.0,  "unit": "U/L"},       # Amylase
    "3040-3":  {"ref_low": 13.0,  "ref_high": 60.0,   "unit": "U/L"},       # Lipase

    # Electrophoresis
    "2458-8":  {"ref_low": 50.0,  "ref_high": 200.0,  "unit": "mg/dL"},     # IgA
    "2465-3":  {"ref_low": 700.0, "ref_high": 1600.0, "unit": "mg/dL"},     # IgG
    "2472-9":  {"ref_low": 40.0,  "ref_high": 230.0,  "unit": "mg/dL"},     # IgM
}

# Standard titre dilution series
TITRE_SERIES: list[str] = [
    "1:10", "1:20", "1:40", "1:80", "1:160",
    "1:320", "1:640", "1:1280", "1:2560", "1:5120", "1:10240",
]

# Spanish ordinal answers organized by meaning
ORDINAL_NORMAL: list[str] = [
    "Negativo", "No reactivo", "Normal", "No detectado",
    "Ausente", "Sin alteraciones",
]
ORDINAL_ABNORMAL: list[str] = [
    "Positivo", "Reactivo", "Anormal", "Detectado",
    "Presente", "Indeterminado",
]

@dataclass
class GeneratedValue:
    """A generated lab test value."""

    value: float | str
    value_formatted: str
    unit: str
    ref_low: Optional[float]
    ref_high: Optional[float]
    ref_formatted: str
    flag: Optional[str]
    is_numeric: bool
    scale_type: str


# Provenance tags from the project's historical range-building jobs. Such
# numeric envelopes are only suitable for explicitly requested stress fixtures.
PROJECT_RANGE_SOURCES = frozenset({
    "constants", "property_template", "default_fallback", "loinc_answers",
    "a6_universe_expansion_2026_04_20", "V4B-12 expansion 2026-04-20",
    "L2_FIX_ADD_2026-04-20", "V4B_ELECTROPHORESIS_ADD_2026-04-21",
    "v4b_audit3_addition", "L3+L6_audit_fix_2026-04-20",
    "L3+L6_audit_remap_2026-04-20", "w2_fix3_expansion",
})


# These plasma terms were active in the pinned 2.82 source. LOINC 2.83
# deprecated both as erroneous and changed the specimen in the replacements.
# Preserve historical lookup data; never generate or automatically recode them.
ERRONEOUS_PLASMA_TERMS = frozenset({"107370-9", "107371-7"})
# Revision 3 supplements the pinned 2.82 names and units with verified term
# status from LOINC 2.83 across the entire private simulation vocabulary.
NONACTIVE_283_ADDITIONS = frozenset({"1385-4", "19568-5", "42242-8", "42493-7"})
NONACTIVE_283_TERMS = ERRONEOUS_PLASMA_TERMS | NONACTIVE_283_ADDITIONS


# Untyped strips require a specific instrument's reading increments. The
# titre term has a published binary example incompatible with this simulator.
UNSUPPORTED_REPORTING_METHODS = frozenset({"5803-2", "5811-5", "71707-4"})


# The correlation engine publishes these five percentages to hundredths.
_AUTO_DIFFERENTIAL = ("770-8", "736-9", "5905-5", "713-8", "706-2")


def _value_decimal_places(value: float) -> int:
    magnitude = abs(value)
    if magnitude >= 1000:
        return 0
    if magnitude >= 10:
        return 1
    if magnitude >= 1:
        return 2
    if magnitude >= 0.01:
        return 3
    return 4


class ValueGenerator:
    """Generates simulated values for admitted LOINC observations.

    Loads reference ranges (plausibility bounds) and LOINC metadata to produce
    values appropriate for each test's scale type.  Supports quantitative (Qn),
    semi-quantitative/titre (SemiQn), and admitted binary ordinal (Ord) scales.
    """

    def __init__(self, ranges_path: str, metadata_path: str, *, include_heuristic_numerics: bool = True) -> None:
        """Load simulation ranges. The second argument preserves the historical API;
        names, scales and units now come exclusively from the pinned catalog.
        """
        with open(ranges_path, encoding="utf-8") as f:
            ranges = json.load(f)
        self._terms = load_terms()
        self._ranges = {}
        # Input ranges are fixed for this generator. Cache each numeric
        # interval and its printed precision separately for the two locales.
        self._numeric_references: dict[tuple[str, bool], tuple[float, float, str]] = {}
        binary_codes = load_simulation_policy()["binary_answer_lists"]
        for code, item in sorted(ranges.items()):
            term = self._terms.get(code)
            if not term or code in NONACTIVE_283_TERMS:
                continue
            if code in UNSUPPORTED_REPORTING_METHODS or (term["scale"] == "Ord" and code not in binary_codes):
                continue
            if code == "98733-9":
                # Neutralization inhibition requires assay-specific calibration.
                continue
            entry = admitted_range(term, item)
            curated = CLINICAL_RANGES.get(code)
            if curated and curated["unit"] in term["units"]:
                low, high = curated["ref_low"], curated["ref_high"]
                # These two legacy immunoglobulin records were populated from
                # different assays in the original bundled reference table.
                legacy_iga = (code == "2458-8" and item.get("unit") == "[IU]/mL"
                              and item.get("min") == 0 and item.get("max") == 10000)
                inconsistent = (entry is not None and entry["unit"] == curated["unit"]
                                and not entry["min"] <= low < high <= entry["max"])
                placeholder = (placeholder_bounds(item) and unit_spelling(item.get("unit") or "") == curated["unit"])
                if legacy_iga or inconsistent or placeholder:
                    # Synthesize fresh bounds in the curated unit. Never relabel
                    # a numeric interval from another scale or a custom unit.
                    width = high - low
                    entry = {"min": max(0, low - width), "max": high + width,
                             "unit": curated["unit"], "scale": term["scale"]}
            if entry is None:
                continue
            if (not include_heuristic_numerics and entry["unit"] and item.get("source") in PROJECT_RANGE_SOURCES
                    and (curated is None or entry["unit"] != curated["unit"])):
                continue
            self._ranges[code] = entry

    @property
    def loinc_codes(self) -> list[str]:
        return list(self._ranges)

    def has_loinc(self, loinc: str) -> bool:
        return loinc in self._ranges

    def get_scale(self, loinc: str) -> str:
        return self._ranges[loinc]["scale"]

    def get_unit(self, loinc: str) -> str:
        """Exact admitted UCUM token for a quantitative result."""
        return self._ranges[loinc]["unit"]

    def simulation_bounds(self, loinc: str) -> tuple[float, float]:
        entry = self._ranges[loinc]
        return entry["min"], entry["max"]

    def within_simulation_bounds(self, loinc: str, value: float) -> bool:
        bounds = self._ranges[loinc]
        return bounds["min"] <= value <= bounds["max"]

    def get_ref_range(self, loinc: str) -> tuple[Optional[float], Optional[float]]:
        """Return simulation thresholds for numeric sampling and optional layout examples.

        Uses CLINICAL_RANGES for known tests, otherwise derives from
        plausibility min/max with a 30%-70% heuristic.
        """
        if loinc in CLINICAL_RANGES and self.get_unit(loinc) == CLINICAL_RANGES[loinc]["unit"]:
            cr = CLINICAL_RANGES[loinc]
            return cr["ref_low"], cr["ref_high"]

        rr = self._ranges.get(loinc, {})
        mn = rr.get("min")
        mx = rr.get("max")
        if mn is not None and mx is not None:
            mn, mx = float(mn), float(mx)
            # Derive a synthetic reference range at 30%-70% of plausibility
            # BUG 3 fix: round to avoid floating-point artifacts like 14.899999999999999
            ref_low = round(mn + (mx - mn) * 0.30, 4)
            ref_high = round(mn + (mx - mn) * 0.70, 4)
            return ref_low, ref_high
        return None, None

    def _numeric_reference(self, loinc: str, european: bool) -> tuple[float, float, str]:
        key = (loinc, european)
        reference = self._numeric_references.get(key)
        if reference is None:
            low, high = self.get_ref_range(loinc)
            if low is None or high is None:
                minimum, maximum = self.simulation_bounds(loinc)
                if low is None:
                    low = minimum + (maximum - minimum) * 0.30
                if high is None:
                    high = minimum + (maximum - minimum) * 0.70
            reference = (low, high, self._format_ref_range(low, high, european))
            self._numeric_references[key] = reference
        return reference

    # ------------------------------------------------------------------
    # Primary generate method
    # ------------------------------------------------------------------

    def generate(
        self,
        loinc: str,
        rng: random.Random,
        abnormal_multiplier: Optional[tuple[float, float]] = None,
        european_format: bool = True,
    ) -> GeneratedValue:
        """Generate a value for the given LOINC code.

        Args:
            loinc: LOINC code (e.g., "2345-7").
            rng: Seeded Random instance for reproducibility.
            abnormal_multiplier: If set, multiply the midpoint by
                ``uniform(low, high)`` to produce pathological values.
                E.g. ``(1.5, 3.0)`` for high abnormal.
            european_format: Use comma as decimal separator when True.

        Returns:
            GeneratedValue with all fields populated.

        Raises:
            KeyError: If the LOINC code is not found in reference ranges.
        """
        if not self.has_loinc(loinc):
            raise KeyError(f"Unknown LOINC code: {loinc}")

        scale = self.get_scale(loinc)

        if scale == "Ord":
            return self._generate_ordinal(loinc, rng, european_format)

        if scale == "SemiQn":
            return self._generate_semiquantitative(
                loinc, rng, abnormal_multiplier, european_format,
            )

        return self._generate_quantitative(
            loinc, rng, abnormal_multiplier, european_format,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def generate_normal(
        self,
        loinc: str,
        rng: random.Random,
        european_format: bool = True,
    ) -> GeneratedValue:
        """Generate a value within the normal reference range.

        Uses a truncated Gaussian centered on the midpoint of the clinical
        reference range with sigma = range_width / 4.
        """
        return self.generate(loinc, rng, abnormal_multiplier=None,
                             european_format=european_format)

    def generate_abnormal(
        self,
        loinc: str,
        rng: random.Random,
        direction: str = "high",
        severity: float = 1.5,
        european_format: bool = True,
    ) -> GeneratedValue:
        """Generate an abnormal value above or below the reference range.

        Args:
            direction: ``"high"`` or ``"low"``.
            severity: Multiplier applied to the midpoint.  Values >1.0
                produce high results; values <1.0 produce low results.
        """
        if direction == "low":
            mult = (0.1, min(0.9, 1.0 / severity))
        else:
            mult = (severity, severity * 1.5)
        return self.generate(loinc, rng, abnormal_multiplier=mult,
                             european_format=european_format)

    def generate_critical(
        self,
        loinc: str,
        rng: random.Random,
        european_format: bool = True,
    ) -> GeneratedValue:
        """Generate a critical/extreme value near plausibility bounds.

        Picks a value in the outer 10% of the plausibility range (either
        the very low or very high end).
        """
        rr = self._ranges.get(loinc, {})
        scale = self.get_scale(loinc)

        if scale == "Ord":
            return self._generate_ordinal(loinc, rng, european_format, force_abnormal=True)
        if scale == "SemiQn" and self._terms[loinc]["property"] == "Titr":
            return self._generate_semiquantitative(loinc, rng, (2.0, 2.0), european_format)

        mn = float(rr.get("min", 0))
        mx = float(rr.get("max", 10000))

        ref_low, ref_high, ref_fmt = self._numeric_reference(loinc, european_format)

        # Pick extreme end — always outside the reference range
        if rng.random() < 0.5:
            # Critical low: between plausibility min and ref_low
            low_ceiling = ref_low * 0.5 if ref_low > 0 else mn + (mx - mn) * 0.02
            low_ceiling = max(mn, min(low_ceiling, ref_low))
            value = rng.uniform(mn, low_ceiling) if mn < low_ceiling else mn
        else:
            # Critical high: between ref_high and plausibility max
            high_floor = ref_high + (mx - ref_high) * 0.7
            high_floor = min(mx, max(high_floor, ref_high))
            value = rng.uniform(high_floor, mx) if high_floor < mx else mx

        # BUG 1 fix: clamp to non-negative for tests where plausibility min >= 0
        # (e.g. hemoglobin, WBC, glucose should never be negative)
        if mn >= 0 and value < 0:
            value = mn * 0.01 if mn > 0 else 0.01

        unit = self.get_unit(loinc)
        rounded = self._round_value(value)
        if mn <= rounded <= mx:
            formatted = self._format_number(value, european_format)
        else:
            value, formatted = self._fit_display_to_bounds(loinc, rounded, mn, mx, unit, european_format)
        flag = self._compute_flag(value, ref_low, ref_high)

        return GeneratedValue(
            value=value,
            value_formatted=formatted,
            unit=unit,
            ref_low=ref_low,
            ref_high=ref_high,
            ref_formatted=ref_fmt,
            flag=flag,
            is_numeric=True,
            scale_type=scale,
        )

    # ------------------------------------------------------------------
    # Internal: Quantitative (Qn)
    # ------------------------------------------------------------------

    def _generate_quantitative(
        self,
        loinc: str,
        rng: random.Random,
        abnormal_multiplier: Optional[tuple[float, float]],
        european_format: bool,
    ) -> GeneratedValue:
        rr = self._ranges.get(loinc, {})
        mn = float(rr.get("min", 0))
        mx = float(rr.get("max", 10000))
        unit = self.get_unit(loinc)
        ref_low, ref_high, ref_fmt = self._numeric_reference(loinc, european_format)

        midpoint = (ref_low + ref_high) / 2.0
        range_width = ref_high - ref_low
        sigma = range_width / 4.0 if range_width > 0 else midpoint * 0.1

        # BUG 2 fix: if sigma is still <= 0 (e.g. midpoint=0 and range_width=0),
        # set to a small positive value to avoid NaN/Inf from Gaussian sampling
        if sigma <= 0:
            sigma = max(abs(midpoint) * 0.01, 0.01)

        if abnormal_multiplier is not None:
            lo_mult, hi_mult = abnormal_multiplier
            multiplier = rng.uniform(lo_mult, hi_mult)
            value = midpoint * multiplier
        else:
            # Truncated Gaussian within plausibility bounds
            for _ in range(100):
                value = rng.gauss(midpoint, sigma)
                if mn <= value <= mx:
                    break
            else:
                value = max(mn, min(mx, value))

        # Clamp to plausibility
        value = max(mn, min(mx, value))

        value = self._round_value(value)
        if mn <= value <= mx:
            formatted = self._format_number(value, european_format)
        else:
            value, formatted = self._fit_display_to_bounds(loinc, value, mn, mx, unit, european_format)
        flag = self._compute_flag(value, ref_low, ref_high)

        return GeneratedValue(
            value=value,
            value_formatted=formatted,
            unit=unit,
            ref_low=ref_low,
            ref_high=ref_high,
            ref_formatted=ref_fmt,
            flag=flag,
            is_numeric=True,
            scale_type="Qn",
        )

    # ------------------------------------------------------------------
    # Internal: Semi-Quantitative (SemiQn / titre)
    # ------------------------------------------------------------------

    def _generate_semiquantitative(
        self,
        loinc: str,
        rng: random.Random,
        abnormal_multiplier: Optional[tuple[float, float]],
        european_format: bool,
    ) -> GeneratedValue:
        rr = self._ranges.get(loinc, {})
        entry_type = rr.get("type", "")

        # If it has a numeric min/max (e.g., property_template SemiQn),
        # treat as quantitative
        if "min" in rr and "max" in rr and entry_type != "titre":
            result = self._generate_quantitative(loinc, rng, abnormal_multiplier, european_format)
            result.scale_type = "SemiQn"
            return result

        # Titre-style entry
        answers = rr.get("answers", [])

        # Build a dilution series from answers or use default
        titres = self._parse_titre_series(answers)
        if not titres:
            titres = list(TITRE_SERIES)

        if abnormal_multiplier is not None:
            # Pick from upper half of titres (higher dilution = more abnormal)
            upper_half = titres[len(titres) // 2:]
            chosen = rng.choice(upper_half) if upper_half else titres[-1]
        else:
            # Normal: pick from lower half (lower dilution = more normal)
            lower_half = titres[: max(1, len(titres) // 2)]
            chosen = rng.choice(lower_half)

        # A strict upper limit has no lower bound. Compare dilution
        # denominators numerically; equality already falls outside "<".
        threshold = titres[len(titres) // 2]
        if len(titres) > 1:
            ref_formatted = f"< {threshold}"
            flag = "H" if int(chosen[2:]) >= int(threshold[2:]) else None
        else:
            ref_formatted, flag = threshold, None

        return GeneratedValue(
            value=chosen,
            value_formatted=chosen,
            unit="",
            ref_low=None,
            ref_high=None,
            ref_formatted=ref_formatted,
            flag=flag,
            is_numeric=False,
            scale_type="SemiQn",
        )

    def _parse_titre_series(self, answers: list[str]) -> list[str]:
        """Parse titre values from answer lists and return sorted dilution series."""
        titres: list[tuple[int, str]] = []
        for ans in answers:
            # Handle formats like "1:64", "Neg@1:4", etc.
            match = re.search(r"1:(\d+)", ans)
            if match:
                denom = int(match.group(1))
                titres.append((denom, f"1:{denom}"))

        if not titres:
            return []

        # Build a complete series from the minimum to the max denominator found
        max_denom = max(t[0] for t in titres)
        series: list[str] = []
        denom = 10
        while denom <= max(max_denom, 10240):
            series.append(f"1:{denom}")
            denom *= 2
            if denom > 10240:
                break
        return series if series else [t[1] for t in sorted(titres)]

    # ------------------------------------------------------------------
    # Internal: Ordinal (Ord)
    # ------------------------------------------------------------------

    def _generate_ordinal(
        self,
        loinc: str,
        rng: random.Random,
        european_format: bool,
        force_abnormal: bool = False,
    ) -> GeneratedValue:
        rr = self._ranges.get(loinc, {})
        answers = rr.get("answers", [])

        if len(answers) >= 2:
            normal_answer = answers[1]
            abnormal_answer = answers[0]
        else:
            normal_answer = "Negativo"
            abnormal_answer = "Positivo"

        if force_abnormal:
            chosen = abnormal_answer
        elif rng.random() < 0.80:
            chosen = normal_answer
        else:
            chosen = abnormal_answer

        is_abnormal = chosen == abnormal_answer
        flag = "A" if is_abnormal else None

        return GeneratedValue(
            value=chosen,
            value_formatted=chosen,
            unit="",
            ref_low=None,
            ref_high=None,
            ref_formatted=normal_answer,
            flag=flag,
            is_numeric=False,
            scale_type="Ord",
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_value(value: float) -> float:
        """Round a value to an appropriate number of decimal places."""
        return round(value, _value_decimal_places(value))

    @staticmethod
    def _fit_display_to_bounds(
        loinc: str, rounded: float, minimum: float, maximum: float, unit: str, european: bool,
    ) -> tuple[float, str]:
        """Find the closest printable value when ordinary rounding crosses a bound."""
        below = rounded < minimum
        boundary = minimum if below else maximum
        differential = unit == "%" and loinc in _AUTO_DIFFERENTIAL
        places = 2 if differential else _value_decimal_places(boundary)
        quantum = Decimal(1).scaleb(-places)
        # Decimal rounding inward keeps an exactly representable endpoint even
        # when its binary floating-point value falls just to one side of a tick.
        candidate = float(Decimal(str(boundary)).quantize(
            quantum, rounding=ROUND_CEILING if below else ROUND_FLOOR,
        ))
        if not minimum <= candidate <= maximum:
            raise ValueError(f"No value for {loinc} fits [{minimum}, {maximum}] at the report's numeric precision")
        if differential:
            return candidate, ValueGenerator._format_number_dp(candidate, places, european)
        return candidate, ValueGenerator._format_number(candidate, european)

    @staticmethod
    def _format_number(value: float, european: bool) -> str:
        """Format a numeric value, using decimal commas for European reports."""
        decimals = _value_decimal_places(value)
        if 100 <= abs(value) < 1000 and value == int(value):
            decimals = 0

        formatted = f"{value:.{decimals}f}"

        if european:
            # Split on dot
            parts = formatted.split(".")
            integer_part = parts[0]
            decimal_part = parts[1] if len(parts) > 1 else ""

            # Add thousands separator (period) for integers >= 1000
            if len(integer_part.lstrip("-")) > 3:
                sign = "-" if integer_part.startswith("-") else ""
                digits = integer_part.lstrip("-")
                groups: list[str] = []
                while digits:
                    groups.append(digits[-3:])
                    digits = digits[:-3]
                integer_part = sign + ".".join(reversed(groups))

            if decimal_part:
                return f"{integer_part},{decimal_part}"
            return integer_part

        return formatted

    @staticmethod
    def _format_ref_range(
        ref_low: Optional[float],
        ref_high: Optional[float],
        european: bool,
    ) -> str:
        """Format a reference range as a display string.

        Both endpoints use the same number of decimal places (the maximum
        of the two) so that the range reads consistently, e.g. "70,0 - 110,0"
        rather than "70,0 - 110".
        """
        if ref_low is None and ref_high is None:
            return ""
        if ref_low is None:
            return f"< {ValueGenerator._format_number(ref_high, european)}"  # type: ignore[arg-type]
        if ref_high is None:
            return f"> {ValueGenerator._format_number(ref_low, european)}"

        # Determine decimal places for each value independently
        low_dp = ValueGenerator._decimals_for(ref_low)
        high_dp = ValueGenerator._decimals_for(ref_high)
        # Use the same (maximum) decimal places for both
        dp = max(low_dp, high_dp)

        low_str = ValueGenerator._format_number_dp(ref_low, dp, european)
        high_str = ValueGenerator._format_number_dp(ref_high, dp, european)
        return f"{low_str} - {high_str}"

    @staticmethod
    def _decimals_for(value: float) -> int:
        """Return the natural number of decimal places for display.

        BUG 6 fix: cap decimals based on value magnitude to avoid
        outputs like "0,0000 - 100,0000".
        """
        absv = abs(value) if value else 0
        if absv >= 100:
            return 0
        if absv >= 10:
            return 1
        if absv >= 1:
            return 1
        if absv >= 0.1:
            return 2
        return 3

    @staticmethod
    def _format_number_dp(value: float, dp: int, european: bool) -> str:
        """Format a number with an explicit number of decimal places."""
        # BUG 3 fix: round before formatting to avoid floating-point artifacts
        value = round(value, dp)
        formatted = f"{value:.{dp}f}"
        if european:
            parts = formatted.split(".")
            integer_part = parts[0]
            decimal_part = parts[1] if len(parts) > 1 else ""
            if len(integer_part.lstrip("-")) > 3:
                sign = "-" if integer_part.startswith("-") else ""
                digits = integer_part.lstrip("-")
                groups: list[str] = []
                while digits:
                    groups.append(digits[-3:])
                    digits = digits[:-3]
                integer_part = sign + ".".join(reversed(groups))
            if decimal_part:
                return f"{integer_part},{decimal_part}"
            return integer_part
        return formatted

    @staticmethod
    def _compute_flag(
        value: float,
        ref_low: Optional[float],
        ref_high: Optional[float],
    ) -> Optional[str]:
        """Compute the abnormality flag for a numeric value."""
        if ref_high is not None and value > ref_high:
            return "H"
        if ref_low is not None and value < ref_low:
            return "L"
        return None
