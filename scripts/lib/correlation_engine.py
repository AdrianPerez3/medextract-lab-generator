"""Correlated value engine for multi-test pathological profiles.

Generates clinically realistic *correlated* lab values: when a patient has
diabetes, glucose AND HbA1c AND triglycerides are all elevated together --
not randomly.  Profiles define shift multipliers for each LOINC relative to
the midpoint of its reference range.

Includes derived-value formulas (Friedewald LDL, A/G ratio, anion gap,
MCHC, MCH, MCV) that are recomputed from source analytes when all sources
are present in the generated panel.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Optional

from .value_generator import GeneratedValue, ValueGenerator, _AUTO_DIFFERENTIAL


# ---------------------------------------------------------------------------
# PathologicalProfile dataclass
# ---------------------------------------------------------------------------
@dataclass
class PathologicalProfile:
    """A pathological profile that defines correlated shift patterns.

    Attributes:
        name: Machine-readable identifier (e.g. "diabetes_t2").
        name_es: Spanish display name (e.g. "Diabetes tipo 2").
        weight: Probability weight for random selection -- higher = more likely.
        description: Brief clinical description of the profile.
        abnormal_shifts: Mapping of LOINC code to (multiplier_min, multiplier_max)
            applied to the reference-range midpoint.  A multiplier of 1.0 means
            the midpoint itself; >1.0 means elevated; <1.0 means depressed.
        recommended_panels: Panel IDs from panel_definitions.json that should
            be included when this profile is active.
    """

    name: str
    name_es: str
    weight: float
    description: str
    abnormal_shifts: dict[str, tuple[float, float]] = field(default_factory=dict)
    recommended_panels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 15 pathological profiles
# ---------------------------------------------------------------------------
PROFILES: list[PathologicalProfile] = [
    # 1. Normal adult (most common)
    PathologicalProfile(
        name="normal_adult",
        name_es="Adulto normal",
        weight=30,
        description="Healthy adult, all values within normal ranges",
        abnormal_shifts={},
        recommended_panels=["58410-2", "24323-8"],
    ),
    # 2. Diabetes Type 2
    PathologicalProfile(
        name="diabetes_t2",
        name_es="Diabetes tipo 2",
        weight=8,
        description="Elevated glucose, HbA1c, triglycerides; low HDL",
        abnormal_shifts={
            "2345-7": (1.3, 2.5),    # Glucose: 130-250% of midpoint
            "4548-4": (1.2, 1.8),    # HbA1c: elevated
            "2571-8": (1.3, 2.5),    # Triglycerides: elevated
            "2085-9": (0.5, 0.8),    # HDL: low
            "2160-0": (1.0, 1.5),    # Creatinine: mildly elevated
        },
        recommended_panels=["24323-8", "57698-3"],
    ),
    # 3. Iron-deficiency anemia
    PathologicalProfile(
        name="iron_deficiency_anemia",
        name_es="Anemia ferropenica",
        weight=6,
        description="Microcytic anemia with low iron stores and elevated TIBC",
        abnormal_shifts={
            "718-7": (0.5, 0.8),     # Hemoglobin: low
            "4544-3": (0.5, 0.8),    # Hematocrit: low
            "787-2": (0.6, 0.85),    # MCV: low (microcytic)
            "786-4": (0.8, 0.95),    # MCHC: low
            "785-6": (0.6, 0.85),    # MCH: low
            "2498-4": (0.1, 0.5),    # Iron: very low
            "2276-4": (0.05, 0.3),   # Ferritin: very low
            "3034-6": (1.2, 1.6),    # Transferrin: elevated
            "777-3": (1.0, 1.4),     # Platelets: normal to elevated
            "788-0": (1.1, 1.3),     # RDW: elevated
        },
        recommended_panels=["58410-2"],
    ),
    # 4. Chronic kidney disease
    PathologicalProfile(
        name="ckd",
        name_es="Enfermedad renal cronica",
        weight=5,
        description="Elevated creatinine and BUN with electrolyte imbalances",
        abnormal_shifts={
            "2160-0": (1.5, 4.0),    # Creatinine: elevated
            "3094-0": (1.5, 3.0),    # BUN/Urea: elevated
            "2823-3": (1.1, 1.6),    # Potassium: elevated
            "17861-6": (0.7, 0.9),   # Calcium: low
            "2777-1": (1.2, 1.8),    # Phosphorus: elevated
            "718-7": (0.6, 0.85),    # Hemoglobin: low (renal anemia)
        },
        recommended_panels=["24362-6", "58410-2"],
    ),
    # 5. Liver disease / hepatitis
    PathologicalProfile(
        name="liver_disease",
        name_es="Enfermedad hepatica",
        weight=5,
        description="Markedly elevated transaminases with cholestasis markers",
        abnormal_shifts={
            "1742-6": (2.0, 10.0),   # ALT: very elevated
            "1920-8": (2.0, 8.0),    # AST: very elevated
            "2324-2": (1.5, 5.0),    # GGT: elevated
            "1975-2": (1.5, 5.0),    # Total bilirubin: elevated
            "1968-7": (1.2, 3.0),    # Direct bilirubin: elevated
            "6768-6": (1.3, 3.0),    # ALP: elevated
            "1751-7": (0.5, 0.8),    # Albumin: low
            "5902-2": (1.3, 2.0),    # PT: prolonged
        },
        recommended_panels=["24325-3"],
    ),
    # 6. Hypothyroidism
    PathologicalProfile(
        name="hypothyroidism",
        name_es="Hipotiroidismo",
        weight=4,
        description="Elevated TSH with low free T4 and T3",
        abnormal_shifts={
            "3016-3": (2.0, 10.0),   # TSH: very elevated
            "3024-7": (0.4, 0.7),    # Free T4: low
            "3051-0": (0.5, 0.8),    # Free T3: low
            "2093-3": (1.1, 1.4),    # Cholesterol: mildly elevated
        },
        recommended_panels=["24348-5"],
    ),
    # 7. Hyperthyroidism
    PathologicalProfile(
        name="hyperthyroidism",
        name_es="Hipertiroidismo",
        weight=3,
        description="Suppressed TSH with elevated free T4 and T3",
        abnormal_shifts={
            "3016-3": (0.01, 0.1),   # TSH: suppressed
            "3024-7": (1.3, 2.0),    # Free T4: elevated
            "3051-0": (1.2, 1.8),    # Free T3: elevated
        },
        recommended_panels=["24348-5"],
    ),
    # 8. Dyslipidemia
    PathologicalProfile(
        name="dyslipidemia",
        name_es="Dislipidemia",
        weight=7,
        description="Elevated total cholesterol, LDL, triglycerides; low HDL",
        abnormal_shifts={
            "2093-3": (1.2, 1.8),    # Total cholesterol: elevated
            "2089-1": (1.3, 2.0),    # LDL: elevated
            "2085-9": (0.5, 0.8),    # HDL: low
            "2571-8": (1.5, 3.0),    # Triglycerides: elevated
        },
        recommended_panels=["57698-3"],
    ),
    # 9. Acute infection / inflammation
    PathologicalProfile(
        name="acute_infection",
        name_es="Infeccion aguda",
        weight=6,
        description="Leukocytosis with elevated inflammatory markers",
        abnormal_shifts={
            "6690-2": (1.5, 3.0),    # WBC: elevated (leukocytosis)
            "770-8": (1.3, 2.0),     # Neutrophils: elevated
            "1988-5": (5.0, 50.0),   # CRP: very elevated
            "30341-2": (2.0, 5.0),   # ESR: elevated
        },
        recommended_panels=["58410-2"],
    ),
    # 10. Megaloblastic anemia (B12/folate deficiency)
    PathologicalProfile(
        name="megaloblastic_anemia",
        name_es="Anemia megaloblastica",
        weight=3,
        description="Macrocytic anemia with B12 and/or folate deficiency",
        abnormal_shifts={
            "718-7": (0.5, 0.8),     # Hemoglobin: low
            "787-2": (1.1, 1.3),     # MCV: high (macrocytic)
            "2132-9": (0.1, 0.4),    # Vitamin B12: very low
            "2284-8": (0.1, 0.5),    # Folate: low
            "788-0": (1.1, 1.3),     # RDW: elevated
        },
        recommended_panels=["58410-2"],
    ),
    # 11. Coagulation disorder
    PathologicalProfile(
        name="coagulopathy",
        name_es="Coagulopatia",
        weight=3,
        description="Prolonged clotting times with low platelets",
        abnormal_shifts={
            "5902-2": (1.3, 2.5),    # PT: prolonged
            "3173-2": (1.3, 2.0),    # aPTT: prolonged
            "6301-6": (1.5, 4.0),    # INR: elevated
            "777-3": (0.3, 0.7),     # Platelets: low
        },
        recommended_panels=["34529-8"],
    ),
    # 12. Gout / hyperuricemia
    PathologicalProfile(
        name="gout",
        name_es="Gota / Hiperuricemia",
        weight=3,
        description="Elevated uric acid with mildly impaired renal function",
        abnormal_shifts={
            "3084-1": (1.3, 2.0),    # Uric acid: elevated
            "2160-0": (1.0, 1.3),    # Creatinine: mildly elevated
        },
        recommended_panels=["24323-8"],
    ),
    # 13. Pregnancy (normal)
    PathologicalProfile(
        name="pregnancy",
        name_es="Embarazo normal",
        weight=3,
        description="Dilutional anemia with elevated ALP and low ferritin",
        abnormal_shifts={
            "718-7": (0.85, 0.95),   # Hemoglobin: slightly low (dilutional)
            "4544-3": (0.85, 0.95),  # Hematocrit: slightly low
            "2345-7": (0.8, 1.0),    # Glucose: normal-low
            "6768-6": (1.5, 2.5),    # ALP: elevated (placental)
            "2276-4": (0.3, 0.6),    # Ferritin: low
        },
        recommended_panels=["58410-2", "24323-8"],
    ),
    # 14. Elderly normal (age-adjusted ranges)
    PathologicalProfile(
        name="elderly_normal",
        name_es="Adulto mayor",
        weight=5,
        description="Age-adjusted values with slightly elevated glucose and creatinine",
        abnormal_shifts={
            "718-7": (0.9, 1.0),     # Hemoglobin: slightly lower
            "2345-7": (1.0, 1.15),   # Glucose: slightly higher
            "2160-0": (1.0, 1.2),    # Creatinine: slightly higher
            "30341-2": (1.0, 1.5),   # ESR: slightly elevated
        },
        recommended_panels=["58410-2", "24323-8"],
    ),
    # 15. Pediatric normal (child ranges)
    PathologicalProfile(
        name="pediatric",
        name_es="Pediatrico",
        weight=4,
        description="Child-appropriate values with elevated ALP from bone growth",
        abnormal_shifts={
            "6768-6": (1.5, 3.0),    # ALP: elevated (growing bones)
            "6690-2": (1.0, 1.3),    # WBC: slightly higher in children
        },
        recommended_panels=["58410-2", "24323-8"],
    ),
]


# ---------------------------------------------------------------------------
# Derived-value specifications
# ---------------------------------------------------------------------------
@dataclass
class DerivedFormula:
    """Specification for a clinically derived lab value.

    Attributes:
        target_loinc: The LOINC code of the derived result.
        source_loincs: LOINCs that must all be present to compute the result.
        compute: Callable that takes a dict of {loinc: float} source values
            and returns the derived float value.
        description: Human-readable formula description.
    """

    target_loinc: str
    source_loincs: list[str]
    compute: Callable[[dict[str, float]], Optional[float]]
    description: str


def _friedewald_ldl(sources: dict[str, float]) -> Optional[float]:
    """Classical Friedewald estimate in mg/dL (Friedewald et al., PMID 4337382)."""
    triglycerides = sources["2571-8"]
    if triglycerides >= 400 or min(sources.values()) < 0:
        return None
    estimated = sources["2093-3"] - sources["2085-9"] - triglycerides / 5.0
    return estimated if estimated >= 0 else None


def _ag_ratio(sources: dict[str, float]) -> Optional[float]:
    """Albumin / Globulin ratio: Albumin / (Total Protein - Albumin).

    Returns None when globulin <= 0 (invalid input) so the derived value
    is skipped rather than producing a fake constant.
    """
    globulin = sources["2885-2"] - sources["1751-7"]
    if globulin <= 0:
        return None  # BUG 3 fix: skip instead of returning constant 2.0
    return sources["1751-7"] / globulin


def _anion_gap(sources: dict[str, float]) -> float:
    """Anion gap = Na - (Cl + HCO3).

    Uses LOINC 2075-0 for chloride and 1963-8 for bicarbonate.
    """
    return sources["2951-2"] - (sources["2075-0"] + sources["1963-8"])


def _mchc(sources: dict[str, float]) -> Optional[float]:
    """MCHC = Hemoglobin / Hematocrit * 100 (g/dL).

    Returns None when Hct <= 0 so the derived value is skipped
    rather than producing a fake normal value with wrong flags.
    """
    if sources.get("4544-3", 0) <= 0:
        return None  # BUG 6 fix: skip, don't fake
    return sources["718-7"] / sources["4544-3"] * 100.0


def _mch(sources: dict[str, float]) -> Optional[float]:
    """MCH = Hemoglobin / RBC * 10 (pg).

    Returns None when RBC <= 0 so the derived value is skipped.
    """
    if sources.get("789-8", 0) <= 0:
        return None  # Skip, don't fake
    return sources["718-7"] / sources["789-8"] * 10.0


def _mcv(sources: dict[str, float]) -> Optional[float]:
    """MCV = Hematocrit / RBC * 10 (fL).

    Returns None when RBC <= 0 so the derived value is skipped.
    """
    if sources.get("789-8", 0) <= 0:
        return None  # Skip, don't fake
    return sources["4544-3"] / sources["789-8"] * 10.0


# These constants assume specific input and output units. A compatible code alone
# is insufficient (e.g. LDL arithmetic in mg/dL cannot be applied to mmol/L).
_FORMULA_UNITS = {
    "2093-3": "mg/dL", "2085-9": "mg/dL", "2571-8": "mg/dL", "13457-7": "mg/dL",
    "1751-7": "g/dL", "2885-2": "g/dL", "1759-0": "{ratio}",
    "2951-2": "mmol/L", "2075-0": "mmol/L", "1963-8": "mmol/L", "10466-1": "mmol/L",
    "718-7": "g/dL", "4544-3": "%", "789-8": "10*6/uL", "786-4": "g/dL",
    "785-6": "pg", "787-2": "fL",
}


DERIVED_FORMULAS: list[DerivedFormula] = [
    DerivedFormula(
        target_loinc="13457-7",  # LDL cholesterol (calculated)
        source_loincs=["2093-3", "2085-9", "2571-8"],
        compute=_friedewald_ldl,
        description="Friedewald: LDL = Total Chol - HDL - TG/5",
    ),
    DerivedFormula(
        target_loinc="1759-0",  # Albumin/globulin mass ratio
        source_loincs=["1751-7", "2885-2"],
        compute=_ag_ratio,
        description="A/G ratio = Albumin / (Total Protein - Albumin)",
    ),
    DerivedFormula(
        target_loinc="10466-1",  # Sodium-only, three-ion calculated anion gap
        source_loincs=["2951-2", "2075-0", "1963-8"],
        compute=_anion_gap,
        description="Anion gap = Na - (Cl + HCO3)",
    ),
    DerivedFormula(
        target_loinc="786-4",  # MCHC
        source_loincs=["718-7", "4544-3"],
        compute=_mchc,
        description="MCHC = Hb / Hct * 100",
    ),
    DerivedFormula(
        target_loinc="785-6",  # MCH
        source_loincs=["718-7", "789-8"],
        compute=_mch,
        description="MCH = Hb / RBC * 10",
    ),
    DerivedFormula(
        target_loinc="787-2",  # MCV
        source_loincs=["4544-3", "789-8"],
        compute=_mcv,
        description="MCV = Hct / RBC * 10",
    ),
]


# ---------------------------------------------------------------------------
# CorrelationEngine
# ---------------------------------------------------------------------------
class CorrelationEngine:
    """Generate multi-test values with clinically realistic correlations.

    Given a pathological profile, applies correlated shift multipliers so
    that related analytes move together (e.g. diabetes elevates glucose,
    HbA1c, and triglycerides simultaneously).

    After generating raw values, applies derived-value formulas (Friedewald
    LDL, MCHC, MCH, MCV, etc.) to ensure mathematical consistency.
    """

    def __init__(
        self,
        value_generator: ValueGenerator,
        profiles: list[PathologicalProfile] | None = None,
    ) -> None:
        """Initialize with a ValueGenerator and optional custom profiles.

        Args:
            value_generator: A configured ValueGenerator instance.
            profiles: Custom profiles list.  Defaults to the built-in 15
                profiles (PROFILES constant) when None.
        """
        self.vg = value_generator
        self.profiles = profiles if profiles is not None else list(PROFILES)
        self._differential_bounds = {}
        for code in _AUTO_DIFFERENTIAL:
            if self.vg.has_loinc(code) and self.vg.get_unit(code) == "%":
                low, high = self.vg.simulation_bounds(code)
                self._differential_bounds[code] = (math.ceil(low * 100 - 1e-9),
                                                   math.floor(high * 100 + 1e-9))
        minima = sum(lo for lo, _ in self._differential_bounds.values())
        maxima = sum(hi for _, hi in self._differential_bounds.values())
        complete = len(self._differential_bounds) == len(_AUTO_DIFFERENTIAL)
        if minima > 10000 or (complete and maxima < 10000) or any(
                lo > hi for lo, hi in self._differential_bounds.values()):
            raise ValueError("Automated differential bounds cannot make coherent percentages")


    # ------------------------------------------------------------------
    # Profile selection
    # ------------------------------------------------------------------

    def select_profile(self, rng: random.Random) -> PathologicalProfile:
        """Select a pathological profile by weighted random choice.

        Uses the ``weight`` field of each profile as a probability weight.
        Higher weight = more likely to be selected.

        Args:
            rng: Seeded Random instance for reproducibility.

        Returns:
            The selected PathologicalProfile.
        """
        weights = [p.weight for p in self.profiles]
        total = sum(weights)
        r = rng.uniform(0, total)
        cumulative = 0.0
        for profile in self.profiles:
            cumulative += profile.weight
            if r <= cumulative:
                return profile
        # Fallback (should not reach here due to float precision)
        return self.profiles[-1]

    # ------------------------------------------------------------------
    # Panel generation
    # ------------------------------------------------------------------

    def generate_panel(
        self,
        loincs: list[str],
        profile: PathologicalProfile,
        rng: random.Random,
        european_format: bool = True,
    ) -> dict[str, GeneratedValue]:
        """Generate correlated values for a set of LOINCs given a profile.

        For LOINCs in the profile's ``abnormal_shifts``, applies the
        multiplier range to the reference-range midpoint.  For LOINCs NOT
        in the shift map, generates normal values.

        Args:
            loincs: List of LOINC codes to generate values for.
            profile: The pathological profile defining shift patterns.
            rng: Seeded Random instance.
            european_format: Use European decimal formatting (comma separator).

        Returns:
            Dict mapping LOINC code to GeneratedValue.  LOINCs that are
            unknown to the ValueGenerator are silently skipped.
        """
        results: dict[str, GeneratedValue] = {}

        for loinc in loincs:
            if not self.vg.has_loinc(loinc):
                continue

            shift = profile.abnormal_shifts.get(loinc)
            if shift is not None:
                # Apply the correlated shift multiplier
                gv = self.vg.generate(
                    loinc,
                    rng,
                    abnormal_multiplier=shift,
                    european_format=european_format,
                )
            else:
                # Normal value -- no shift for this LOINC in this profile
                gv = self.vg.generate_normal(
                    loinc, rng, european_format=european_format,
                )

            results[loinc] = gv

        self._balance_cholesterol(results, rng, european_format)
        self._balance_differential(results, profile, rng, european_format)
        return results

    def _balance_cholesterol(self, results, rng, european_format):
        """Sample HDL and method-unspecified LDL within the same-unit total.

        The remaining cholesterol can belong to other fractions. Triglycerides
        play no part in this sampling; calculated LDL has a separate code.
        """
        total = results.get("2093-3")
        if total is None or total.unit != "mg/dL":
            return
        # Decimal arithmetic keeps the constraint exact at printed precision.
        remaining = Decimal(str(total.value))
        for code in ("2085-9", "2089-1"):
            component = results.get(code)
            if component is None or component.unit != total.unit:
                continue
            amount = Decimal(str(component.value))
            if amount > remaining:
                low, high = self.vg.simulation_bounds(code)
                # Tenths of mg/dL give a visible, reproducible sample. Omit a
                # component if its supplied envelope cannot fit the balance.
                first = math.ceil(Decimal(str(low)) * 10)
                last = math.floor(min(remaining, Decimal(str(high))) * 10)
                if first > last:
                    del results[code]
                    continue
                amount = Decimal(rng.randint(first, last)) / 10
                component.value = float(amount)
                component.value_formatted = self.vg._format_number_dp(component.value, 1, european_format)
                component.flag = self.vg._compute_flag(component.value, component.ref_low, component.ref_high)
            remaining -= amount

    def _balance_differential(self, results, profile, rng, european_format):
        """Allocate whole basis points within the admitted five-part envelopes."""
        bounds = self._differential_bounds
        if not any(code in results for code in bounds):
            return
        # Include available but unprinted components in the same draw. A
        # partial report then retains a residual for the unprinted components.
        ticks = {}
        for code, (low, high) in bounds.items():
            row = results.get(code)
            if row is None:
                row = self.vg.generate(code, rng, profile.abnormal_shifts.get(code), european_format)
            ticks[code] = max(low, min(high, round(float(row.value) * 100)))

        observed = sum(ticks.values())
        complete = len(bounds) == len(_AUTO_DIFFERENTIAL)
        target = 10000 if complete else min(observed, 10000)
        delta = target - observed
        if delta:
            # Allocate the adjustment in proportion to remaining headroom.
            # Largest remainders preserve the exact total after display rounding.
            sign = 1 if delta > 0 else -1
            capacity = {c: bounds[c][1] - n if delta > 0 else n - bounds[c][0]
                        for c, n in ticks.items()}
            total_capacity = sum(capacity.values())
            residuals = {}
            assigned = 0
            for code, available in capacity.items():
                step, residuals[code] = divmod(abs(delta) * available, total_capacity)
                ticks[code] += sign * step
                assigned += step
            order = sorted(residuals, key=residuals.get, reverse=True)
            for code in order[:abs(delta) - assigned]:
                ticks[code] += sign

        for code, amount in ticks.items():
            if code in results:
                row = results[code]
                row.value = amount / 100
                row.value_formatted = ValueGenerator._format_number_dp(row.value, 2, european_format)
                row.flag = ValueGenerator._compute_flag(row.value, row.ref_low, row.ref_high)

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------

    def apply_derived_values(
        self,
        values: dict[str, GeneratedValue],
        european_format: bool = True,
    ) -> dict[str, GeneratedValue]:
        """Recompute derived values from their source analytes.

        Applies each formula in ``DERIVED_FORMULAS`` when ALL source LOINCs
        are present in ``values`` and have numeric results.  The target LOINC
        is overwritten (or inserted) with the computed value, maintaining
        proper formatting, units, reference ranges, and flags.

        Args:
            values: Mutable dict of LOINC -> GeneratedValue (modified in place).
            european_format: European decimal formatting.

        Returns:
            The same dict (modified in place), for convenience.
        """
        for formula in DERIVED_FORMULAS:
            target = formula.target_loinc
            if not self.vg.has_loinc(target):
                values.pop(target, None)
                continue
            if not all(code in values for code in formula.source_loincs):
                continue
            # Once inputs are present, a failed derivation must not leave the
            # previously sampled independent target looking like its result.
            values.pop(target, None)
            if self.vg.get_unit(target) != _FORMULA_UNITS[target]:
                continue
            if any(not values[c].is_numeric or values[c].unit != _FORMULA_UNITS[c]
                   for c in formula.source_loincs):
                continue
            sources = {c: float(values[c].value) for c in formula.source_loincs}
            if not all(map(math.isfinite, sources.values())):
                continue
            result = formula.compute(sources)
            if result is None:
                continue
            low_bound, high_bound = self.vg.simulation_bounds(target)
            value = ValueGenerator._round_value(result)
            # Both the formula and its rounded result must fit the target's range.
            if not (low_bound <= result <= high_bound and low_bound <= value <= high_bound):
                continue
            low, high = self.vg.get_ref_range(target)
            values[target] = GeneratedValue(
                value=value,
                value_formatted=ValueGenerator._format_number(value, european_format),
                unit=self.vg.get_unit(target),
                ref_low=low, ref_high=high,
                ref_formatted=ValueGenerator._format_ref_range(low, high, european_format),
                flag=ValueGenerator._compute_flag(value, low, high),
                is_numeric=True, scale_type="Qn",
            )

        return values

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------

    def generate_correlated_report(
        self,
        loincs: list[str],
        rng: random.Random,
        profile: PathologicalProfile | None = None,
        european_format: bool = True,
    ) -> tuple[PathologicalProfile, dict[str, GeneratedValue]]:
        """Generate a complete correlated report with derived values.

        Selects a profile (if not provided), generates correlated panel
        values, and applies derived formulas.

        Args:
            loincs: List of LOINC codes to generate.
            rng: Seeded Random instance.
            profile: Explicit profile, or None for weighted random selection.
            european_format: European decimal formatting.

        Returns:
            Tuple of (selected_profile, {loinc: GeneratedValue}).
        """
        if profile is None:
            profile = self.select_profile(rng)

        values = self.generate_panel(
            loincs, profile, rng, european_format=european_format,
        )
        self.apply_derived_values(values, european_format=european_format)

        return profile, values
