"""Report Composition Engine for synthetic lab report generation.

Orchestrates panel selection, correlated value generation, and synonym
rotation to produce complete lab reports with LOINC coverage guarantees.

This is the "brain" that decides what goes into each report:
 - Selects a pathological profile (weighted random)
 - Selects a region (weighted: spain 30%, mexico 25%, etc.)
 - Picks panels for the profile + optional extras + coverage gap-filling
 - Generates correlated values via CorrelationEngine
 - Selects display names via SynonymEngine (coverage-first + regional)
 - Generates patient demographics and lab info (no Faker dependency)
 - Builds ground truth JSON for training data
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from .template_engine import SYNTHETIC_LABS
from .correlation_engine import (
    CorrelationEngine,
    PathologicalProfile,
)
from .terminology import CATALOG_PATH, POLICY_PATH, load_terms, simulation_names
from .synonym_engine import SynonymEngine, SynonymSelection
from .value_generator import ValueGenerator


# ---------------------------------------------------------------------------
# Region weights
# ---------------------------------------------------------------------------

REGION_WEIGHTS: dict[str, int] = {
    "spain": 30,
    "mexico": 25,
    "argentina": 15,
    "colombia": 10,
    "chile": 10,
    "peru": 10,
}

# All facility fields come from the same fictional regional catalog as the templates.
LAB_INFO = SYNTHETIC_LABS

# ---------------------------------------------------------------------------
# Enriched panel definitions
# ---------------------------------------------------------------------------
# The LOINC panel_definitions.json has panels with few members. We define
# "enriched" panel templates that map common clinical panels to the full
# set of LOINCs typically reported for that panel in Spanish labs.

ENRICHED_PANELS: dict[str, dict[str, Any]] = {
    "hemograma_completo": {
        "name_es": "Hemograma Completo",
        "loincs": [
            "6690-2", "789-8", "718-7", "4544-3", "787-2", "785-6",
            "786-4", "788-0", "777-3", "32623-1",
        ],
    },
    "bioquimica_basica": {
        "name_es": "Bioquimica Basica",
        "loincs": [
            "2345-7", "2160-0", "3094-0", "2132-9", "1751-7", "2885-2",
            "2823-3", "2951-2", "17861-6", "19123-9", "2777-1",
        ],
    },
    "perfil_hepatico": {
        "name_es": "Perfil Hepatico",
        "loincs": [
            "1742-6", "1920-8", "2324-2", "6768-6", "1975-2", "1968-7",
            "1751-7", "2885-2",
        ],
    },
    "perfil_lipidico": {
        "name_es": "Perfil Lipidico",
        "loincs": [
            "2093-3", "2571-8", "2085-9", "13457-7",
        ],
    },
    "perfil_tiroideo": {
        "name_es": "Perfil Tiroideo",
        "loincs": [
            "3016-3", "3024-7", "3051-0",
        ],
    },
    "coagulacion": {
        "name_es": "Coagulacion",
        "loincs": [
            "5902-2", "3173-2", "6301-6", "3255-7",
        ],
    },
    "perfil_renal": {
        "name_es": "Perfil Renal",
        "loincs": [
            "2160-0", "3094-0", "2823-3", "2951-2", "17861-6", "2777-1",
        ],
    },
    "perfil_hierro": {
        "name_es": "Metabolismo del Hierro",
        "loincs": [
            "2498-4", "2276-4", "2500-7",
        ],
    },
    "inflamacion": {
        "name_es": "Marcadores Inflamatorios",
        "loincs": [
            "1988-5", "30341-2",
        ],
    },
    "enzimas_cardiacas": {
        "name_es": "Enzimas Cardiacas",
        "loincs": [
            "2157-6", "49563-0",
        ],
    },
    "enzimas_pancreaticas": {
        "name_es": "Enzimas Pancreaticas",
        "loincs": [
            "1798-8", "3040-3",
        ],
    },
    "inmunoglobulinas": {
        "name_es": "Inmunoglobulinas",
        "loincs": [
            "2458-8", "2465-3", "2472-9",
        ],
    },
    "hormonas_basicas": {
        "name_es": "Hormonas Basicas",
        "loincs": [
            "3016-3", "10501-5", "2143-6",
        ],
    },
    "diabetes_control": {
        "name_es": "Control de Diabetes",
        "loincs": [
            "2345-7", "17856-6", "2093-3", "2571-8", "2085-9", "2160-0",
        ],
    },
    "urinalisis": {
        "name_es": "Analisis de Orina",
        "loincs": [
            "5811-5", "5803-2",
        ],
    },
}

# FIX 7: veterinary-only keywords to filter from microbiology panels
_VETERINARY_KEYWORDS = frozenset({
    'avian', 'bovine', 'porcine', 'equine', 'canine', 'feline',
    'ovine', 'caprine', 'poultry', 'swine', 'cattle',
    'rinderpest', 'bluetongue', 'infectious laryngotracheitis',
    'transmissible gastroenteritis', 'panleucopenia',
})

# Map profile recommended panel IDs to enriched panels
_PROFILE_PANEL_MAP: dict[str, list[str]] = {
    "58410-2": ["hemograma_completo"],
    "24323-8": ["bioquimica_basica"],
    "57698-3": ["perfil_lipidico"],
    "24362-6": ["perfil_renal"],
    "24325-3": ["perfil_hepatico"],
    "24348-5": ["perfil_tiroideo"],
    "34529-8": ["coagulacion"],
}

# Extra panels that can be added randomly
_EXTRA_PANEL_IDS: list[str] = [
    "perfil_hepatico", "perfil_lipidico", "perfil_tiroideo",
    "coagulacion", "perfil_hierro", "inflamacion",
    "enzimas_cardiacas", "enzimas_pancreaticas", "inmunoglobulinas",
    "hormonas_basicas", "diabetes_control", "urinalisis",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReportTest:
    """A single test result in a report."""

    loinc: str
    display_name: str
    value: str
    unit: str
    ref_range: str
    flag: str | None
    panel_name: str
    is_numeric: bool


@dataclass
class ReportContent:
    """Complete content for one lab report."""

    report_id: str
    patient_name: str
    patient_age: int
    patient_sex: str
    patient_id: str
    collection_date: str
    lab_name: str
    lab_address: str
    doctor_name: str
    panels: list[tuple[str, list[ReportTest]]]
    profile_name: str
    region: str
    ground_truth: dict


@dataclass
class CoverageTracker:
    """Tracks LOINC and synonym coverage across the dataset."""

    loinc_counts: Counter = field(default_factory=Counter)
    synonym_counts: Counter = field(default_factory=Counter)
    panel_counts: Counter = field(default_factory=Counter)
    profile_counts: Counter = field(default_factory=Counter)
    region_counts: Counter = field(default_factory=Counter)

    def uncovered_loincs(self, all_loincs: set[str]) -> set[str]:
        """Return LOINCs that have never appeared."""
        return all_loincs.difference(self.loinc_counts)

    def least_covered_loincs(self, n: int = 50) -> list[str]:
        """Return the n LOINCs with the fewest appearances."""
        if not self.loinc_counts:
            return []
        return [loinc for loinc, _ in self.loinc_counts.most_common()[: -(n + 1): -1]]


# ---------------------------------------------------------------------------
# Report Composer
# ---------------------------------------------------------------------------

class ReportComposer:
    """Compose complete lab reports by orchestrating panel selection, value
    generation (with correlations), and synonym rotation -- with LOINC
    coverage guarantees.
    """

    def __init__(self, data_dir: str, *, include_heuristic_numerics: bool = True) -> None:
        """Load all data files and initialize engines.

        Args:
            data_dir: Path to ``src/paddle_ocr/dictionaries/data/`` directory.
        """
        self._data_dir = Path(data_dir)

        # Load core data files
        ranges_path = str(self._data_dir / "reference_ranges.json")
        metadata_path = str(self._data_dir / "loinc_metadata.json")

        # Initialize ValueGenerator
        self._vg = ValueGenerator(ranges_path, metadata_path, include_heuristic_numerics=include_heuristic_numerics)

        # Initialize CorrelationEngine (uses built-in profiles)
        self._ce = CorrelationEngine(self._vg)

        self._se = SynonymEngine()
        self.input_paths = {
            "reference_ranges": Path(ranges_path),
            "loinc_metadata": Path(metadata_path),
            "panel_definitions": self._data_dir / "panel_definitions.json",
            "loinc_2_82": CATALOG_PATH,
            "simulation_policy": POLICY_PATH,
        }

        # Load panel definitions
        with open(self._data_dir / "panel_definitions.json") as f:
            raw_panels = json.load(f)["panels"]
        self._panel_defs: dict[str, dict[str, Any]] = {}
        for p in raw_panels:
            self._panel_defs[p["id"]] = p

        # Components and names supply the veterinary filter and fallback panel labels.
        with open(metadata_path, encoding="utf-8") as f:
            self._metadata: dict[str, dict] = json.load(f)

        # Build the set of all LOINCs known to the value generator
        self.eligible_loincs = frozenset(code for code in self._vg.loinc_codes if not self._is_veterinary(code))
        self.all_loincs = self.eligible_loincs
        if not self.eligible_loincs:
            raise ValueError("No eligible LOINC terms in supplied synthetic ranges")

        # Precompute: for each enriched panel, which LOINCs are known
        self._enriched_panels: dict[str, dict[str, Any]] = {}
        for panel_id, panel_def in ENRICHED_PANELS.items():
            known = [code for code in panel_def["loincs"] if code in self.eligible_loincs]
            # A panel may use method-unspecified LDL when calculated LDL is unavailable.
            if panel_id == "perfil_lipidico" and "13457-7" not in known and "2089-1" in self.eligible_loincs:
                known.append("2089-1")
            if known:
                self._enriched_panels[panel_id] = {
                    "name_es": panel_def["name_es"],
                    "loincs": known,
                }

        # Gap selection scores unique members; rendering still uses the original
        # ordered member lists for seeded shuffling. Enriched panels win ties.
        self._gap_panels: list[tuple[str, frozenset[str]]] = []
        self._panel_names: dict[str, str] = {}
        for panel_id, panel in self._enriched_panels.items():
            members = frozenset(panel["loincs"])
            self._gap_panels.append((panel_id, members))
            for code in members:
                self._panel_names.setdefault(code, panel["name_es"])
        for panel_id, panel in self._panel_defs.items():
            members = frozenset(panel.get("members", ()))
            self._gap_panels.append((panel_id, members))
            name = panel.get("name_es", panel.get("name_en", "Otros"))
            for code in members:
                self._panel_names.setdefault(code, name)

        # Lab info indexed by region for fast lookup
        self._labs_by_region: dict[str, list[dict[str, str]]] = {}
        for lab in LAB_INFO:
            self._labs_by_region.setdefault(lab["region"], []).append(lab)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def allowed_names(self, code: str) -> frozenset[str]:
        """Exact display strings admitted for this code and this input dataset."""
        if code not in self.eligible_loincs:
            raise KeyError(code)
        return frozenset(s for labels in simulation_names(code, load_terms()[code]).values() for s in labels)

    def allowed_units(self, code: str) -> frozenset[str]:
        """Exact units emitted for the admitted scale and synthetic bounds."""
        if code not in self.eligible_loincs:
            raise KeyError(code)
        return frozenset((self._vg.get_unit(code),))

    @property
    def profiles(self) -> list[PathologicalProfile]:
        """Return the list of pathological profiles."""
        return self._ce.profiles

    @property
    def enriched_panel_ids(self) -> list[str]:
        """Return all enriched panel IDs."""
        return list(self._enriched_panels.keys())

    # ------------------------------------------------------------------
    # Veterinary filter (FIX 7)
    # ------------------------------------------------------------------

    def _is_veterinary(self, loinc: str) -> bool:
        """Return True if the LOINC is a veterinary-only test.

        Checks the COMPONENT and LONG_COMMON_NAME fields in metadata for
        veterinary keywords (avian, bovine, porcine, etc.).
        """
        meta = self._metadata.get(loinc, {})
        component = meta.get("COMPONENT", "").lower()
        long_name = meta.get("LONG_COMMON_NAME", "").lower()
        text = f"{component} {long_name}"
        return any(kw in text for kw in _VETERINARY_KEYWORDS)

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def _select_region(self, rng: random.Random) -> str:
        """Select a region by weighted random choice."""
        regions = list(REGION_WEIGHTS.keys())
        weights = list(REGION_WEIGHTS.values())
        total = sum(weights)
        r = rng.uniform(0, total)
        cumulative = 0.0
        for region, weight in zip(regions, weights):
            cumulative += weight
            if r <= cumulative:
                return region
        return regions[-1]

    # ------------------------------------------------------------------
    # Panel selection
    # ------------------------------------------------------------------

    def _select_panels(
        self,
        profile: PathologicalProfile,
        rng: random.Random,
        coverage: CoverageTracker,
        gap_filling: bool = False,
    ) -> list[str]:
        """Select enriched panels for a report.

        Algorithm:
        1. Map the profile's recommended LOINC panel IDs to enriched panels.
        2. 50% chance to add 1-2 extra enriched panels.
        3. If coverage tracker shows uncovered LOINCs, 30% chance to add
           the panel containing the most of them.
        4. In gap-filling mode, always try to add under-covered panels.

        Returns:
            Deduplicated list of enriched panel IDs.
        """
        panels: list[str] = []

        # 1. Profile's recommended panels -> enriched panels
        for panel_id in profile.recommended_panels:
            enriched_ids = _PROFILE_PANEL_MAP.get(panel_id, [])
            for eid in enriched_ids:
                if eid in self._enriched_panels:
                    panels.append(eid)

        # If no enriched panels mapped, default to bioquimica + hemograma
        if not panels:
            panels.extend(p for p in ("bioquimica_basica", "hemograma_completo")
                          if p in self._enriched_panels)

        # 2. Random extra panels (50% chance, 1-2 panels)
        if rng.random() < 0.5:
            n_extra = rng.randint(1, 2)
            available = [p for p in _EXTRA_PANEL_IDS if p in self._enriched_panels and p not in panels]
            if available:
                extras = rng.sample(available, min(n_extra, len(available)))
                panels.extend(extras)

        # 3. Coverage gap-filling
        uncovered = coverage.uncovered_loincs(self.all_loincs)
        gap_chance = 1.0 if gap_filling else 0.3
        if uncovered and rng.random() < gap_chance:
            best_panel = self._find_best_gap_panel(uncovered, panels)
            if best_panel:
                panels.append(best_panel)

        return list(dict.fromkeys(panels))

    def _find_best_gap_panel(
        self,
        uncovered: set[str],
        exclude: list[str],
    ) -> str | None:
        """Find the first non-excluded panel with the largest coverage gain."""
        best_id: str | None = None
        best_count = 0
        for panel_id, members in self._gap_panels:
            if len(members) <= best_count or panel_id in exclude:
                continue
            overlap = len(members & uncovered)
            if overlap > best_count:
                best_count = overlap
                best_id = panel_id
                if best_count == len(uncovered):
                    break
        return best_id

    # ------------------------------------------------------------------
    # Collect LOINCs from panels + individual gap-fill
    # ------------------------------------------------------------------

    def _collect_loincs(
        self,
        panel_ids: list[str],
        profile: PathologicalProfile,
        coverage: CoverageTracker,
        rng: random.Random,
        gap_filling: bool = False,
    ) -> dict[str, str]:
        """Collect all member LOINCs from selected panels.

        Also adds profile-specific LOINCs that may not be in any panel,
        and optionally adds individual under-represented LOINCs.

        Returns:
            Dict of LOINC -> panel_name_es.
        """
        loinc_to_panel: dict[str, str] = {}
        micro_count = 0  # FIX 8: cap microbiology tests at 10 per report

        # Collect from enriched panels first
        for panel_id in panel_ids:
            pdef = self._enriched_panels.get(panel_id)
            if pdef:
                for loinc in pdef["loincs"]:
                    if loinc not in loinc_to_panel:
                        loinc_to_panel[loinc] = pdef["name_es"]
                continue
            # Fall back to panel_definitions.json (class groups, LOINC panels)
            pdef_full = self._panel_defs.get(panel_id)
            if pdef_full:
                panel_name = pdef_full.get("name_es", pdef_full.get("name_en", panel_id))
                # BUG 9 fix: shuffle members before applying the cap so that
                # LOINCs at the end of the deterministic list are not
                # systematically excluded.  Without this, the same LOINCs
                # always fill the 40-member cap and tail LOINCs never appear.
                members = list(pdef_full.get("members", []))
                rng.shuffle(members)
                for loinc in members:
                    if loinc not in loinc_to_panel and loinc in self.eligible_loincs:
                        # FIX 8: cap MICRO/SERO tests at 10 per report
                        loinc_meta = self._metadata.get(loinc, {})
                        loinc_cls = loinc_meta.get("CLASS", "")
                        is_micro = "MICRO" in loinc_cls or loinc_cls == "SERO"
                        if is_micro and micro_count >= 10:
                            continue
                        loinc_to_panel[loinc] = panel_name
                        if is_micro:
                            micro_count += 1
                        if len(loinc_to_panel) > 40:
                            break  # cap per-panel contribution

        # Add profile-specific LOINCs that have shifts but may not be in panels
        for loinc in profile.abnormal_shifts:
            if loinc not in loinc_to_panel and loinc in self.eligible_loincs:
                # Find a panel name for it, or use a generic name
                panel_name = self._find_panel_name_for_loinc(loinc)
                loinc_to_panel[loinc] = panel_name

        # Gap-filling: add individual under-represented LOINCs
        if gap_filling:
            uncovered = coverage.uncovered_loincs(self.all_loincs)
            # Prefer uncovered LOINCs first, then least-covered
            candidates = sorted(uncovered)
            rng.shuffle(candidates)
            if not candidates:
                candidates = coverage.least_covered_loincs(n=50)
            added = 0
            for loinc in candidates:
                if loinc not in loinc_to_panel and loinc in self.eligible_loincs:
                    # FIX 8: cap MICRO/SERO tests at 10 per report
                    loinc_meta = self._metadata.get(loinc, {})
                    loinc_cls = loinc_meta.get("CLASS", "")
                    is_micro = "MICRO" in loinc_cls or loinc_cls == "SERO"
                    if is_micro and micro_count >= 10:
                        continue
                    panel_name = self._find_panel_name_for_loinc(loinc)
                    loinc_to_panel[loinc] = panel_name
                    if is_micro:
                        micro_count += 1
                    added += 1
                    if added >= 15 or len(loinc_to_panel) > 40:
                        break  # cap at ~15 gap-fill additions per report

        return loinc_to_panel

    def _find_panel_name_for_loinc(self, loinc: str) -> str:
        """Find a panel name for an individual LOINC."""
        if loinc in self._panel_names:
            return self._panel_names[loinc]
        # Fallback: use metadata CLASS
        meta = self._metadata.get(loinc, {})
        cls = meta.get("CLASS", "")
        class_map = {
            "CHEM": "Bioquimica",
            "HEM/BC": "Hematologia",
            "COAG": "Coagulacion",
            "SERO": "Serologia",
            "MICRO": "Microbiologia",
            "UA": "Orina",
            "DRUG/TOX": "Toxicologia",
        }
        return class_map.get(cls, "Otros Analisis")

    # ------------------------------------------------------------------
    # Patient demographics
    # ------------------------------------------------------------------

    def _generate_patient(
        self,
        profile: PathologicalProfile,
        rng: random.Random,
    ) -> dict[str, Any]:
        """Generate patient demographics appropriate for the profile."""
        # Determine sex
        if profile.name == "pregnancy":
            sex = "F"
        else:
            sex = rng.choice(["M", "F"])

        # Determine age range based on profile
        age_ranges: dict[str, tuple[int, int]] = {
            "normal_adult": (18, 65),
            "diabetes_t2": (35, 75),
            "iron_deficiency_anemia": (18, 60),
            "ckd": (40, 80),
            "liver_disease": (30, 70),
            "hypothyroidism": (25, 65),
            "hyperthyroidism": (20, 55),
            "dyslipidemia": (30, 70),
            "acute_infection": (18, 70),
            "megaloblastic_anemia": (40, 80),
            "coagulopathy": (25, 70),
            "gout": (35, 70),
            "pregnancy": (18, 42),
            "elderly_normal": (65, 90),
            "pediatric": (2, 17),
        }
        age_low, age_high = age_ranges.get(profile.name, (18, 65))
        age = rng.randint(age_low, age_high)

        case_token = f"{rng.getrandbits(64):016X}"
        return {
            "name": f"SIM-CASO-{case_token}",
            "age": age,
            "sex": sex,
            "patient_id": f"SIM-ID-{case_token}",
        }

    # ------------------------------------------------------------------
    # Collection date
    # ------------------------------------------------------------------

    def _generate_date(self, rng: random.Random) -> str:
        """Generate a random collection date in DD/MM/YYYY format."""
        year = rng.choice([2024, 2025, 2026])
        month = rng.randint(1, 12)
        # Simplify: pick day 1-28 to avoid month-length issues
        day = rng.randint(1, 28)
        return f"{day:02d}/{month:02d}/{year}"

    # ------------------------------------------------------------------
    # Lab info
    # ------------------------------------------------------------------

    def _select_lab(
        self,
        region: str,
        rng: random.Random,
    ) -> dict[str, str]:
        """Select a lab from the given region, falling back to any lab."""
        labs = self._labs_by_region.get(region, [])
        if labs:
            return rng.choice(labs)
        # Fallback: pick from any region
        return rng.choice(LAB_INFO)

    # ------------------------------------------------------------------
    # Compose a single report
    # ------------------------------------------------------------------

    def compose(
        self,
        report_id: str,
        rng: random.Random,
        coverage: CoverageTracker,
        european_format: bool = True,
        gap_filling: bool = False,
        simulated_reference_ranges: bool = True,
    ) -> ReportContent:
        """Compose a complete lab report.

        Algorithm:
        1. Select pathological profile (weighted random).
        2. Select region (weighted).
        3. Select panels (profile-recommended + extras + gap-filling).
        4. Collect all member LOINCs from selected panels.
        5. Generate correlated values via CorrelationEngine.
        6. Select display names via SynonymEngine.
        7. Generate patient demographics (age-appropriate for profile).
        8. Generate lab info (regional).
        9. Build ground truth JSON.

        Args:
            report_id: Unique report identifier (e.g. "RPT_000001").
            rng: Seeded Random instance for reproducibility.
            coverage: CoverageTracker for LOINC/synonym coverage tracking.
            european_format: Use European decimal formatting (comma separator).
            gap_filling: If True, prioritize under-covered LOINCs.

        Returns:
            ReportContent with all fields populated.
        """
        # 1. Select profile
        profile = self._ce.select_profile(rng)
        coverage.profile_counts[profile.name] += 1

        # 2. Select region
        region = self._select_region(rng)
        coverage.region_counts[region] += 1

        # 3. Select panels
        panel_ids = self._select_panels(profile, rng, coverage, gap_filling=gap_filling)
        for pid in panel_ids:
            coverage.panel_counts[pid] += 1

        # 4. Collect all LOINCs
        loinc_to_panel = self._collect_loincs(
            panel_ids, profile, coverage, rng, gap_filling=gap_filling,
        )

        all_loincs = list(loinc_to_panel.keys())
        if not all_loincs:
            code = rng.choice(sorted(self.eligible_loincs))
            loinc_to_panel[code] = self._find_panel_name_for_loinc(code)
            all_loincs = [code]

        # 4b. Cap at 25 tests to prevent JSON truncation during training/eval
        if len(all_loincs) > 25:
            kept = rng.sample(all_loincs, 25)
            kept_set = set(kept)
            loinc_to_panel = {k: v for k, v in loinc_to_panel.items() if k in kept_set}
            all_loincs = kept

        # 5. Generate correlated values
        _, values = self._ce.generate_correlated_report(
            all_loincs, rng, profile=profile, european_format=european_format,
        )

        # Derived values may be computed from other selected measurements.
        # Keep the selected report order and its explicit maximum row count.
        values = {code: values[code] for code in all_loincs if code in values}

        # 6 & 9. Build ReportTest objects and ground truth, grouped by panel
        panel_tests: dict[str, list[ReportTest]] = {}
        ground_truth_results: list[dict[str, Any]] = []

        for loinc, gv in values.items():
            panel_name = loinc_to_panel.get(loinc, "Otros")

            # Select display name via SynonymEngine
            selection: SynonymSelection = self._se.select(
                loinc,
                rng,
                region=region,
                coverage_tracker=coverage.synonym_counts,
            )

            # Build the ReportTest
            rt = ReportTest(
                loinc=loinc,
                display_name=selection.display_name,
                value=gv.value_formatted,
                unit=gv.unit,
                ref_range=gv.ref_formatted if simulated_reference_ranges else "",
                flag=gv.flag if simulated_reference_ranges else None,
                panel_name=panel_name,
                is_numeric=gv.is_numeric,
            )

            panel_tests.setdefault(panel_name, []).append(rt)

            # Track LOINC coverage
            coverage.loinc_counts[loinc] += 1

            # Build ground truth entry
            # For ground truth, use raw unformatted values
            gt_value = str(gv.value) if gv.is_numeric else gv.value_formatted
            if not simulated_reference_ranges:
                gt_ref = ""
            elif gv.ref_low is not None and gv.ref_high is not None:
                gt_ref = f"{gv.ref_low} - {gv.ref_high}"
            elif gv.ref_low is not None:
                gt_ref = f"> {gv.ref_low}"
            elif gv.ref_high is not None:
                gt_ref = f"< {gv.ref_high}"
            else:
                gt_ref = gv.ref_formatted

            ground_truth_results.append({
                "test_name": selection.display_name,
                "value": gt_value,
                "unit": gv.unit,
                "reference_range": gt_ref,
                "flag": rt.flag,
                "loinc": loinc,
                "panel": panel_name,
            })

        # Build ordered panels list
        panels_ordered: list[tuple[str, list[ReportTest]]] = []
        # Maintain deterministic order matching panel_ids
        seen_panels: set[str] = set()
        for pid in panel_ids:
            pname = self._enriched_panels.get(pid, {}).get("name_es", pid)
            if pname in panel_tests and pname not in seen_panels:
                panels_ordered.append((pname, panel_tests[pname]))
                seen_panels.add(pname)
        # Add any remaining panels not yet included
        for pname, tests in panel_tests.items():
            if pname not in seen_panels:
                panels_ordered.append((pname, tests))
                seen_panels.add(pname)

        # 7. Generate patient demographics
        patient = self._generate_patient(profile, rng)

        # 8. Generate lab info + doctor + date
        lab = self._select_lab(region, rng)
        doctor = f"SIM-CLIN-{rng.getrandbits(64):016X}"
        collection_date = self._generate_date(rng)

        # 10. Build ground truth
        ground_truth = {"results": ground_truth_results}

        return ReportContent(
            report_id=report_id,
            patient_name=patient["name"],
            patient_age=patient["age"],
            patient_sex=patient["sex"],
            patient_id=patient["patient_id"],
            collection_date=collection_date,
            lab_name=lab["name"],
            lab_address=lab["addr"],
            doctor_name=doctor,
            panels=panels_ordered,
            profile_name=profile.name,
            region=region,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # Batch composition with coverage guarantees
    # ------------------------------------------------------------------

    def compose_batch(
        self,
        count: int,
        seed: int = 42,
        min_loinc_coverage: int = 3,
        european_format: bool = True,
    ) -> list[ReportContent]:
        """Compose a batch of reports with coverage guarantees.

        First 90% of reports: normal weighted generation.
        Last 10%: gap-filling mode -- prioritize uncovered/under-covered LOINCs.

        After generation, verify every LOINC that was generated appears
        at least ``min_loinc_coverage`` times.  If not, generate additional
        gap-filling reports until the threshold is met or a maximum of
        20% extra reports are generated.

        Args:
            count: Number of reports to generate.
            seed: Random seed for reproducibility.
            min_loinc_coverage: Minimum times each LOINC should appear.
            european_format: Use European decimal formatting.

        Returns:
            List of ReportContent instances.
        """
        rng = random.Random(seed)
        coverage = CoverageTracker()
        reports: list[ReportContent] = []

        normal_count = int(count * 0.9)
        gap_count = count - normal_count

        # Phase 1: Normal generation (90%)
        for i in range(normal_count):
            report_id = f"RPT_{i + 1:06d}"
            report = self.compose(
                report_id, rng, coverage,
                european_format=european_format,
                gap_filling=False,
            )
            reports.append(report)

        # Phase 2: Gap-filling generation (10%)
        for i in range(gap_count):
            idx = normal_count + i
            report_id = f"RPT_{idx + 1:06d}"
            report = self.compose(
                report_id, rng, coverage,
                european_format=european_format,
                gap_filling=True,
            )
            reports.append(report)

        # Phase 3: Extra gap-filling if coverage threshold not met
        # BUG 22 fix: track ALL known LOINCs, not just the ~100 from
        # enriched panels.  The old code only checked 15 enriched panels
        # (~100 LOINCs) and missed the remaining 6,300+ LOINCs entirely.
        trackable_loincs: set[str] = self.all_loincs

        max_extra = int(count * 0.2)
        extra_generated = 0

        while extra_generated < max_extra:
            under_covered = [
                code for code in trackable_loincs
                if coverage.loinc_counts.get(code, 0) < min_loinc_coverage
            ]
            if not under_covered:
                break

            idx = count + extra_generated
            report_id = f"RPT_{idx + 1:06d}"
            report = self.compose(
                report_id, rng, coverage,
                european_format=european_format,
                gap_filling=True,
            )
            reports.append(report)
            extra_generated += 1

        return reports
