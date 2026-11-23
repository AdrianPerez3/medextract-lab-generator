"""Synonym Rotation Engine for synthetic report generation.

Selects whole, same-code display names from the pinned LOINC catalog,
with regional preferences and optional typography variation.
"""

from __future__ import annotations

import random
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .terminology import load_terms, simulation_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Remove diacritical marks: a->a, e->e, i->i, o->o, u->u, n->n, u->u."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=2048)
def _normalize_key(name: str) -> str:
    """Case-insensitive dedup key that also strips accents."""
    return strip_accents(name).strip().lower()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SynonymEntry:
    """One synonym for a LOINC code, with provenance metadata."""

    display_name: str
    source: str
    region: str | None = None  # e.g. "spain", "mexico"

    @property
    def dedup_key(self) -> str:
        return _normalize_key(self.display_name)


@dataclass(frozen=True, slots=True)
class SynonymSelection:
    """Result of a synonym selection."""

    display_name: str
    source: str
    region: str | None = None
    case_variant: str = "original"  # "title", "upper", "lower", "mixed", "original"
    accent_variant: str = "original"  # "with_accents", "without_accents", "original"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SynonymEngine:
    """Selects pinned LOINC display names with coverage tracking."""

    def __init__(
        self,
        dict_path: str | Path | None = None,
        regional_path: str | Path | None = None,
        translations_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        # Existing direct callers may pass legacy file paths. Unverified aliases
        # from those files cannot override the names attached to an official code.
        self.synonyms: dict[str, list[SynonymEntry]] = {}
        self._regional_map: dict[str, dict[str, list[SynonymEntry]]] = {}
        for code, term in load_terms().items():
            combined = {}
            regional = {}
            for locale, labels in simulation_names(code, term).items():
                region = None if locale == "en" else locale
                entries = [SynonymEntry(name, "loinc_translation", region) for name in labels]
                if region:
                    regional[region] = entries
                for entry in entries:
                    combined.setdefault(entry.dedup_key, entry)
            self.synonyms[code] = list(combined.values())
            self._regional_map[code] = regional

    def select(
        self,
        loinc: str,
        rng: random.Random,
        *,
        region: str | None = None,
        coverage_tracker: Counter | None = None,
        apply_case_variation: bool = False,
        apply_accent_variation: bool = False,
    ) -> SynonymSelection:
        """Select a synonym for the given LOINC.

        Selection priority:
        1. If coverage_tracker provided AND some synonyms have count=0, prefer those.
        2. If region specified, 60% chance to use that region's variant.
        3. Otherwise, sample among the code-bound display names.

        Optional typography variants must themselves occur in this code's catalog.
        """
        entries = self.synonyms.get(loinc)
        if not entries:
            raise KeyError(f"Unknown LOINC display name: {loinc}")

        chosen = self._pick_entry(entries, loinc, rng, region, coverage_tracker)

        # Some antigen labels are case-sensitive (e.g. Rh C vs c). Only
        # select an alternate spelling if it belongs to this code as written.
        display = chosen.display_name
        attested = {e.display_name for e in entries} if apply_accent_variation or apply_case_variation else ()
        case_variant = accent_variant = "original"
        if apply_accent_variation and rng.random() < 0.20:
            alternate = strip_accents(display)
            if alternate in attested and alternate != display:
                display, accent_variant = alternate, "without_accents"
        if apply_case_variation and rng.random() < 0.30:
            alternate, variant = self._apply_case(display, rng)
            if alternate in attested and alternate != display:
                display, case_variant = alternate, variant

        # Update coverage tracker
        if coverage_tracker is not None:
            coverage_tracker[chosen.dedup_key] += 1

        return SynonymSelection(
            display_name=display,
            source=chosen.source,
            region=chosen.region,
            case_variant=case_variant,
            accent_variant=accent_variant,
        )

    def _pick_entry(
        self,
        entries: list[SynonymEntry],
        loinc: str,
        rng: random.Random,
        region: str | None,
        tracker: Counter | None,
    ) -> SynonymEntry:
        """Prefer an uncovered name, then a regional name, then another attested spelling."""

        # Priority 1: coverage-first — pick uncovered synonyms first
        if tracker is not None:
            uncovered = [e for e in entries if tracker[e.dedup_key] == 0]
            if uncovered:
                # BUG 5 fix: even in coverage mode, prefer longer names
                long_uncovered = [e for e in uncovered if len(e.display_name) >= 4]
                if long_uncovered:
                    return rng.choice(long_uncovered)
                # All uncovered are short — still pick from them to maintain
                # coverage, but only when no long alternatives remain uncovered
                return rng.choice(uncovered)

        # Priority 2: regional preference (60% chance)
        if region and loinc in self._regional_map and region in self._regional_map[loinc]:
            if rng.random() < 0.60:
                regional_entries = self._regional_map[loinc][region]
                return rng.choice(regional_entries)

        # BUG 5 fix: prefer longer names (>=4 chars) to avoid ultra-short
        # abbreviations like "K+", "Hb", "Na" dominating the output
        entries_to_choose = entries
        if len(entries) > 1:
            long_entries = [e for e in entries if len(e.display_name) >= 4]
            if long_entries and rng.random() < 0.85:
                entries_to_choose = long_entries
            else:
                entries_to_choose = entries  # Fallback to ALL entries including short
        # Safety check: if filtering somehow left us empty, fall back to full list
        if not entries_to_choose:
            entries_to_choose = entries

        return rng.choice(entries_to_choose)

    @staticmethod
    def _apply_case(text: str, rng: random.Random) -> tuple[str, str]:
        """Apply case variation to text. Returns (modified_text, variant_name).

        FIX 6: mixed casing removed entirely -- it produces garbled names
        like "gLuCoSa" that corrupt training data.  Weights redistributed:
        Title 55%, UPPER 30%, lower 15%.
        """
        roll = rng.random()
        if roll < 0.55:
            return text.title(), "title"
        elif roll < 0.85:
            return text.upper(), "upper"
        else:
            return text.lower(), "lower"

    # ------------------------------------------------------------------
    # Coverage stats
    # ------------------------------------------------------------------

    def get_coverage_stats(self, tracker: Counter) -> dict[str, Any]:
        """Return coverage statistics."""
        all_keys = set()
        for entries in self.synonyms.values():
            for e in entries:
                all_keys.add(e.dedup_key)

        total = len(all_keys)
        covered = sum(1 for k in all_keys if tracker.get(k, 0) > 0)
        counts = [tracker.get(k, 0) for k in all_keys]

        return {
            "total_synonyms": total,
            "covered": covered,
            "uncovered": total - covered,
            "coverage_pct": (covered / total * 100) if total else 0.0,
            "min_count": min(counts) if counts else 0,
            "max_count": max(counts) if counts else 0,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_synonyms(self) -> int:
        """Total unique synonyms across all LOINCs (deduplicated within each LOINC)."""
        return sum(len(entries) for entries in self.synonyms.values())

    @property
    def total_loincs(self) -> int:
        """Total LOINCs with at least one synonym."""
        return len(self.synonyms)

    def synonyms_for(self, loinc: str) -> list[str]:
        """Return all synonym display names for a LOINC code."""
        return [e.display_name for e in self.synonyms.get(loinc, [])]

    def regions_for(self, loinc: str) -> list[str]:
        """Return all regions that have variants for a LOINC."""
        return list(self._regional_map.get(loinc, {}).keys())
