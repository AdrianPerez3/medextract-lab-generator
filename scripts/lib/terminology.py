"""Pinned, code-bound vocabulary for simulated lab-report observations."""
from functools import lru_cache
import json
import math
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("resources") / "loinc_2_82.json"


POLICY_PATH = CATALOG_PATH.with_name("simulation_policy.json")
ROUTINE_ES_NAMES_PATH = CATALOG_PATH.with_name("routine_es_names.json")


@lru_cache(maxsize=1)
def load_simulation_policy():
    with POLICY_PATH.open(encoding="utf-8") as stream:
        policy = json.load(stream)
    if (policy["loinc_version"], policy["positive"], policy["negative"]) != ("2.83", "LA6576-8", "LA6577-6"):
        raise ValueError("Unexpected binary-answer policy")
    return policy


def simulation_names(code, term):
    blocked = load_simulation_policy()["excluded_display_names"].get(code, ())
    return {locale: [label for label in labels if label not in blocked]
            for locale, labels in term["names"].items() if any(label not in blocked for label in labels)}


@lru_cache(maxsize=1)
def load_terms():
    with CATALOG_PATH.open(encoding="utf-8") as f:
        catalog = json.load(f)
    if catalog["version"] != "2.82":
        raise ValueError("Expected LOINC terminology version 2.82")
    return catalog["terms"]


# Only spelling equivalents. A different scale, prefix, or dimension requires
# different bounds and cannot be repaired by replacing the unit string.
_UNIT_SPELLINGS = {
    "mIU/L": "m[IU]/L", "mIU/mL": "m[IU]/mL", "IU/L": "[IU]/L",
    "UI/L": "[IU]/L", "mUI/mL": "m[IU]/mL", "{pH}": "[pH]", "{SG}": "{Spec grav}",
    "x 10^3/uL": "10*3/uL", "x 10^6/uL": "10*6/uL",
}


def unit_spelling(unit):
    spelling = unit.strip().replace("µ", "u").replace("μ", "u")
    return _UNIT_SPELLINGS.get(spelling, spelling)


def numeric_bounds(entry):
    lo, hi = entry.get("min"), entry.get("max")
    return (type(lo) in (int, float) and type(hi) in (int, float)
            and math.isfinite(lo) and math.isfinite(hi) and lo < hi)


def placeholder_bounds(entry):
    return (entry.get("source") in {"property_template", "constants"}
            and entry.get("min") == 0 and entry.get("max") in {10000, 100000})


def admitted_range(term, entry):
    """Return an eligible synthetic range or None when units/scale are unverified."""
    scale, prop = term["scale"], term["property"]
    if prop in {"Date", "TmStp", "TmStpRange"}:
        return None  # Dates and clock values require a temporal generator.
    if scale == "Ord" and term["property"] == "PrThr":
        # Only binary presence, not arbitrary ordinal classes (e.g. grade).
        return {"answers": ["Positivo", "Negativo"], "scale": scale, "unit": ""}
    if scale == "SemiQn" and term["property"] == "Titr":
        return {"type": "titre", "scale": scale, "unit": ""}
    if scale not in {"Qn", "SemiQn"} or not numeric_bounds(entry):
        return None
    unit = unit_spelling(entry.get("unit") or "")
    if placeholder_bounds(entry):
        return None
    if prop == "Likelihood" and unit not in {"%", "1"}:
        return None  # A printed risk ratio needs a defined representation.
    if prop in {"Likelihood", "NFr", "MFr", "SFr", "VFr", "CFr", "LenFr"} and unit in {"%", "1"}:
        ceiling = 100 if unit == "%" else 1
        if not 0 <= entry["min"] < entry["max"] <= ceiling:
            return None
    if unit and unit in term["units"] and (scale == "Qn" or term["property"] in {"SpGrav", "LsCnc"}):
        return {**entry, "unit": unit, "scale": scale}
    return None
