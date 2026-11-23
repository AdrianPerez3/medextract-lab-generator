"""Build the generator's compact, code-bound vocabulary from the LOINC 2.82 release.

Run with python -m scripts.lib.build_terminology --release /path/to/Loinc_2.82
Only generator input codes are included. Regional names come from official
same-code rows. RELATEDNAMES2 is a search index and never a display label.
"""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("resources") / "loinc_2_82.json"
TABLE = "LoincTable/Loinc.csv"
VARIANTS = {
    "spain": "AccessoryFiles/LinguisticVariants/esES12LinguisticVariant.csv",
    "mexico": "AccessoryFiles/LinguisticVariants/esMX28LinguisticVariant.csv",
    "argentina": "AccessoryFiles/LinguisticVariants/esAR7LinguisticVariant.csv",
}
DIGESTS = {
    TABLE: "6187d5b2fcd89398fba45073f706fd15e24d0563afb750d9fef533881442fe1d",
    VARIANTS["spain"]: "b54d815e80fa6ffca5f47a336ddf0a62fd581dce3a82c3a6fdf6f7d2abc5b46c",
    VARIANTS["mexico"]: "fc42ac2b1492a2f96d5a3c478487510cc836fd48244cc8beae9ab227a376ee42",
    VARIANTS["argentina"]: "293a34d8078eff5686c045cc1c46fddc68fbb9cb8e1d794fbe9d5366267c6355",
}


def names(*candidates):
    """Keep whole labels which fit the generator; never truncate qualifiers."""
    labels = []
    for text in candidates:
        text = text.strip()
        if not (0 < len(text) <= 80 and text.isprintable()):
            continue
        try:
            text.encode("cp1252")  # Portable across the three PDF Type 1 families.
        except UnicodeEncodeError:
            continue
        if text not in labels:
            labels.append(text)
    return labels


def display_key(label):
    if label.isascii():
        folded = label.casefold()
    else:
        decomposed = unicodedata.normalize("NFKD", label).casefold()
        folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    text = " ".join(folded.split())
    text = re.sub(r"\s*([,:;])\s*", r"\1", text)
    return re.sub(r"(?<=\w)\.(?=[,:;]|$)", "", text)


def record_owners(owners, candidates, code):
    """Keep one owner per normalized label, or None once two codes share it."""
    for label in candidates:
        if label.strip():
            key = display_key(label)
            if owners.setdefault(key, code) != code:
                owners[key] = None


def remove_ambiguous_names(terms, owners):
    for code, term in list(terms.items()):
        surviving = {}
        for region, labels in term["names"].items():
            unique = [s for s in labels if owners[display_key(s)] == code]
            if unique:
                surviving[region] = unique
        if surviving:
            term["names"] = surviving
        else:
            del terms[code]


def variant_candidates(row, single_timepoint):
    component = row["COMPONENT"].strip()
    # Preserve the dimensions that distinguish similarly named tests, e.g.
    # free/total hormones, MCH/MCHC, timed samples and challenge protocols.
    fields = ("PROPERTY", "SYSTEM", "METHOD_TYP") if single_timepoint else ("PROPERTY", "SYSTEM", "TIME_ASPCT", "METHOD_TYP")
    axes = [row[k].strip() for k in fields]
    qualified = ", ".join([component] + [s for s in axes if s and s != "-"]) if component else ""
    return row["LinguisticVariantDisplayName"], row["LONG_COMMON_NAME"], row["SHORTNAME"], qualified


def variant_names(row, single_timepoint):
    return names(*variant_candidates(row, single_timepoint))


def build(release, ranges, output):
    codes = set(json.loads(ranges.read_text()))
    for name, digest in DIGESTS.items():
        with (release / name).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"LOINC 2.82 source checksum mismatch: {name}")
    # Inspect other observable codes in the official release even when the
    # requested range pack is tiny: a small export must not disambiguate an alias.
    terms = {}
    spot_measurements = set()
    universe = set()
    owners = {}
    with (release / TABLE).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            code = r["LOINC_NUM"]
            if (r["STATUS"] != "ACTIVE" or r["CLASSTYPE"] != "1"
                    or r["ORDER_OBS"] not in {"Both", "Observation"} or r["PanelType"]):
                continue
            universe.add(code)
            if r["TIME_ASPCT"] == "Pt":
                spot_measurements.add(code)
            original = (r["SHORTNAME"], r["LONG_COMMON_NAME"], r["DisplayName"])
            record_owners(owners, original, code)
            if code not in codes or r["EXTERNAL_COPYRIGHT_NOTICE"] or r["EXTERNAL_COPYRIGHT_LINK"]:
                continue
            labels = names(*original)
            if labels:
                terms[code] = {"names": {"en": labels}, "units": sorted({
                    s.strip() for s in r["EXAMPLE_UCUM_UNITS"].split(";") if s.strip()
                }), "scale": r["SCALE_TYP"], "property": r["PROPERTY"]}
    for region, name in VARIANTS.items():
        with (release / name).open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                code = r["LOINC_NUM"]
                if code not in universe:
                    continue
                original = variant_candidates(r, code in spot_measurements)
                record_owners(owners, original, code)
                if code in terms:
                    labels = names(*original)
                    if labels:
                        terms[r["LOINC_NUM"]]["names"][region] = labels
    remove_ambiguous_names(terms, owners)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write('{"version":"2.82","source_sha256":')
        f.write(json.dumps(DIGESTS, sort_keys=True))
        f.write(',"terms":{\n')
        for i, (code, term) in enumerate(sorted(terms.items())):
            if i:
                f.write(",\n")
            f.write(json.dumps(code) + ":" + json.dumps(term, ensure_ascii=False, separators=(",", ":")))
        f.write("\n}}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--ranges", type=Path, default=ROOT / "src/paddle_ocr/dictionaries/data/reference_ranges.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.release, args.ranges, args.output)
