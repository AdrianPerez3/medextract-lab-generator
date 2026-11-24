"""Check the generator's local Spanish print names against its packaged LOINC vocabulary."""

import json
from pathlib import Path

import pytest


RESOURCES = Path(__file__).parent / "resources"


def routine_names():
    return json.loads((RESOURCES / "routine_es_names.json").read_text(encoding="utf-8"))


def test_local_titles_cover_each_starter_observation_once():
    titles = routine_names()
    catalog = json.loads((RESOURCES / "loinc_2_82.json").read_bytes())
    assert catalog["version"] == "2.82"
    assert len(titles) == 32
    assert titles.keys() == catalog["terms"].keys()
    assert len({title.casefold() for title in titles.values()}) == len(titles)
    assert all(
        title
        and title.isprintable()
        and title == " ".join(title.split())
        and ":" not in title
        for title in titles.values()
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("718-7", "Hemoglobina"),
        ("785-6", "Hemoglobina corpuscular media (HCM)"),
        ("786-4", "Conc. de hemoglobina corpuscular media (CHCM)"),
        ("787-2", "Volumen corpuscular medio (VCM)"),
        ("788-0", "Amplitud de distribución eritrocitaria (RDW-CV)"),
        ("3094-0", "Nitrógeno ureico (BUN)"),
        ("2028-9", "Dióxido de carbono total"),
        ("17861-6", "Calcio total"),
        ("2093-3", "Colesterol total"),
        ("2085-9", "Colesterol HDL"),
        ("2089-1", "Colesterol LDL"),
        ("770-8", "Neutrófilos (%)"),
        ("736-9", "Linfocitos (%)"),
        ("5905-5", "Monocitos (%)"),
        ("713-8", "Eosinófilos (%)"),
        ("706-2", "Basófilos (%)"),
    ],
)
def test_distinct_observations_keep_their_print_identity(code, expected):
    assert routine_names()[code] == expected
