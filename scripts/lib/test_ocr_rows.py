import unittest

import pytest

from scripts.lib.ocr_rows import predict_page


def word(text, x, y, width=36, height=10):
    return text, [x, y - height / 2, x + width, y + height / 2]


def header(y=100, offset=0):
    return [
        word("Prueba", 20 + offset, y),
        word("Resultado", 190 + offset, y, 65),
        word("Unidad", 325 + offset, y),
        word("Referencia", 400 + offset, y),
    ]


def row(name, value, y, unit="mg/dL", offset=0):
    parts = [word(name, 20 + offset, y, 74), word(value, 270 + offset, y, 30)]
    if unit:
        parts.append(word(unit, 307 + offset, y))
    return parts


class TestTableBoundary(unittest.TestCase):
    def predict(self, tokens):
        return predict_page(tokens, 600, 800, page=2)[0]

    def values(self, tokens):
        return [p["value_text"] for p in self.predict(tokens)]

    def test_header_masks_surrounding_metadata_name_numbers_and_reference_numbers(self):
        tokens = (
            row("Edad", "56", 40, "")
            + header()
            + row("Basófilos", "0,19", 150)
            + [
                word("100", 115, 150),
                word("leucocitos", 145, 150),
                word("2", 415, 150),
                word("Página", 16, 740),
                word("2", 57, 740),
                word("Documento", 18, 760),
                word("2026", 271, 760),
            ]
        )
        self.assertEqual(
            self.predict(tokens),
            [
                {
                    "page": 2,
                    "value_text": "0,19",
                    "test_point": [57, 150],
                    "value_point": [285, 150],
                }
            ],
        )

    def test_recognized_header_supports_unitless_single_row(self):
        self.assertEqual(self.values(header() + row("A", "-2,7", 125, "")), ["-2,7"])

    def test_single_headerless_row_needs_unit_evidence(self):
        self.assertEqual(self.values(row("A", "6.1", 148)), ["6.1"])
        self.assertEqual(self.values(row("A", "6.1", 148, "")), [])
        self.assertEqual(self.values([]), [])

    def test_headerless_sparse_separate_unit_column_needs_three_aligned_cells(self):
        name = word("Analyte", 20, 152, 55)
        number = word("6.4", 270, 152, 30)
        unit = word("mg/dL", 425, 152)
        self.assertEqual(self.values([name, number, unit]), ["6.4"])
        self.assertEqual(self.values([word("Volume", 224, 152), number, unit]), [])
        self.assertEqual(self.values([name, number, word("mg/dL", 425, 158)]), [])

    def test_headerless_repeated_unitless_column_survives_incidental_unit_elsewhere(
        self,
    ):
        tokens = [word("Specimen", 20, 60), word("1", 130, 60, 10), word("mL", 148, 60)]
        for i in range(4):
            tokens += row("Analyte", str(10 + i), 150 + 30 * i, "")
        self.assertEqual(self.values(tokens), ["10", "11", "12", "13"])

    def test_headerless_prose_number_beside_title_and_aligned_footer(self):
        tokens = [word("Facility", 190, 60, 78), word("42", 270, 60)]
        tokens += row("A", "2", 135) + row("B", "3", 165, "") + row("C", "4", 195)
        tokens += [word("Page", 225, 730), word("1", 270, 730)]
        self.assertEqual(self.values(tokens), ["2", "3", "4"])

    def test_headerless_equally_supported_separate_columns_abstain(self):
        tokens = []
        for i in range(3):
            y = 100 + 30 * i
            tokens += row("A", str(i + 1), y, "") + [word("9", 410, y)]
        self.assertEqual(self.values(tokens), [])

    def test_prefer_words_at_the_correct_row_height_and_allow_wrapping(self):
        tokens = header() + [
            word("RightmostPrevious", 140, 132, 90, 24),
            word("Analyte", 20, 146),
            word("desc", 95, 146),
            word("6.0", 270, 146, 30, 28),
            word("mg/dL", 307, 146),
            word("continued", 20, 162),
        ]
        self.assertEqual(self.predict(tokens)[0]["test_point"], [38, 146])

    def test_new_heading_can_move_columns_and_row_alignment(self):
        tokens = (
            header()
            + row("First", "100", 148)
            + header(302, 60)
            + row("Last", "200", 359, offset=60)
        )
        self.assertEqual(self.values(tokens), ["100", "200"])

    def test_duplicate_tokens_on_one_line_count_as_one_row(self):
        tokens = header() + row("A", "8", 150) + [word("8", 271, 150, 28)]
        self.assertEqual(self.values(tokens), ["8"])

    def test_preserves_first_and_last_unitless_rows_with_column_spacing(self):
        tokens = (
            row("A", "101", 145, "") + row("B", "102", 180) + row("C", "103", 215, "")
        )
        self.assertEqual(self.values(tokens), ["101", "102", "103"])

    def test_input_order_and_resolution(self):
        tokens = header() + row("A", "8", 150) + row("B", "9", 180)
        scaled = [
            (t, [coordinate * 2 for coordinate in box]) for t, box in reversed(tokens)
        ]
        expected = self.predict(tokens)
        for p in expected:
            for key in ("test_point", "value_point"):
                p[key] = [v * 2 for v in p[key]]
        self.assertEqual(predict_page(scaled, 1200, 1600, 2)[0], expected)


@pytest.mark.parametrize("symbol", ["<", ">", "≤", "≥", "<=", ">="])
def test_joined_and_split_comparators_preserve_the_ocr_characters(symbol):
    # Run both tokenizations, including a decimal comma and unit less than one.
    for tokens in [
        row("Analito", symbol + "0,1", 150),
        row("Analito", "0,1", 150) + [word(symbol, 254, 150, 12)],
    ]:
        predictions, review = predict_page(header() + tokens, 600, 800)
        assert review == []
        assert [p["value_text"] for p in predictions] == [symbol + "0,1"]


def test_split_ascii_equal_and_number_unit_are_kept_together():
    parts = [
        word("Analito", 20, 150),
        word(">", 248, 150, 8),
        word("=", 257, 150, 8),
        word("-0.50mg/dL", 270, 150, 70),
    ]
    predictions, review = predict_page(header() + parts, 600, 800)
    assert review == []
    (result,) = predictions
    assert result["value_text"] == ">=-0.50"
    assert result["value_point"] == [287.5, 150]


@pytest.mark.parametrize("printed", ["1:80", "Negativo", "1.2.3", "<<0,1", "1e3"])
def test_unsupported_result_is_reported_once(printed):
    predictions, review = predict_page(
        header() + row("Analito", printed, 150), 600, 800
    )
    assert predictions == []
    assert len(review) == 1
    assert review[0]["value_text"] == printed


@pytest.mark.parametrize(
    "pieces",
    [
        [("<", 200), ("0,1", 270)],  # unrelated or too distant to attach
        [("~", 259), ("0,1", 270)],  # unsupported qualifier
        [("1", 254), (":", 263), ("80", 272)],
        [("5", 254), ("-", 263), ("10", 272)],
        [("<", 244), (">", 254), ("0,1", 270)],
        [("0,1", 263), ("<", 277)],
    ],
)
def test_ambiguous_fragments_never_yield_a_bare_numeric_value(pieces):
    tokens = header() + [word("Analito", 20, 150)]
    tokens += [word(text, x, 150, 8) for text, x in pieces]
    predictions, review = predict_page(tokens, 600, 800)
    assert predictions == []
    assert len(review) == 1
    assert review[0]["reason"] == "ambiguous_result"


def test_comparator_does_not_leak_between_result_and_reference_or_between_rows():
    tokens = (
        header()
        + row("Primero", "3", 150)
        + row("Segundo", "3", 175)
        + [word("<", 263, 175, 5), word("<", 409, 150, 5), word("9", 416, 150, 5)]
    )
    predictions, review = predict_page(tokens, 600, 800)
    assert [p["value_text"] for p in predictions] == ["3", "<3"]
    assert review == []


@pytest.mark.parametrize("printed", ["12,2%", "12,2", "≥12,2%"])
def test_percentage_unit_ends_the_result(printed):
    tokens = row("Hematocrito", printed, 150, "%" if not printed.endswith("%") else "")
    predictions, review = predict_page(header() + tokens, 600, 800)
    assert [p["value_text"] for p in predictions] == [printed.removesuffix("%")]
    assert review == []


def test_far_comparator_on_headerless_row_is_referred():
    tokens = row("Analito", "0,1", 150) + [word("<", 207, 150, 8)]
    predictions, review = predict_page(tokens, 600, 800)
    assert predictions == []
    assert [item["reason"] for item in review] == ["ambiguous_result"]


def test_comparator_straddling_the_left_edge_of_result_header():
    # The header begins at x=190. Its inferred result band starts at 180.
    tokens = header() + [
        word("Analito", 20, 150),
        word("≤", 174, 150, 8),
        word("2,7", 184, 150, 22),
    ]
    predictions, review = predict_page(tokens, 600, 800)
    assert [p["value_text"] for p in predictions] == ["≤2,7"]
    assert review == []


def test_thin_touching_boxes_do_not_crash_or_form_a_self_pair():
    tokens = header() + [word("Analito", 205, 150, 25), word("7", 229, 150, 1)]
    predictions, review = predict_page(tokens, 600, 800)
    assert [p["value_text"] for p in predictions] == ["7"]
    assert review == []
    assert predict_page(header() + [word("Negativo", 229, 150, 1)], 600, 800) == (
        [],
        [],
    )


def test_a_distant_parenthesized_percent_belongs_to_the_name():
    tokens = row("Monocitos", "25,4", 150, "%") + [word("(%)", 108, 150, 17)]
    predictions, review = predict_page(tokens, 600, 800)
    assert [p["value_text"] for p in predictions] == ["25,4"]
    assert review == []


def test_noisy_token_under_the_unit_heading_is_outside_the_value_cell():
    tokens = header() + [
        word("Analito", 20, 150),
        word("3,14", 290, 150, 33),
        word("3", 327, 150, 6),
    ]
    predictions, review = predict_page(tokens, 600, 800)
    assert [p["value_text"] for p in predictions] == ["3,14"]
    assert review == []


@pytest.mark.parametrize(
    "prefix",
    [("Less", "than"), ("Menor", "que"), ("Greater", "than"), ("Mayor", "que")],
)
def test_spelled_comparison_is_referred_only_in_result_column(prefix):
    def words(x):
        return [word(prefix[0], x, 150, 20), word(prefix[1], x + 23, 150, 18)]

    tokens = header() + row("Analito", "0,1", 150)
    predictions, review = predict_page(tokens + words(220), 600, 800)
    assert predictions == []
    assert [r["value_text"] for r in review] == [" ".join((*prefix, "0,1"))]
    predictions, review = predict_page(tokens + words(104), 600, 800)
    assert [r["value_text"] for r in predictions] == ["0,1"]
    assert review == []


@pytest.mark.parametrize("noise", ["O", "ND<"])
def test_an_observed_comparator_with_ocr_noise_never_returns_a_bare_number(noise):
    tokens = (
        header()
        + row("Analito", "0,1", 150)
        + [word("<", 246, 150, 6), word(noise, 257, 150, 9)]
    )
    predictions, review = predict_page(tokens, 600, 800)
    assert predictions == []
    assert len(review) == 1 and "<" in review[0]["value_text"]


def test_split_exponent_is_referred_as_one_expression():
    tokens = header() + [
        word("Analito", 20, 150),
        word("1.2", 250, 150, 17),
        word("E", 269, 150, 8),
        word("-3", 280, 150, 11),
    ]
    predictions, review = predict_page(tokens, 600, 800)
    assert predictions == []
    assert [r["value_text"] for r in review] == ["1.2 E -3"]
