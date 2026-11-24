"""Pair OCR result literals with test anchors using word boxes in page pixels."""

import math
import re
import unicodedata
from dataclasses import dataclass
from statistics import median

DECIMAL = r"[+−-]?(?:[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+)"
COMPARATOR = r"(?:<=|>=|[<>=≤≥])?"
NUMBER = re.compile(DECIMAL)
RESULT = re.compile(COMPARATOR + DECIMAL)
# Textual and compound values can be located and referred for review.
QUALITATIVE = {
    "positivo",
    "negativo",
    "positive",
    "negative",
    "reactivo",
    "reactiva",
    "detectado",
    "detectada",
    "trazas",
    "indeterminado",
}
RESULT_HEADINGS = {
    "result",
    "results",
    "resultado",
    "resultados",
    "resultdo",
    "value",
    "valor",
}
NAME_HEADINGS = {"test", "prueba", "analito", "analyte", "examen", "determinacion"}
UNIT_HEADINGS = {"unidad", "unidades", "unid", "unit", "units", "nid"}
OTHER_HEADINGS = {
    "ref",
    "referencia",
    "reference",
    "valref",
    "range",
    "rango",
    "flag",
    "metodo",
    "method",
}
PAGE_WORDS = {"page", "pagina", "pag"}
COMPARISON_GLYPHS = "<>=≤≥≠≦≧≪≫≈~±"
COMPARISON_PHRASES = {
    ("less", "than"),
    ("greater", "than"),
    ("more", "than"),
    ("at", "least"),
    ("at", "most"),
    ("menor", "que"),
    ("mayor", "que"),
    ("inferior", "a"),
    ("superior", "a"),
}
# Unit evidence is optional on pages with a header or a repeated result column.
UNIT = re.compile(
    r"(?:[%‰]|(?:[a-zµμ]{1,8}|10[°^*]?[0-9]+)/[a-zµμ]{1,5}|fl|pg|mmol|mol|g|mg|ng|ml|dl|iu)",
    re.I,
)


@dataclass(frozen=True)
class Word:
    text: str
    left: float
    right: float
    x: float
    y: float
    h: float
    key: str


def _split_unit(text, left, right):
    """Apportion a single OCR box only when its suffix is a recognized unit."""
    value = RESULT.match(text)
    if value and UNIT.fullmatch(text[value.end() :]):
        cut = left + (right - left) * value.end() / len(text)
        return [(value.group(), left, cut), (text[value.end() :], cut, right)]
    return [(text, left, right)]


def _words(tokens):
    words = []
    for text, (left, top, right, bottom) in tokens:
        if not text or right <= left or bottom <= top:
            continue
        for part, start, end in _split_unit(text, left, right):
            key = "".join(
                c for c in unicodedata.normalize("NFKD", part.casefold()) if c.isalpha()
            )
            words.append(
                Word(
                    part,
                    start,
                    end,
                    (start + end) / 2,
                    (top + bottom) / 2,
                    bottom - top,
                    key,
                )
            )
    return sorted(words, key=lambda word: (word.y, word.x, word.text))


def _same_line(a, b):
    return abs(a.y - b.y) <= 0.65 * max(a.h, b.h)


def _headers(words, scale):
    """Use neighboring column labels to disambiguate prose mentioning results."""
    headers = []
    for result in words:
        if result.key not in RESULT_HEADINGS:
            continue
        line = [w for w in words if _same_line(result, w)]
        names = [w for w in line if w.x < result.x and w.key in NAME_HEADINGS]
        right = [
            w
            for w in line
            if w.x > result.x and w.key in UNIT_HEADINGS | OTHER_HEADINGS
        ]
        if not (names or right):
            continue
        # Templates can left-align column headings and right-align their data.
        left_limit = result.left - scale
        right_heading = min(right, key=lambda w: w.x, default=None)
        right_limit = right_heading.x if right_heading else math.inf
        next_column = right_heading.left if right_heading else math.inf
        headers.append((result.y, left_limit, right_limit, next_column))
    return headers


@dataclass(frozen=True)
class Pair:
    label: Word
    value: Word
    has_unit: bool
    column_gap: bool


def _candidate(word):
    return word.key in QUALITATIVE or (
        any(c.isdecimal() for c in word.text)
        and not word.text[0].isalpha()
        and not UNIT.fullmatch(word.text)
    )


def _pairs(words, scale):
    pairs = []
    for value in words:
        if not _candidate(value):
            continue
        line = [w for w in words if _same_line(value, w)]
        left = [
            w for w in line if w.key and w.x < value.x and w.right <= value.left + 1
        ]
        if not left:
            continue
        # Compare baselines first: a longer label on the previous row may end
        # nearer the number horizontally. Allow small height variation in words.
        dy = min(abs(w.y - value.y) for w in left)
        aligned = [w for w in left if abs(w.y - value.y) <= dy + 0.2 * scale]
        label = min(aligned, key=lambda w: w.x)
        # For whitespace, allow ordinary baseline differences (accented words
        # and capital letters can have different OCR box centers).
        adjacent = [w for w in left if abs(w.y - value.y) <= dy + 0.45 * scale]
        nearest = max(adjacent, key=lambda w: w.right)
        if nearest.key in PAGE_WORDS:
            continue
        right = min(
            (w for w in line if w is not value and w.left >= value.right - 1),
            key=lambda w: w.left,
            default=None,
        )
        label_gap = value.left - nearest.right
        unit = False
        if right and UNIT.fullmatch(right.text.rstrip(".,;:")):
            unit_gap = right.left - value.right
            # A separated unit cell also supports a sparse row if the label,
            # number and unit share a baseline and both gutters are visible.
            separate_cells = (
                3 * scale <= unit_gap <= label_gap
                and abs(nearest.y - value.y) <= 0.35 * min(nearest.h, value.h)
                and abs(right.y - value.y) <= 0.35 * min(right.h, value.h)
            )
            unit = unit_gap <= 3 * scale or separate_cells
        pairs.append(Pair(label, value, unit, label_gap >= scale))
    return pairs


def _column(pairs, tolerance, has_header):
    """Estimate the result column from distinct supporting rows.

    A unit gives a sparse headerless page positive evidence. With no such
    evidence require at least three rows and a unique dominant column.
    """
    if not pairs:
        return []
    options = []
    for seed in pairs:
        aligned = [p for p in pairs if abs(p.value.x - seed.value.x) <= tolerance]
        rows = []
        for pair in sorted(aligned, key=lambda p: p.value.y):
            if (
                rows
                and abs(pair.value.y - rows[-1][0].value.y)
                <= min(pair.value.h, rows[-1][0].value.h) / 2
            ):
                rows[-1].append(pair)
            else:
                rows.append([pair])
        selected = [
            min(
                row,
                key=lambda p: (
                    not p.has_unit,
                    abs(p.value.x - seed.value.x),
                    p.value.left,
                ),
            )
            for row in rows
        ]
        units = sum(p.has_unit for p in selected)
        # Actual measurements often adjoin a unit. Reference bounds or numbers
        # inside test names can also recur at a fixed x on several rows.
        rank = (len(selected) + 2 * units, units, len(selected))
        spread = sum(abs(p.value.x - seed.value.x) for p in selected)
        options.append((rank, spread, seed.value.x, selected))
    options.sort(key=lambda opt: (-opt[0][0], -opt[0][1], opt[1], opt[2]))
    rank, _, x, best = options[0]
    if not has_header:
        if rank[1] == 0 and rank[2] < 3:
            return []
        if any(r == rank and abs(cx - x) > 2 * tolerance for r, _, cx, _ in options):
            return []
    return best


def _is_mark(word):
    return all(not c.isalnum() for c in word.text)


def _possible_prefix(word):
    return all(c in COMPARISON_GLYPHS + "+−-" for c in word.text)


def _reading(pair, words, scale, left, right):
    """Assemble nearby result fragments and keep ambiguous readings for review."""
    value = pair.value
    # Start just after the analyte anchor. A prefix can fall left of the
    # estimated result band when the heading and values align differently.
    line = sorted(
        (
            w
            for w in words
            if w is value
            or (
                _same_line(w, value)
                and pair.label.right <= w.x <= right
                and (left <= w.x or _possible_prefix(w))
            )
        ),
        key=lambda w: (w.left, w.right),
    )
    i = line.index(value)
    parts = [value]
    for step, neighbors in [(-1, reversed(line[:i])), (1, iter(line[i + 1 :]))]:
        edge = value
        for word in neighbors:
            gap = edge.left - word.right if step < 0 else word.left - edge.right
            if word.text == value.text and gap < 0:
                # Some engines repeat the same word box on the same row.
                continue
            if UNIT.fullmatch(word.text) or not (
                _is_mark(word) or _candidate(word) or word.text in {"e", "E"}
            ):
                break
            if gap > 1.5 * scale:
                if step < 0 and _possible_prefix(word):
                    return " ".join(
                        [word.text] + [w.text for w in parts]
                    ), "ambiguous_result"
                break
            if step < 0:
                parts.insert(0, word)
            else:
                parts.append(word)
            edge = word

    observed = "".join(w.text for w in parts)
    # Only join an operator (or split sign) to one complete numeric token.
    # In particular, adjacent digit tokens may be a range or two results.
    numeric_parts = sum(any(c.isdecimal() for c in w.text) for w in parts)
    if numeric_parts == 1 and RESULT.fullmatch(observed):
        # A bare OCR number can survive even when intervening text prevents
        # attaching its comparator. Preserve that observed evidence for review.
        for j, prefix in enumerate(line[:i]):
            in_column = left != -math.inf and prefix.x >= left
            nearby = value.left - prefix.right <= 4 * scale
            if prefix in parts or not (in_column or nearby):
                continue
            phrase = (prefix.key, line[j + 1].key)
            if phrase in COMPARISON_PHRASES or any(
                c in COMPARISON_GLYPHS for c in prefix.text
            ):
                return " ".join(w.text for w in line[j : i + 1]), "ambiguous_result"
        return observed, None
    reason = "ambiguous_result" if len(parts) > 1 else "unsupported_result"
    return " ".join(w.text for w in parts), reason


def predict_page(tokens, width, height, page=1):
    """Return (predictions, review) from (text, [left, top, right, bottom]) words.

    Page numbers are one-based. Points use the original image's top-left
    origin. The output preserves decimal punctuation and comparator glyphs.
    """
    if not all(math.isfinite(n) and n > 0 for n in (width, height)):
        raise ValueError("page dimensions must be finite and positive")
    words = _words(tokens)
    if not words:
        return [], []
    scale = median(w.h for w in words)
    tolerance = max(1.5 * scale, 0.012 * width)
    heads = _headers(words, scale)
    pairs = _pairs(words, scale)
    # Recognize a generic page-number footer only when followed by a nearby
    # number and located in the lower portion of the page.
    footer = min(
        (
            w.y
            for w in words
            if w.key in PAGE_WORDS
            and w.y > 0.70 * height
            and any(
                NUMBER.fullmatch(n.text)
                and _same_line(w, n)
                and -1 <= n.left - w.right <= 3 * scale
                for n in words
            )
        ),
        default=height,
    )
    selected = []
    if heads:
        for (top, left, right, next_column), bottom in zip(
            heads, [h[0] for h in heads[1:]] + [footer]
        ):
            inside = [
                p
                for p in pairs
                if top < p.value.y < min(bottom, footer) and left <= p.value.x <= right
            ]
            for pair in _column(inside, tolerance, has_header=True):
                # Only text on the result side of the next printed heading
                # can expand this reading. Unit OCR can itself look numeric.
                end = next_column if pair.value.x < next_column else right
                selected.append((pair, left, end))
    else:
        # Without a heading, require a unit or visible space between label
        # and number. Adjacent title/ID suffixes can align with the table.
        possible = [
            p for p in pairs if p.value.y < footer and (p.has_unit or p.column_gap)
        ]
        selected = [
            (pair, -math.inf, math.inf)
            for pair in _column(possible, tolerance, has_header=False)
        ]
    predictions, review = [], []
    for pair, left, right in selected:
        text, reason = _reading(pair, words, scale, left, right)
        entry = {
            "page": page,
            "test_point": [pair.label.x, pair.label.y],
            "value_point": [pair.value.x, pair.value.y],
            "value_text": text,
        }
        if reason:
            review.append({**entry, "reason": reason})
        else:
            predictions.append(entry)
    return predictions, review
