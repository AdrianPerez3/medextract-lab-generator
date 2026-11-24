"""Shuffle UTF-8 records by byte offset while preserving the original row text."""

from array import array
from collections.abc import Callable, Iterator
from io import TextIOWrapper
import os
from pathlib import Path
import random


def check_source_unchanged(path: Path, original: os.stat_result) -> None:
    current = path.stat()
    if (not os.path.samestat(original, current) or original.st_size != current.st_size
            or original.st_mtime_ns != current.st_mtime_ns):
        raise ValueError(f"Training source changed while packaging {path.name}")


def shuffled_rows(
    source: TextIOWrapper, seed: int, validate: Callable[[str, int], None] | None = None,
) -> tuple[int, Iterator[bytes]]:
    """Return the record count and a single-pass iterator. Open source with newline=''."""
    boundaries = array("Q", [0])
    order = array("Q")
    for line_number, line in enumerate(source, 1):
        boundaries.append(boundaries[-1] + len(line.encode("utf-8")))
        if line.strip():
            if validate is not None:
                validate(line, line_number)
            order.append(line_number - 1)
    random.Random(seed).shuffle(order)

    def rows():
        raw = source.buffer
        newline = os.linesep.encode("ascii")
        for index in order:
            start = boundaries[index]
            raw.seek(start)
            length = boundaries[index + 1] - start
            data = raw.read(length)
            if len(data) != length:
                raise ValueError("Training source changed while reading shuffled records")
            yield data.rstrip(b"\r\n") + newline

    return len(order), rows()
