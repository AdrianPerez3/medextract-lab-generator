"""Make clean and simulated photocopy views of the same coordinate-annotated pages."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

import numpy as np
from PIL import Image, ImageFilter

if __package__:
    from .lib.dataset_integrity import json_lines
    from .lib.dataset_output import file_sha256, staged_output
    from .score_row_associations import _prepare_pages
else:
    from lib.dataset_integrity import json_lines
    from lib.dataset_output import file_sha256, staged_output
    from score_row_associations import _prepare_pages


def paired_page_seed(seed: int, pair_id: str, page_number: int) -> int:
    """Derive a stable page seed independent of source order and sibling pages."""
    key = json.dumps([seed, pair_id, page_number], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def write_photocopy(source: Image.Image, destination: Path, *, seed: int) -> None:
    """Simulate resampling, blur, uneven illumination, noise and JPEG compression on a fixed canvas."""
    settings = random.Random(seed)
    fraction = settings.choice((0.59, 0.65, 0.72))
    blur = settings.choice((0.28, 0.38, 0.48))
    quality = settings.choice((61, 68, 74))
    small = tuple(max(1, round(d * fraction)) for d in source.size)
    gray = source.convert("L").resize(small, Image.Resampling.LANCZOS)
    scan = gray.filter(ImageFilter.GaussianBlur(blur)).resize(
        source.size, Image.Resampling.BILINEAR
    )
    noise = np.random.default_rng(seed)
    gradient = np.linspace(0.0, 4.0, scan.height, dtype=np.float32)
    # Keep noise and float intermediates to a narrow strip even for 600 DPI inputs.
    for top in range(0, scan.height, 128):
        bottom = min(top + 128, scan.height)
        pixels = np.array(scan.crop((0, top, scan.width, bottom)), dtype=np.float32)
        pixels *= 0.88
        pixels += (22.0 - gradient[top:bottom])[:, None]
        pixels += noise.standard_normal(pixels.shape, dtype=np.float32) * 1.9
        np.clip(pixels, 0, 255, out=pixels)
        scan.paste(Image.fromarray(pixels.astype(np.uint8)), (0, top))
    save_options = {"dpi": source.info["dpi"]} if "dpi" in source.info else {}
    scan.save(destination, format="JPEG", quality=quality, **save_options)


def _input_path(root, relative, folder):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError(f"Invalid {folder} path")
    parts = relative.split("/")
    if len(parts) != 2 or parts[0] != folder or parts[1] in ("", ".", ".."):
        raise ValueError(f"Expected {folder}/<filename>")
    path = (root / relative).resolve()
    if path.parent != root / folder:
        raise ValueError(f"Source path escapes {folder}")
    return path


def _bind_digest(actual, expected, name):
    if actual != expected:
        raise ValueError(f"{name}: missing or mismatched SHA-256")


def _write_images(root, dest, pair_id, page, source_name, digests, seed):
    """Write verified clean bytes and their photocopy, returning both image receipts."""
    source = _input_path(root, source_name, "images")
    number = page["page"]
    info = page["image"]
    if info["path"] != source_name:
        raise ValueError(
            f"{pair_id}: page {number} image path disagrees with coordinates"
        )
    ext = source.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"{source_name}: expected a JPEG or PNG")

    clean_name = f"images/{pair_id}_clean_p{number:02}{ext}"
    clean = dest / clean_name
    shutil.copyfile(source, clean)
    clean_digest = file_sha256(clean)
    _bind_digest(clean_digest, digests.get(source_name), source_name)
    if "sha256" in info:
        _bind_digest(clean_digest, info["sha256"], source_name)

    copy_name = f"images/{pair_id}_photocopy_p{number:02}.jpg"
    with Image.open(clean) as image:
        if (
            image.size != (info["width"], info["height"])
            or getattr(image, "n_frames", 1) != 1
            or image.format != ("PNG" if ext == ".png" else "JPEG")
        ):
            raise ValueError(
                f"{source_name}: raster dimensions or format disagree with source"
            )
        # Read verified staging bytes, even if the input file changes concurrently.
        write_photocopy(
            image, dest / copy_name, seed=paired_page_seed(seed, pair_id, number)
        )
    return {
        "clean": dict(info, path=clean_name, sha256=clean_digest),
        "photocopy": dict(info, path=copy_name, sha256=file_sha256(dest / copy_name)),
    }


def _write_pair(root, dest, record, seed):
    pair_id = record.get("report_id")
    if not isinstance(pair_id, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", pair_id
    ):
        raise ValueError("Expected a filename-safe report_id")
    # The generator marks clean JPEGs with tier 0; fixed benchmark sources use condition.
    tier = record.get("tier", 0 if record.get("condition") == "clean" else None)
    if (
        type(tier) is not int
        or tier != 0
        or record.get("condition", "clean") != "clean"
        or not record.get("images")
        or not record.get("coordinates")
    ):
        raise ValueError(
            f"{pair_id}: expected clean tier-0 page images and row/cell coordinates"
        )
    digests = record.get("sha256")
    if not isinstance(digests, dict):
        raise ValueError(f"{pair_id}: missing source artifact SHA-256 values")
    source_truth = record["ground_truth"]
    truth = _input_path(root, source_truth, "ground_truth").read_bytes()
    truth_hash = hashlib.sha256(truth).hexdigest()
    _bind_digest(truth_hash, digests.get(source_truth), source_truth)
    truth_name = f"ground_truth/{pair_id}.json"
    with (dest / truth_name).open("xb") as f:
        f.write(truth)
    coord_name = record["coordinates"]
    geometry = _input_path(root, coord_name, "coordinates").read_bytes()
    _bind_digest(
        hashlib.sha256(geometry).hexdigest(), digests.get(coord_name), coord_name
    )
    sidecar = json.loads(geometry)
    if not isinstance(sidecar, dict):
        raise ValueError(f"{pair_id}: coordinate sidecar must be an object")
    _prepare_pages(sidecar, truth, "image-pixels")
    pages = sidecar["pages"]
    if (
        sidecar["report_id"] != pair_id
        or type(record["pages"]) is not int
        or record["pages"] != len(pages)
        or not isinstance(record["images"], list)
        or len(record["images"]) != len(pages)
        or len(set(record["images"])) != len(pages)
    ):
        raise ValueError(
            f"{pair_id}: input page manifest disagrees with coordinate sidecar"
        )

    image_pairs = [
        _write_images(root, dest, pair_id, page, source_name, digests, seed)
        for page, source_name in zip(pages, record["images"], strict=True)
    ]
    fields = {
        k: record[k]
        for k in ("family", "case_mix", "decimal_mark", "source_dpi")
        if k in record
    }
    views = []
    for condition in ("clean", "photocopy"):
        view_id = f"{pair_id}_{condition}"
        view = dict(
            fields,
            report_id=view_id,
            pair_id=pair_id,
            group_id=record.get("group_id", pair_id),
            split=record.get("split", "test"),
            condition=condition,
            pages=len(pages),
            images=[],
            ground_truth=truth_name,
            coordinates=f"coordinates/{view_id}.json",
            sha256={truth_name: truth_hash},
        )
        view_pages = []
        for page, image_pair in zip(pages, image_pairs, strict=True):
            image = image_pair[condition]
            view["images"].append(image["path"])
            view["sha256"][image["path"]] = image["sha256"]
            view_pages.append({**page, "image": image})
        view_sidecar = {**sidecar, "report_id": view_id, "pages": view_pages}
        coord_path = dest / view["coordinates"]
        coord_path.write_text(
            json.dumps(
                view_sidecar, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        view["sha256"][view["coordinates"]] = file_sha256(coord_path)
        views.append(view)
    return views


def make_paired_views(source: Path, destination: Path, *, seed: int) -> dict:
    """Publish a benchmark directory grouped by underlying report, staging all outputs first."""
    source = source.resolve()
    # An output subdirectory is fine, unless it would contaminate the source artifact inventory.
    resolved_output = destination.resolve()
    if resolved_output.is_relative_to(source):
        parts = resolved_output.relative_to(source).parts
        if not parts or parts[0] in {
            "coordinates",
            "ground_truth",
            "normalized_ground_truth",
            "images",
            "pdfs",
        }:
            raise ValueError(
                "Place paired output outside the source artifact directories"
            )
    with staged_output(destination) as stage:
        return write_paired_views(source, stage, seed=seed)


def write_paired_views(source: Path, stage: Path, *, seed: int) -> dict:
    """Write into a private, empty staging area owned by the calling entry point."""
    source = source.resolve()
    seen = set()
    splits = {}
    page_count = 0
    for name in ("ground_truth", "coordinates", "images"):
        (stage / name).mkdir()
    with (stage / "reports.jsonl").open("x", encoding="utf-8") as manifest:
        for record in json_lines(source / "reports.jsonl"):
            rid = record.get("report_id")
            if not isinstance(rid, str) or rid in seen:
                raise ValueError("Missing or duplicate source report ID")
            seen.add(rid)
            group = record.get("group_id", rid)
            split = record.get("split", "test")
            if (
                not isinstance(group, str)
                or not group.strip()
                or not isinstance(split, str)
                or not split.strip()
            ):
                raise ValueError(f"{rid}: expected nonempty group_id and split")
            if splits.setdefault(group, split) != split:
                raise ValueError(f"{rid}: a source group spans multiple splits")
            for view in _write_pair(source, stage, record, seed):
                manifest.write(
                    json.dumps(view, ensure_ascii=False, allow_nan=False) + "\n"
                )
                page_count += view["pages"]
    if not seen:
        raise ValueError("The source manifest contains no reports")
    summary = {
        "recipe": "photocopy-v1",
        "seed": seed,
        "pairs": len(seen),
        "views": 2 * len(seen),
        "images": page_count,
    }
    (stage / "pairing.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Generator directory with --coordinates --format jpg --tiers 1,0,0,0,0",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="New paired benchmark directory. POSIX also permits an existing empty directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Photocopy noise and resampling seed (default: 42)",
    )
    args = parser.parse_args(argv)
    try:
        stats = make_paired_views(args.source, args.output, seed=args.seed)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(
        f"{stats['pairs']} report pairs, {stats['views']} views, {stats['images']} page images"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
