from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "weights/maskrcnn_person_latest.pth": "813131d3ea54835e3a13fbe5a7dd9eb457ee54587823057178cf1711d428bef2",
    "weights/sam2_mural_best.pt": "7309812352317f82507a6b5a13ea1718b0ae8014b03779f419421098054d7f44",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-weight-hash", action="store_true")
    args = parser.parse_args()
    errors = []

    coco_path = ROOT / "data/test_138/annotations/person_test_coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    image_names = {item["file_name"] for item in coco["images"]}
    actual_names = {path.name for path in (ROOT / "data/test_138/images").glob("*.jpg")}
    if len(coco["images"]) != 138 or len(coco["annotations"]) != 422:
        errors.append("Expected 138 test images and 422 annotations")
    if image_names != actual_names:
        errors.append("COCO image list does not match data/test_138/images")

    sample_images = sorted((ROOT / "data/25temple_public_samples").glob("*/sample_grayscale.jpg"))
    if len(sample_images) != 25:
        errors.append(f"Expected 25 public samples, found {len(sample_images)}")
    for path in sample_images:
        with Image.open(path) as image:
            if image.mode != "L" or max(image.size) > 1024:
                errors.append(f"Invalid grayscale derivative: {path.relative_to(ROOT)}")

    if not args.skip_weight_hash:
        for relative, expected in EXPECTED.items():
            path = ROOT / relative
            if not path.exists() or sha256(path) != expected:
                errors.append(f"Weight checksum mismatch: {relative}")

    forbidden = re.compile(r"(?:[A-Za-z]:\\|C:/Users/|D:/codex learning/|api[_-]?key|password\s*=)", re.I)
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv", ".py", ".yml", ".yaml", ".txt"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if forbidden.search(text):
                errors.append(f"Potential private path or secret in {path.relative_to(ROOT)}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"status": "ok", "test_images": 138, "test_annotations": 422, "temple_samples": 25}, indent=2))


if __name__ == "__main__":
    main()
