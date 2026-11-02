from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path, PureWindowsPath

PERSON_NAMES = {"person", "people", "人物"}


def basename(value: str) -> str:
    return PureWindowsPath(str(value)).name


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a leakage-free 15temple COCO split.")
    parser.add_argument("--source-json", default="data/15temple/result.json")
    parser.add_argument("--image-dir", default="data/15temple/images")
    parser.add_argument("--output-dir", default="runs/15temple_person_maskrcnn")
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--test-count", type=int, default=138)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_path = Path(args.source_json)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    test_image_dir = out_dir / "test_images_138"
    out_dir.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    person_ids = {c["id"] for c in source.get("categories", []) if str(c.get("name", "")).strip().lower() in PERSON_NAMES}
    if not person_ids:
        raise ValueError("No person/people category found")

    image_by_original_id = {int(item["id"]): item for item in source["images"]}
    canonical_id_by_name: dict[str, int] = {}
    canonical_images: dict[int, dict] = {}
    original_to_canonical: dict[int, int] = {}
    for original_id, image in sorted(image_by_original_id.items()):
        name = basename(image["file_name"])
        if name not in canonical_id_by_name:
            canonical_id_by_name[name] = original_id
            canonical_images[original_id] = {
                **image,
                "id": original_id,
                "file_name": name,
            }
        original_to_canonical[original_id] = canonical_id_by_name[name]

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in source.get("annotations", []):
        original_image_id = int(annotation["image_id"])
        canonical_id = original_to_canonical.get(original_image_id)
        if canonical_id is None or annotation.get("category_id") not in person_ids or not annotation.get("segmentation"):
            continue
        clean = dict(annotation)
        clean["image_id"] = canonical_id
        clean["category_id"] = 1
        clean["iscrowd"] = int(clean.get("iscrowd", 0))
        annotations_by_image[canonical_id].append(clean)

    all_file_ids = sorted(
        image_id for image_id, image in canonical_images.items()
        if (image_dir / image["file_name"]).exists()
    )
    labeled_ids = [image_id for image_id in all_file_ids if annotations_by_image[image_id]]
    unlabeled_ids = [image_id for image_id in all_file_ids if not annotations_by_image[image_id]]
    expected = args.train_count + args.test_count
    if len(all_file_ids) != expected:
        raise ValueError(f"Expected exactly {expected} image files, found {len(all_file_ids)}")
    if len(labeled_ids) < args.train_count:
        raise ValueError(f"Need {args.train_count} labeled training images, found {len(labeled_ids)}")

    rng = random.Random(args.seed)
    rng.shuffle(labeled_ids)
    train_ids = set(labeled_ids[:args.train_count])
    test_ids = set(labeled_ids[args.train_count:]) | set(unlabeled_ids)
    if len(test_ids) != args.test_count:
        raise ValueError(f"Expected {args.test_count} test images, found {len(test_ids)}")

    def subset(ids: set[int]) -> dict:
        images = [canonical_images[i] for i in sorted(ids)]
        annotations = []
        for image_id in sorted(ids):
            annotations.extend(annotations_by_image[image_id])
        for new_id, annotation in enumerate(annotations, 1):
            annotation["id"] = new_id
        return {
            "info": source.get("info", {}),
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "person"}],
        }

    all_coco = subset(set(all_file_ids))
    train_coco = subset(train_ids)
    test_coco = subset(test_ids)
    paths = {
        "normalized": out_dir / "coco_person_all_1338.json",
        "train": out_dir / "coco_person_train_1200.json",
        "test": out_dir / "coco_person_test_138.json",
    }
    for key, payload in [("normalized", all_coco), ("train", train_coco), ("test", test_coco)]:
        paths[key].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    test_image_dir.mkdir(parents=True, exist_ok=True)
    for image_id in sorted(test_ids):
        name = canonical_images[image_id]["file_name"]
        shutil.copy2(image_dir / name, test_image_dir / name)

    with (out_dir / "split_1200_138.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "file_name", "split"])
        writer.writeheader()
        for split, ids in [("train", train_ids), ("test", test_ids)]:
            for image_id in sorted(ids):
                writer.writerow({"image_id": image_id, "file_name": canonical_images[image_id]["file_name"], "split": split})

    summary = {
        "source_image_records": len(source.get("images", [])),
        "unique_image_files": len(canonical_images),
        "duplicate_image_records_merged": len(source.get("images", [])) - len(canonical_images),
        "person_category_ids": sorted(person_ids),
        "person_instances": len(all_coco["annotations"]),
        "unlabeled_test_images": len(unlabeled_ids),
        "train_images": len(train_ids),
        "train_instances": len(train_coco["annotations"]),
        "test_images": len(test_ids),
        "test_instances": len(test_coco["annotations"]),
        "seed": args.seed,
        "test_image_folder": str(test_image_dir),
        "train_json": str(paths["train"]),
        "test_json": str(paths["test"]),
    }
    (out_dir / "dataset_preparation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
