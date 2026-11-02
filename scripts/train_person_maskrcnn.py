from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn


PERSON_CATEGORY_NAMES = {"person", "people", "人物", "人"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Mask R-CNN on the project COCO person masks.")
    parser.add_argument("--coco-json", default="coco/result.json")
    parser.add_argument("--image-dir", default="coco/images")
    parser.add_argument("--output-dir", default="runs/person_maskrcnn_v1")
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=0, help="Debug only. 0 means use all train images.")
    parser.add_argument("--max-test-images", type=int, default=0, help="Debug only. 0 means use all test images.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(Path("models/torch_cache").resolve()))
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    source = load_coco(args.coco_json)
    prepared = prepare_person_coco_split(
        source,
        image_dir=Path(args.image_dir),
        out_dir=out_dir,
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    print(json.dumps(prepared["summary"], ensure_ascii=False, indent=2))

    train_dataset = CocoPersonDataset(prepared["train_json"], args.image_dir, max_images=args.max_train_images)
    test_dataset = CocoPersonDataset(prepared["test_json"], args.image_dir, max_images=args.max_test_images)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"train_images={len(train_dataset)} test_images={len(test_dataset)}")

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, epoch)
        lr_scheduler.step()
        ckpt_path = out_dir / f"maskrcnn_person_epoch_{epoch:03d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "train_stats": train_stats,
            },
            ckpt_path,
        )
        row = {
            "epoch": epoch,
            "checkpoint": str(ckpt_path),
            "seconds": round(time.time() - epoch_start, 2),
            **train_stats,
        }
        history.append(row)
        (out_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False))

    metrics = evaluate_model(
        model,
        test_loader,
        device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = out_dir / "maskrcnn_person_latest.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "args": vars(args),
            "test_metrics": metrics,
        },
        latest_path,
    )
    print("test_metrics")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"latest_checkpoint={latest_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_coco(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_person_coco_split(
    coco: dict[str, Any],
    image_dir: Path,
    out_dir: Path,
    train_count: int,
    test_count: int,
    seed: int,
) -> dict[str, Any]:
    categories = coco.get("categories", [])
    person_cat_ids = {
        cat["id"]
        for cat in categories
        if str(cat.get("name", "")).strip().lower() in PERSON_CATEGORY_NAMES
    }
    if not person_cat_ids:
        raise ValueError(f"No person category found. Categories: {categories}")

    images_by_id = {img["id"]: dict(img) for img in coco.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in person_cat_ids and ann.get("segmentation"):
            anns_by_image[ann["image_id"]].append(dict(ann))

    usable_image_ids = [
        image_id
        for image_id, anns in anns_by_image.items()
        if image_id in images_by_id and anns and (image_dir / image_basename(images_by_id[image_id]["file_name"])).exists()
    ]
    usable_image_ids = sorted(set(usable_image_ids))
    rng = random.Random(seed)
    rng.shuffle(usable_image_ids)

    needed = train_count + test_count
    if len(usable_image_ids) < needed:
        raise ValueError(f"Need {needed} usable images, found {len(usable_image_ids)}")

    train_ids = set(usable_image_ids[:train_count])
    test_ids = set(usable_image_ids[train_count:needed])
    unused_ids = set(usable_image_ids[needed:])

    train_json = out_dir / "coco_person_train_300.json"
    test_json = out_dir / "coco_person_test_60.json"
    unused_json = out_dir / "coco_person_unused_holdout.json"
    split_csv = out_dir / "split_300_60.csv"

    write_subset_coco(coco, images_by_id, anns_by_image, train_ids, train_json)
    write_subset_coco(coco, images_by_id, anns_by_image, test_ids, test_json)
    write_subset_coco(coco, images_by_id, anns_by_image, unused_ids, unused_json)
    write_split_csv(images_by_id, train_ids, test_ids, unused_ids, split_csv)

    summary = {
        "source_images": len(coco.get("images", [])),
        "source_annotations": len(coco.get("annotations", [])),
        "person_category_ids": sorted(person_cat_ids),
        "usable_person_images": len(usable_image_ids),
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "unused_holdout_images": len(unused_ids),
        "train_annotations": sum(len(anns_by_image[i]) for i in train_ids),
        "test_annotations": sum(len(anns_by_image[i]) for i in test_ids),
        "unused_annotations": sum(len(anns_by_image[i]) for i in unused_ids),
        "train_json": str(train_json),
        "test_json": str(test_json),
        "split_csv": str(split_csv),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train_json": train_json, "test_json": test_json, "summary": summary}


def write_subset_coco(
    source: dict[str, Any],
    images_by_id: dict[int, dict[str, Any]],
    anns_by_image: dict[int, list[dict[str, Any]]],
    image_ids: set[int],
    output_path: Path,
) -> None:
    images = []
    annotations = []
    ann_id = 1
    for image_id in sorted(image_ids):
        img = dict(images_by_id[image_id])
        img["file_name"] = image_basename(img["file_name"])
        images.append(img)
        for ann in anns_by_image[image_id]:
            clean_ann = dict(ann)
            clean_ann["id"] = ann_id
            clean_ann["category_id"] = 1
            clean_ann["iscrowd"] = int(clean_ann.get("iscrowd", 0))
            annotations.append(clean_ann)
            ann_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
        "info": source.get("info", {}),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split_csv(
    images_by_id: dict[int, dict[str, Any]],
    train_ids: set[int],
    test_ids: set[int],
    unused_ids: set[int],
    output_path: Path,
) -> None:
    rows = ["image_id,file_name,split"]
    for split_name, ids in [("train", train_ids), ("test", test_ids), ("unused_holdout", unused_ids)]:
        for image_id in sorted(ids):
            rows.append(f"{image_id},{image_basename(images_by_id[image_id]['file_name'])},{split_name}")
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def image_basename(file_name: str) -> str:
    return PureWindowsPath(str(file_name)).name


class CocoPersonDataset(Dataset):
    def __init__(self, coco_json: str | Path, image_dir: str | Path, max_images: int = 0) -> None:
        self.coco = load_coco(coco_json)
        self.image_dir = Path(image_dir)
        self.images = sorted(self.coco["images"], key=lambda x: x["id"])
        if max_images and max_images > 0:
            self.images = self.images[:max_images]
        self.anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        allowed_ids = {img["id"] for img in self.images}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in allowed_ids:
                self.anns_by_image[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_meta = self.images[idx]
        path = self.image_dir / image_basename(img_meta["file_name"])
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        image_tensor = torch.as_tensor(image_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        anns = self.anns_by_image[img_meta["id"]]
        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for ann in anns:
            mask = annotation_to_mask(ann, int(img_meta["height"]), int(img_meta["width"]))
            if mask.sum() == 0:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            masks.append(mask)
            areas.append(float(ann.get("area", mask.sum())))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if not boxes:
            raise ValueError(f"No valid annotations for image_id={img_meta['id']} path={path}")

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
            "image_id": torch.tensor([int(img_meta["id"])]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target


def annotation_to_mask(ann: dict[str, Any], height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool = True):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    return maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=3 if pretrained else None,
        min_size=768,
        max_size=1024,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool, epoch: int) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for step, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        count += 1
        totals["loss"] += float(losses.detach().cpu())
        for key, value in loss_dict.items():
            totals[key] += float(value.detach().cpu())
        if step == 1 or step % 25 == 0 or step == len(loader):
            avg = totals["loss"] / count
            print(f"epoch={epoch} step={step}/{len(loader)} avg_loss={avg:.4f}")
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_model(model, loader, device, score_threshold: float, iou_threshold: float) -> dict[str, Any]:
    model.eval()
    total_gt = 0
    total_pred = 0
    total_matched = 0
    matched_ious: list[float] = []
    per_image: list[dict[str, Any]] = []
    for idx, (images, targets) in enumerate(loader, 1):
        image = images[0].to(device)
        target = targets[0]
        output = model([image])[0]
        pred_keep = (output["labels"].detach().cpu() == 1) & (output["scores"].detach().cpu() >= score_threshold)
        pred_masks = (output["masks"].detach().cpu()[pred_keep, 0] >= 0.5).numpy().astype(bool)
        gt_masks = target["masks"].detach().cpu().numpy().astype(bool)
        matched, ious = match_masks(gt_masks, pred_masks, iou_threshold)
        total_gt += len(gt_masks)
        total_pred += len(pred_masks)
        total_matched += matched
        matched_ious.extend(ious)
        per_image.append(
            {
                "image_id": int(target["image_id"].item()),
                "gt": int(len(gt_masks)),
                "pred": int(len(pred_masks)),
                "matched": int(matched),
                "mean_iou": float(np.mean(ious)) if ious else 0.0,
            }
        )
        if idx == 1 or idx % 10 == 0 or idx == len(loader):
            print(f"eval_image={idx}/{len(loader)} matched={total_matched} gt={total_gt} pred={total_pred}")

    precision = total_matched / total_pred if total_pred else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "gt_instances": total_gt,
        "pred_instances": total_pred,
        "matched_instances": total_matched,
        "precision_at_iou": precision,
        "recall_at_iou": recall,
        "f1_at_iou": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "per_image": per_image,
    }


def match_masks(gt_masks: np.ndarray, pred_masks: np.ndarray, threshold: float) -> tuple[int, list[float]]:
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return 0, []
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_masks):
        gt_sum = gt.sum()
        for pi, pred in enumerate(pred_masks):
            inter = np.logical_and(gt, pred).sum()
            if inter == 0:
                continue
            union = gt_sum + pred.sum() - inter
            iou = float(inter / union) if union else 0.0
            if iou >= threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    ious: list[float] = []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        ious.append(iou)
    return len(ious), ious


if __name__ == "__main__":
    main()
