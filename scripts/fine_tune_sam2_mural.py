from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam2_mural_common as common
import train_person_maskrcnn as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the SAM 2 prompt and mask decoder for mural person masks.")
    parser.add_argument("--train-json", default="runs/15temple_person_maskrcnn/coco_person_train_1200.json")
    parser.add_argument("--image-dir", default="data/15temple/images")
    parser.add_argument("--output-dir", default="runs/15temple_sam2_tuned")
    parser.add_argument("--sam2-checkpoint", default="models/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-jitter", type=float, default=0.05)
    parser.add_argument("--instances-per-step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def split_indices(count: int, val_count: int, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_count = min(max(1, val_count), max(1, count - 1))
    return sorted(indices[val_count:]), sorted(indices[:val_count])


def model_for_training(args: argparse.Namespace, device: torch.device):
    checkpoint = args.resume or args.sam2_checkpoint
    model = build_sam2(args.sam2_config, checkpoint, device=str(device))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.sam_prompt_encoder.parameters():
        parameter.requires_grad = True
    for parameter in model.sam_mask_decoder.parameters():
        parameter.requires_grad = True
    return model


def resume_epoch(path: str) -> int:
    if not path:
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("training_metadata", {}).get("epoch", 0))


def image_and_target(dataset: base.CocoPersonDataset, index: int) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    image_meta = dataset.images[index]
    image_path = dataset.image_dir / base.image_basename(image_meta["file_name"])
    image = np.asarray(Image.open(image_path).convert("RGB"))
    _, target = dataset[index]
    return image, target


@torch.inference_mode()
def validate(
    predictor: SAM2ImagePredictor,
    dataset: base.CocoPersonDataset,
    indices: list[int],
    max_images: int,
) -> dict[str, Any]:
    if max_images > 0:
        indices = indices[:max_images]
    modes = ["box", "box_point", "iterative_box_point_mask"]
    ious = {mode: [] for mode in modes}
    started = time.time()
    predictor.model.eval()
    for position, index in enumerate(indices, 1):
        image, target = image_and_target(dataset, index)
        boxes = target["boxes"].numpy().astype(np.float32)
        gt_masks = target["masks"].numpy().astype(bool)
        for mode in modes:
            predicted, _ = common.predict_prompt_masks(predictor, image, boxes, mode)
            ious[mode].extend(common.mask_iou(gt, pred) for gt, pred in zip(gt_masks, predicted))
        if position == 1 or position % 20 == 0 or position == len(indices):
            print(f"validation={position}/{len(indices)}")
    return {
        "images": len(indices),
        "seconds": round(time.time() - started, 2),
        "prompt_metrics": {mode: common.summarize_instance_ious(values) for mode, values in ious.items()},
    }


def train_epoch(
    predictor: SAM2ImagePredictor,
    dataset: base.CocoPersonDataset,
    indices: list[int],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    predictor.model.sam_prompt_encoder.train()
    predictor.model.sam_mask_decoder.train()
    rng = np.random.default_rng(args.seed + epoch)
    order = list(indices)
    random.Random(args.seed + epoch).shuffle(order)
    if args.max_train_images > 0:
        order = order[: args.max_train_images]
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "iou_head": 0.0}
    prompt_counts = {"box": 0, "box_point": 0, "box_point_mask": 0}
    steps = 0
    started = time.time()
    for image_number, index in enumerate(order, 1):
        image, target = image_and_target(dataset, index)
        predictor.set_image(image)
        height, width = image.shape[:2]
        boxes = target["boxes"].numpy().astype(np.float32)
        gt_masks = target["masks"]
        prompt_boxes = np.stack([common.jitter_box(box, width, height, args.box_jitter, rng) for box in boxes])
        points = np.stack([common.interior_point(mask.numpy(), box) for mask, box in zip(gt_masks, prompt_boxes)])[:, None, :]
        point_labels = np.ones((len(points), 1), dtype=np.int32)

        prompt_draw = float(rng.random())
        if prompt_draw < 0.25:
            prompt_mode = "box"
        elif prompt_draw < 0.75:
            prompt_mode = "box_point"
        else:
            prompt_mode = "box_point_mask"
        prompt_counts[prompt_mode] += 1

        for start in range(0, len(boxes), args.instances_per_step):
            end = min(len(boxes), start + args.instances_per_step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=predictor.device.type == "cuda"):
                chunk_points = None if prompt_mode == "box" else points[start:end]
                chunk_labels = None if prompt_mode == "box" else point_labels[start:end]
                mask_input = None
                if prompt_mode == "box_point_mask":
                    coarse = []
                    for mask in gt_masks[start:end].numpy().astype(np.uint8):
                        resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                        kernel_size = int(rng.choice([3, 5, 7, 9]))
                        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                        if rng.random() < 0.5:
                            resized = cv2.erode(resized, kernel, iterations=1)
                        else:
                            resized = cv2.dilate(resized, kernel, iterations=1)
                        coarse.append(np.where(resized > 0, 4.0, -4.0)[None, ...])
                    mask_input = np.stack(coarse).astype(np.float32)
                logits, iou_predictions = common.decode_with_grad(
                    predictor,
                    prompt_boxes[start:end],
                    chunk_points,
                    chunk_labels,
                    mask_input,
                )
                targets = common.resize_targets_to_logits(gt_masks[start:end].to(predictor.device), logits)
                bce = F.binary_cross_entropy_with_logits(logits, targets)
                dice = common.dice_loss(logits, targets)
                target_iou = common.soft_iou(logits.detach(), targets)
                quality = iou_predictions[:, 0]
                iou_head = F.mse_loss(quality.float(), target_iou.float())
                loss = bce + dice + 0.2 * iou_head
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(predictor.model.sam_prompt_encoder.parameters()) + list(predictor.model.sam_mask_decoder.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            totals["loss"] += float(loss.detach().cpu())
            totals["bce"] += float(bce.detach().cpu())
            totals["dice"] += float(dice.detach().cpu())
            totals["iou_head"] += float(iou_head.detach().cpu())
        if image_number == 1 or image_number % 25 == 0 or image_number == len(order):
            print(f"epoch={epoch} images={image_number}/{len(order)} avg_loss={totals['loss']/max(steps,1):.5f}")
    return {
        **{key: value / max(steps, 1) for key, value in totals.items()},
        "steps": steps,
        "images": len(order),
        "seconds": round(time.time() - started, 2),
        "prompt_image_counts": prompt_counts,
    }


def save_checkpoint(path: Path, predictor: SAM2ImagePredictor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": predictor.model.state_dict(), "training_metadata": metadata}, path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(Path("models/torch_cache").resolve()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This fine-tuning script requires CUDA for the SAM 2 Base+ checkpoint.")

    dataset = base.CocoPersonDataset(args.train_json, args.image_dir)
    train_indices, val_indices = split_indices(len(dataset), args.val_count, args.seed)
    split_payload = {
        "seed": args.seed,
        "train_images": len(train_indices),
        "validation_images": len(val_indices),
        "train_image_ids": [int(dataset.images[i]["id"]) for i in train_indices],
        "validation_image_ids": [int(dataset.images[i]["id"]) for i in val_indices],
    }
    common.save_json(output_dir / "sam2_train_validation_split.json", split_payload)

    start_epoch = resume_epoch(args.resume)
    model = model_for_training(args, device)
    predictor = SAM2ImagePredictor(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    print(f"device={device} trainable_parameters={sum(p.numel() for p in trainable)}")
    print(f"train_images={len(train_indices)} validation_images={len(val_indices)}")

    baseline_path = output_dir / "validation_baseline.json"
    if baseline_path.exists():
        baseline = common.load_json(baseline_path)
    else:
        baseline = validate(predictor, dataset, val_indices, args.max_val_images)
        common.save_json(baseline_path, baseline)
    history_path = output_dir / "training_history.json"
    history: list[dict[str, Any]] = common.load_json(history_path) if history_path.exists() else []
    best_score = max(
        [baseline["prompt_metrics"]["box_point"]["mean_iou"]]
        + [max(metric["mean_iou"] for metric in row["validation"]["prompt_metrics"].values()) for row in history]
    )
    best_path = output_dir / "sam2_mural_best.pt"

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_stats = train_epoch(predictor, dataset, train_indices, optimizer, scaler, args, epoch)
        validation = validate(predictor, dataset, val_indices, args.max_val_images)
        row = {"epoch": epoch, "train": train_stats, "validation": validation}
        history.append(row)
        common.save_json(history_path, history)
        epoch_path = output_dir / f"sam2_mural_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch_path, predictor, {"args": vars(args), "epoch": epoch, "train": train_stats, "validation": validation})
        score = max(metric["mean_iou"] for metric in validation["prompt_metrics"].values())
        if score > best_score or not best_path.exists():
            best_score = score
            save_checkpoint(best_path, predictor, {"args": vars(args), "epoch": epoch, "train": train_stats, "validation": validation})
        print(json.dumps(row, ensure_ascii=False))

    summary = {
        "base_checkpoint": args.sam2_checkpoint,
        "best_checkpoint": str(best_path),
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "frozen_candidate_model": "outputs/deliverables/15temple_person_segmentation_training_package/model/maskrcnn_person_latest.pth",
        "baseline": baseline,
        "history": history,
    }
    common.save_json(output_dir / "sam2_fine_tuning_summary.json", summary)
    print(f"best_checkpoint={best_path}")


if __name__ == "__main__":
    main()
