from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_person_maskrcnn as base


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN detection boxes refined by SAM 2 masks.")
    parser.add_argument("--checkpoint", default="runs/person_maskrcnn_v1/maskrcnn_person_latest.pth")
    parser.add_argument("--test-json", default="runs/person_maskrcnn_v1/coco_person_test_60.json")
    parser.add_argument("--image-dir", default="coco/images")
    parser.add_argument("--output-dir", default="runs/person_maskrcnn_sam2_v1")
    parser.add_argument("--sam2-checkpoint", default="models/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--prompt-mode", choices=["box", "box_point"], default="box_point")
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-test-images", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(Path("models/torch_cache").resolve()))

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"prompt_mode={args.prompt_mode}")
    print(f"thresholds={thresholds}")

    dataset = base.CocoPersonDataset(args.test_json, args.image_dir, max_images=args.max_test_images)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=base.collate_fn)

    maskrcnn = base.build_model(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    maskrcnn.load_state_dict(checkpoint["model_state"])
    maskrcnn.to(device)
    maskrcnn.eval()

    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)

    per_image: list[dict[str, Any]] = []
    start = time.time()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader, 1):
            image_tensor = images[0].to(device)
            target = targets[0]
            image_id = int(target["image_id"].item())
            image_meta = next(img for img in dataset.images if int(img["id"]) == image_id)
            image_path = Path(args.image_dir) / base.image_basename(image_meta["file_name"])
            pil_img = Image.open(image_path).convert("RGB")
            img_np = np.array(pil_img)

            output = maskrcnn([image_tensor])[0]
            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()
            boxes = output["boxes"].detach().cpu()
            keep = (labels == 1) & (scores >= args.min_score)
            kept_boxes = boxes[keep].numpy().astype(np.float32)
            kept_scores = scores[keep].numpy().astype(float)
            kept_mask_logits = output["masks"].detach().cpu()[keep, 0].numpy()
            point_coords = mask_centroid_points(kept_mask_logits, kept_boxes) if args.prompt_mode == "box_point" else None
            sam_masks = run_sam2_for_boxes(predictor, img_np, kept_boxes, device, point_coords=point_coords)
            gt_masks = target["masks"].detach().cpu().numpy().astype(bool)

            threshold_rows = {}
            for threshold in thresholds:
                th_keep = kept_scores >= threshold
                pred_masks = sam_masks[th_keep]
                matched, ious = base.match_masks(gt_masks, pred_masks, args.iou_threshold)
                threshold_rows[str(threshold)] = {
                    "gt": int(len(gt_masks)),
                    "pred": int(len(pred_masks)),
                    "matched": int(matched),
                    "mean_iou": float(np.mean(ious)) if ious else 0.0,
                }

            per_image.append(
                {
                    "image_id": image_id,
                    "file_name": base.image_basename(image_meta["file_name"]),
                    "candidate_boxes_min_score": int(len(kept_boxes)),
                    "thresholds": threshold_rows,
                }
            )
            if idx == 1 or idx % 10 == 0 or idx == len(loader):
                print(f"processed={idx}/{len(loader)} image_id={image_id} candidates={len(kept_boxes)}")

    summary = {
        "checkpoint": args.checkpoint,
        "test_json": args.test_json,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_config": args.sam2_config,
        "min_score": args.min_score,
        "prompt_mode": args.prompt_mode,
        "iou_threshold": args.iou_threshold,
        "test_images": len(dataset),
        "seconds": round(time.time() - start, 2),
        "threshold_metrics": summarize_thresholds(per_image, thresholds),
        "per_image": per_image,
    }
    output_path = out_dir / "maskrcnn_sam2_metrics.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["threshold_metrics"], ensure_ascii=False, indent=2))
    print(f"output={output_path}")


def run_sam2_for_boxes(
    predictor: SAM2ImagePredictor,
    img_np: np.ndarray,
    boxes: np.ndarray,
    device: torch.device,
    point_coords: np.ndarray | None = None,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=bool)
    masks: list[np.ndarray] = []
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.set_image(img_np)
                for idx, box in enumerate(boxes):
                    kwargs = {"box": box[None, :], "multimask_output": False}
                    if point_coords is not None:
                        kwargs["point_coords"] = point_coords[idx][None, :]
                        kwargs["point_labels"] = np.array([1], dtype=np.int32)
                    mask, _, _ = predictor.predict(**kwargs)
                    masks.append(mask[0].astype(bool))
        else:
            predictor.set_image(img_np)
            for idx, box in enumerate(boxes):
                kwargs = {"box": box[None, :], "multimask_output": False}
                if point_coords is not None:
                    kwargs["point_coords"] = point_coords[idx][None, :]
                    kwargs["point_labels"] = np.array([1], dtype=np.int32)
                mask, _, _ = predictor.predict(**kwargs)
                masks.append(mask[0].astype(bool))
    return np.stack(masks) if masks else np.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=bool)


def mask_centroid_points(mask_logits: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    points: list[list[float]] = []
    for mask, box in zip(mask_logits, boxes):
        binary = mask >= 0.5
        ys, xs = np.nonzero(binary)
        if len(xs) == 0:
            x1, y1, x2, y2 = box.tolist()
            points.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        else:
            points.append([float(np.mean(xs)), float(np.mean(ys))])
    return np.asarray(points, dtype=np.float32)


def summarize_thresholds(per_image: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        key = str(threshold)
        gt = sum(row["thresholds"][key]["gt"] for row in per_image)
        pred = sum(row["thresholds"][key]["pred"] for row in per_image)
        matched = sum(row["thresholds"][key]["matched"] for row in per_image)
        ious = [
            row["thresholds"][key]["mean_iou"]
            for row in per_image
            if row["thresholds"][key]["matched"] > 0
        ]
        precision = matched / pred if pred else 0.0
        recall = matched / gt if gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "score_threshold": threshold,
                "gt_instances": int(gt),
                "pred_instances": int(pred),
                "matched_instances": int(matched),
                "precision_at_iou": precision,
                "recall_at_iou": recall,
                "f1_at_iou": f1,
                "mean_image_iou_when_matched": float(np.mean(ious)) if ious else 0.0,
                "image_any_success": sum(1 for row in per_image if row["thresholds"][key]["matched"] > 0) / len(per_image),
                "image_all_gt_matched": sum(
                    1
                    for row in per_image
                    if row["thresholds"][key]["matched"] >= row["thresholds"][key]["gt"]
                )
                / len(per_image),
                "image_exact_count": sum(
                    1
                    for row in per_image
                    if row["thresholds"][key]["matched"] == row["thresholds"][key]["gt"]
                    and row["thresholds"][key]["pred"] == row["thresholds"][key]["gt"]
                )
                / len(per_image),
            }
        )
    return rows


if __name__ == "__main__":
    main()
