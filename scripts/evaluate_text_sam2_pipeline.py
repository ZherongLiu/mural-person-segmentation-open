from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam2_mural_common as common
import train_person_maskrcnn as base


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and test a text-guided candidate -> tuned SAM 2 mural workflow.")
    parser.add_argument("--train-json", default="runs/15temple_person_maskrcnn/coco_person_train_1200.json")
    parser.add_argument("--test-json", default="runs/15temple_person_maskrcnn/coco_person_test_138.json")
    parser.add_argument("--image-dir", default="data/15temple/images")
    parser.add_argument("--sam2-split-json", default="runs/15temple_sam2_tuned/sam2_train_validation_split.json")
    parser.add_argument("--output-dir", default="runs/15temple_sam2_tuned")
    parser.add_argument("--rcnn-checkpoint", default="outputs/deliverables/15temple_person_segmentation_training_package/model/maskrcnn_person_latest.pth")
    parser.add_argument("--base-sam2-checkpoint", default="models/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--tuned-sam2-checkpoint", default="runs/15temple_sam2_tuned/sam2_mural_best.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--grounding-cache", default="models/hf_cache")
    parser.add_argument("--text-prompt", default="person.")
    parser.add_argument("--candidate-min-dino-score", type=float, default=0.12)
    parser.add_argument("--candidate-min-rcnn-score", type=float, default=0.30)
    parser.add_argument("--rcnn-score", type=float, default=0.50)
    parser.add_argument("--dino-thresholds", default="0.25,0.30,0.35,0.40")
    parser.add_argument("--box-expansions", default="0,0.03,0.06")
    parser.add_argument("--mask-thresholds", default="-0.5,0,0.5")
    parser.add_argument("--prompt-modes", default="box,box_point,iterative_box_point_mask")
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-calibration-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument("--calibration-mask-size", type=int, default=512)
    parser.add_argument("--reuse-candidate-cache", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["all", "cache-calibration", "calibrate", "cache-test", "test"],
        default="all",
    )
    parser.add_argument("--test-variant", choices=["both", "base", "tuned"], default="both")
    return parser.parse_args()


def free_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_rcnn(checkpoint_path: str, device: torch.device):
    model = base.build_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_grounding(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(
        args.grounding_model,
        cache_dir=args.grounding_cache,
        local_files_only=True,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model,
        cache_dir=args.grounding_cache,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return processor, model


def dataset_indices_for_ids(dataset: base.CocoPersonDataset, image_ids: set[int]) -> list[int]:
    return [index for index, image in enumerate(dataset.images) if int(image["id"]) in image_ids]


def candidate_cache(
    args: argparse.Namespace,
    dataset: base.CocoPersonDataset,
    indices: list[int],
    cache_path: Path,
    device: torch.device,
    max_images: int,
) -> dict[str, Any]:
    if args.reuse_candidate_cache and cache_path.exists():
        print(f"reuse_candidate_cache={cache_path}")
        return common.load_json(cache_path)
    if max_images > 0:
        indices = indices[:max_images]
    rcnn = load_rcnn(args.rcnn_checkpoint, device)
    processor, grounding = load_grounding(args, device)
    records: list[dict[str, Any]] = []
    started = time.time()
    for position, index in enumerate(indices, 1):
        image_meta = dataset.images[index]
        image_path = dataset.image_dir / base.image_basename(image_meta["file_name"])
        pil_image = Image.open(image_path).convert("RGB")
        image = np.asarray(pil_image).copy()
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32, device=device) / 255.0

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            rcnn_output = rcnn([image_tensor])[0]
        rcnn_scores = rcnn_output["scores"].float().cpu().numpy()
        rcnn_labels = rcnn_output["labels"].cpu().numpy()
        rcnn_keep = (rcnn_labels == 1) & (rcnn_scores >= args.candidate_min_rcnn_score)
        rcnn_boxes = rcnn_output["boxes"].float().cpu().numpy()[rcnn_keep]
        rcnn_scores = rcnn_scores[rcnn_keep]
        rcnn_masks = rcnn_output["masks"].float().cpu().numpy()[rcnn_keep, 0] >= 0.5
        rcnn_points = [common.interior_point(mask, box).tolist() for mask, box in zip(rcnn_masks, rcnn_boxes)]

        inputs = processor(images=pil_image, text=[args.text_prompt], return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = grounding(**inputs)
        grounded = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.candidate_min_dino_score,
            text_threshold=0.15,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        dino_boxes = grounded["boxes"].float().cpu().numpy()
        dino_scores = grounded["scores"].float().cpu().numpy()
        dino_labels = [str(label) for label in grounded["text_labels"]]
        dino_points = [
            [float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)] for box in dino_boxes
        ]
        proposals = []
        for box, score, point in zip(rcnn_boxes, rcnn_scores, rcnn_points):
            proposals.append({"box": [float(v) for v in box], "score": float(score), "source": "frozen_rcnn", "label": "person", "point": point})
        for box, score, label, point in zip(dino_boxes, dino_scores, dino_labels, dino_points):
            proposals.append({"box": [float(v) for v in box], "score": float(score), "source": "text_grounding_dino", "label": label, "point": point})
        records.append(
            {
                "image_id": int(image_meta["id"]),
                "file_name": base.image_basename(image_meta["file_name"]),
                "width": int(image_meta["width"]),
                "height": int(image_meta["height"]),
                "proposals": proposals,
            }
        )
        if position == 1 or position % 10 == 0 or position == len(indices):
            print(f"candidate_images={position}/{len(indices)} rcnn={len(rcnn_boxes)} text={len(dino_boxes)}")
    payload = {
        "text_prompt": args.text_prompt,
        "rcnn_checkpoint": args.rcnn_checkpoint,
        "grounding_model": args.grounding_model,
        "candidate_min_dino_score": args.candidate_min_dino_score,
        "candidate_min_rcnn_score": args.candidate_min_rcnn_score,
        "images": len(records),
        "seconds": round(time.time() - started, 2),
        "records": records,
    }
    common.save_json(cache_path, payload)
    free_cuda(rcnn, grounding, processor)
    return payload


def select_candidates(record: dict[str, Any], dino_score: float, rcnn_score: float, nms_iou: float) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    proposals = [
        proposal
        for proposal in record["proposals"]
        if (proposal["source"] == "frozen_rcnn" and proposal["score"] >= rcnn_score)
        or (proposal["source"] == "text_grounding_dino" and proposal["score"] >= dino_score)
    ]
    if not proposals:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
            np.zeros((0, 2), dtype=np.float32),
        )
    boxes = torch.tensor([proposal["box"] for proposal in proposals], dtype=torch.float32)
    scores = torch.tensor([proposal["score"] for proposal in proposals], dtype=torch.float32)
    keep = nms(boxes, scores, nms_iou).cpu().numpy().tolist()
    selected = [proposals[index] for index in keep]
    return (
        np.asarray([proposal["box"] for proposal in selected], dtype=np.float32),
        np.asarray([proposal["score"] for proposal in selected], dtype=np.float32),
        [proposal["source"] for proposal in selected],
        np.asarray([proposal["point"] for proposal in selected], dtype=np.float32),
    )


def load_sam_predictor(args: argparse.Namespace, checkpoint: str, device: torch.device) -> SAM2ImagePredictor:
    model = build_sam2(args.sam2_config, checkpoint, device=str(device))
    model.eval()
    return SAM2ImagePredictor(model)


def dataset_index_by_id(dataset: base.CocoPersonDataset) -> dict[int, int]:
    return {int(image_meta["id"]): index for index, image_meta in enumerate(dataset.images)}


def empty_accumulator(params: dict[str, Any]) -> dict[str, Any]:
    return {"params": params, "gt": 0, "pred": 0, "matched": 0, "ious": [], "images": 0, "any_success": 0, "all_gt": 0, "exact_count": 0}


def finalize_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    precision = acc["matched"] / acc["pred"] if acc["pred"] else 0.0
    recall = acc["matched"] / acc["gt"] if acc["gt"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "params": acc["params"],
        "images": acc["images"],
        "gt_instances": acc["gt"],
        "pred_instances": acc["pred"],
        "matched_instances": acc["matched"],
        "precision_at_iou_0_5": precision,
        "recall_at_iou_0_5": recall,
        "f1_at_iou_0_5": f1,
        "mean_matched_iou": float(np.mean(acc["ious"])) if acc["ious"] else 0.0,
        "median_matched_iou": float(np.median(acc["ious"])) if acc["ious"] else 0.0,
        "image_any_success": acc["any_success"] / acc["images"] if acc["images"] else 0.0,
        "image_all_gt_matched": acc["all_gt"] / acc["images"] if acc["images"] else 0.0,
        "image_exact_count": acc["exact_count"] / acc["images"] if acc["images"] else 0.0,
    }


def calibrate(
    args: argparse.Namespace,
    cache: dict[str, Any],
    dataset: base.CocoPersonDataset,
    device: torch.device,
) -> dict[str, Any]:
    predictor = load_sam_predictor(args, args.tuned_sam2_checkpoint, device)
    index_by_id = dataset_index_by_id(dataset)
    dino_thresholds = parse_csv_floats(args.dino_thresholds)
    box_expansions = parse_csv_floats(args.box_expansions)
    mask_thresholds = parse_csv_floats(args.mask_thresholds)
    modes = [item.strip() for item in args.prompt_modes.split(",") if item.strip()]
    configurations = [
        {"dino_score": dino, "rcnn_score": args.rcnn_score, "nms_iou": args.nms_iou, "prompt_mode": mode, "box_expansion": expansion, "mask_threshold": mask_threshold}
        for dino, mode, expansion, mask_threshold in product(dino_thresholds, modes, box_expansions, mask_thresholds)
    ]
    accumulators = [empty_accumulator(config) for config in configurations]
    accumulator_by_key = {
        (config["dino_score"], config["prompt_mode"], config["box_expansion"], config["mask_threshold"]): acc
        for config, acc in zip(configurations, accumulators)
    }
    min_dino_threshold = min(dino_thresholds)
    started = time.time()
    records = cache["records"][: args.max_calibration_images] if args.max_calibration_images > 0 else cache["records"]
    for position, record in enumerate(records, 1):
        image = np.asarray(Image.open(dataset.image_dir / record["file_name"]).convert("RGB")).copy()
        _, target = dataset[index_by_id[int(record["image_id"])]]
        gt_masks = target["masks"].numpy().astype(bool)
        calibration_size = args.calibration_mask_size
        gt_masks_small = np.stack(
            [cv2.resize(mask.astype(np.uint8), (calibration_size, calibration_size), interpolation=cv2.INTER_NEAREST).astype(bool) for mask in gt_masks]
        )
        predictor.set_image(image)
        boxes, scores, sources, points = select_candidates(record, min_dino_threshold, args.rcnn_score, args.nms_iou)
        for mode, expansion in product(modes, box_expansions):
            logits, _ = common.predict_prompt_masks(
                predictor,
                image,
                boxes,
                mode,
                box_expansion=expansion,
                mask_threshold=0.0,
                seed_points=points,
                image_is_set=True,
                return_logits=True,
            )
            for mask_threshold, dino_threshold in product(mask_thresholds, dino_thresholds):
                keep = np.asarray(
                    [source == "frozen_rcnn" or score >= dino_threshold for source, score in zip(sources, scores)],
                    dtype=bool,
                )
                masks = logits[keep] > mask_threshold
                masks_small = (
                    np.stack(
                        [cv2.resize(mask.astype(np.uint8), (calibration_size, calibration_size), interpolation=cv2.INTER_NEAREST).astype(bool) for mask in masks]
                    )
                    if len(masks)
                    else np.zeros((0, calibration_size, calibration_size), dtype=bool)
                )
                acc = accumulator_by_key[(dino_threshold, mode, expansion, mask_threshold)]
                matched, ious = base.match_masks(gt_masks_small, masks_small, args.match_iou)
                acc["images"] += 1
                acc["gt"] += len(gt_masks)
                acc["pred"] += len(masks)
                acc["matched"] += matched
                acc["ious"].extend(ious)
                acc["any_success"] += int(matched > 0)
                acc["all_gt"] += int(matched >= len(gt_masks))
                acc["exact_count"] += int(matched == len(gt_masks) and len(masks) == len(gt_masks))
        if position == 1 or position % 10 == 0 or position == len(records):
            print(f"calibration_images={position}/{len(records)}")
    rows = [finalize_accumulator(acc) for acc in accumulators]
    rows.sort(key=lambda row: (row["f1_at_iou_0_5"], row["mean_matched_iou"], row["recall_at_iou_0_5"]), reverse=True)
    payload = {"checkpoint": args.tuned_sam2_checkpoint, "text_prompt": args.text_prompt, "seconds": round(time.time() - started, 2), "best": rows[0], "all_results": rows}
    common.save_json(Path(args.output_dir) / "sam2_prompt_calibration.json", payload)
    free_cuda(predictor)
    return payload


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [(0, 186, 136), (255, 193, 7), (3, 169, 244), (233, 30, 99), (156, 39, 176), (255, 87, 34)]
    return palette[index % len(palette)]


def write_visualization(image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray, sources: list[str], path: Path) -> None:
    canvas = image.astype(np.float32).copy()
    for index, mask in enumerate(masks):
        color = np.asarray(color_for_index(index), dtype=np.float32)
        canvas[mask] = canvas[mask] * 0.55 + color * 0.45
    output = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    for index, (box, score, source) in enumerate(zip(boxes, scores, sources)):
        color = color_for_index(index)
        draw.rectangle(tuple(float(v) for v in box), outline=color, width=4)
        label = f"person {score:.2f} {'text' if source == 'text_grounding_dino' else 'domain'}"
        draw.text((float(box[0]) + 3, max(0.0, float(box[1]) - 16)), label, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def evaluate_one_checkpoint(
    args: argparse.Namespace,
    cache: dict[str, Any],
    dataset: base.CocoPersonDataset,
    params: dict[str, Any],
    checkpoint: str,
    label: str,
    device: torch.device,
    write_outputs: bool,
) -> dict[str, Any]:
    predictor = load_sam_predictor(args, checkpoint, device)
    index_by_id = dataset_index_by_id(dataset)
    annotated_ids = {int(annotation["image_id"]) for annotation in dataset.coco["annotations"]}
    acc = empty_accumulator(params)
    per_image = []
    output_root = Path(args.output_dir) / "final_test_outputs"
    started = time.time()
    for position, record in enumerate(cache["records"], 1):
        image_path = dataset.image_dir / record["file_name"]
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        boxes, scores, sources, points = select_candidates(record, params["dino_score"], params["rcnn_score"], params["nms_iou"])
        masks, sam_scores = common.predict_prompt_masks(
            predictor,
            image,
            boxes,
            params["prompt_mode"],
            box_expansion=params["box_expansion"],
            mask_threshold=params["mask_threshold"],
            seed_points=points,
        )
        image_id = int(record["image_id"])
        if image_id in annotated_ids:
            _, target = dataset[index_by_id[image_id]]
            gt_masks = target["masks"].numpy().astype(bool)
            matched, ious = base.match_masks(gt_masks, masks, args.match_iou)
            acc["images"] += 1
            acc["gt"] += len(gt_masks)
            acc["pred"] += len(masks)
            acc["matched"] += matched
            acc["ious"].extend(ious)
            acc["any_success"] += int(matched > 0)
            acc["all_gt"] += int(matched >= len(gt_masks))
            acc["exact_count"] += int(matched == len(gt_masks) and len(masks) == len(gt_masks))
            gt_count: int | None = len(gt_masks)
        else:
            matched, ious, gt_count = 0, [], None
        per_image.append({"image_id": image_id, "file_name": record["file_name"], "gt": gt_count, "pred": len(masks), "matched": matched, "matched_ious": ious, "candidate_sources": sources, "candidate_scores": scores.tolist(), "sam_quality_scores": sam_scores.tolist()})
        if write_outputs:
            stem = Path(record["file_name"]).stem
            write_visualization(image, masks, boxes, scores, sources, output_root / "visualizations" / f"{stem}_tuned_sam2_overlay.png")
            mask_dir = output_root / "masks" / stem
            mask_dir.mkdir(parents=True, exist_ok=True)
            for mask_index, mask in enumerate(masks, 1):
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"person_{mask_index:03d}.png")
        if position == 1 or position % 10 == 0 or position == len(cache["records"]):
            print(f"{label}_test_images={position}/{len(cache['records'])} matched={acc['matched']}/{acc['gt']}")
    metrics = finalize_accumulator(acc)
    payload = {"label": label, "checkpoint": checkpoint, "text_prompt": args.text_prompt, "inference_images": len(cache["records"]), "quantitative_images": metrics["images"], "seconds": round(time.time() - started, 2), "metrics": metrics, "per_image": per_image}
    common.save_json(Path(args.output_dir) / f"test_metrics_{label}.json", payload)
    free_cuda(predictor)
    return payload


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(Path("models/torch_cache").resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this complete Grounding DINO + SAM 2 evaluation.")

    train_dataset = base.CocoPersonDataset(args.train_json, args.image_dir)
    split = common.load_json(args.sam2_split_json)
    validation_ids = {int(value) for value in split["validation_image_ids"]}
    calibration_indices = dataset_indices_for_ids(train_dataset, validation_ids)
    calibration_cache_path = output_dir / "candidate_cache_calibration.json"
    calibration_path = output_dir / "sam2_prompt_calibration.json"
    if args.stage in {"all", "cache-calibration"}:
        candidate_cache(args, train_dataset, calibration_indices, calibration_cache_path, device, args.max_calibration_images)
        if args.stage == "cache-calibration":
            return
    if args.stage in {"all", "calibrate"}:
        if not calibration_cache_path.exists():
            raise FileNotFoundError(f"Create the calibration candidate cache first: {calibration_cache_path}")
        calibration_cache = common.load_json(calibration_cache_path)
        calibrate(args, calibration_cache, train_dataset, device)
        if args.stage == "calibrate":
            return
    if not calibration_path.exists():
        raise FileNotFoundError(f"Run calibration first: {calibration_path}")
    calibration = common.load_json(calibration_path)
    params = calibration["best"]["params"]

    test_dataset = base.CocoPersonDataset(args.test_json, args.image_dir, max_images=args.max_test_images)
    test_indices = list(range(len(test_dataset)))
    test_cache_path = output_dir / "candidate_cache_test.json"
    if args.stage in {"all", "cache-test"}:
        candidate_cache(args, test_dataset, test_indices, test_cache_path, device, args.max_test_images)
        if args.stage == "cache-test":
            return
    if not test_cache_path.exists():
        raise FileNotFoundError(f"Create the test candidate cache first: {test_cache_path}")
    test_cache = common.load_json(test_cache_path)
    baseline_path = output_dir / "test_metrics_base_sam2.json"
    tuned_path = output_dir / "test_metrics_tuned_sam2.json"
    if args.test_variant in {"both", "base"}:
        baseline = evaluate_one_checkpoint(args, test_cache, test_dataset, params, args.base_sam2_checkpoint, "base_sam2", device, False)
    elif baseline_path.exists():
        baseline = common.load_json(baseline_path)
    else:
        raise FileNotFoundError(f"Run the base test first: {baseline_path}")
    if args.test_variant in {"both", "tuned"}:
        tuned = evaluate_one_checkpoint(args, test_cache, test_dataset, params, args.tuned_sam2_checkpoint, "tuned_sam2", device, True)
    elif tuned_path.exists():
        tuned = common.load_json(tuned_path)
    else:
        tuned = {"metrics": {}}
    summary = {
        "workflow": "text prompt Grounding DINO + frozen deliverable RCNN candidate boxes -> tuned SAM 2 masks",
        "text_prompt": args.text_prompt,
        "rcnn_checkpoint": args.rcnn_checkpoint,
        "rcnn_weights_modified": False,
        "selected_parameters": params,
        "calibration_best": calibration["best"],
        "base_sam2_test": baseline["metrics"],
        "tuned_sam2_test": tuned["metrics"],
    }
    common.save_json(output_dir / "complete_workflow_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
