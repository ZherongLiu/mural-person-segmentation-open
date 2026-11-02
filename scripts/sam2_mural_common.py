from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def expand_box(box: np.ndarray, ratio: float, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    dx = (x2 - x1) * ratio
    dy = (y2 - y1) * ratio
    return np.asarray(
        [max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(width - 1), x2 + dx), min(float(height - 1), y2 + dy)],
        dtype=np.float32,
    )


def jitter_box(
    box: np.ndarray,
    width: int,
    height: int,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    offsets = rng.uniform(-ratio, ratio, size=4) * np.asarray([bw, bh, bw, bh])
    result = np.asarray([x1, y1, x2, y2], dtype=np.float32) + offsets.astype(np.float32)
    result[0] = np.clip(result[0], 0, width - 2)
    result[1] = np.clip(result[1], 0, height - 2)
    result[2] = np.clip(result[2], result[0] + 1, width - 1)
    result[3] = np.clip(result[3], result[1] + 1, height - 1)
    return result


def interior_point(mask: np.ndarray, fallback_box: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    if binary.any():
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        return np.asarray([float(x), float(y)], dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in fallback_box]
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = logits.sigmoid().flatten(1)
    flat_targets = targets.flatten(1)
    numerator = 2.0 * (probs * flat_targets).sum(dim=1) + eps
    denominator = probs.sum(dim=1) + flat_targets.sum(dim=1) + eps
    return (1.0 - numerator / denominator).mean()


def soft_iou(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = logits.sigmoid().flatten(1)
    flat_targets = targets.flatten(1)
    intersection = (probs * flat_targets).sum(dim=1)
    union = probs.sum(dim=1) + flat_targets.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def decode_with_grad(
    predictor: SAM2ImagePredictor,
    boxes: np.ndarray,
    point_coords: np.ndarray | None,
    point_labels: np.ndarray | None,
    mask_input: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_input_t, coords_t, labels_t, boxes_t = predictor._prep_prompts(
        point_coords,
        point_labels,
        boxes,
        mask_input,
        True,
    )
    concat_points = (coords_t, labels_t) if coords_t is not None else None
    if boxes_t is not None:
        box_coords = boxes_t.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes_t.device).repeat(boxes_t.size(0), 1)
        if concat_points is None:
            concat_points = (box_coords, box_labels)
        else:
            concat_points = (
                torch.cat([box_coords, concat_points[0]], dim=1),
                torch.cat([box_labels, concat_points[1]], dim=1),
            )

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=mask_input_t,
    )
    repeat_image = concat_points is not None and concat_points[0].shape[0] > 1
    high_res_features = [level[-1].unsqueeze(0) for level in predictor._features["high_res_feats"]]
    low_res_masks, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=repeat_image,
        high_res_features=high_res_features,
    )
    return low_res_masks, iou_predictions


@torch.inference_mode()
def predict_prompt_masks(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    boxes: np.ndarray,
    mode: str,
    box_expansion: float = 0.0,
    mask_threshold: float = 0.0,
    seed_masks: np.ndarray | None = None,
    seed_points: np.ndarray | None = None,
    image_is_set: bool = False,
    return_logits: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, height, width), dtype=bool), np.zeros((0,), dtype=np.float32)
    predictor.mask_threshold = float(mask_threshold)
    if not image_is_set:
        predictor.set_image(image)
    outputs: list[np.ndarray] = []
    qualities: list[float] = []
    for index, raw_box in enumerate(boxes):
        box = expand_box(raw_box, box_expansion, width, height)
        point = seed_points[index] if seed_points is not None and index < len(seed_points) else None
        if seed_masks is not None and index < len(seed_masks):
            point = interior_point(seed_masks[index], box)
        if point is None:
            x1, y1, x2, y2 = box.tolist()
            point = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

        if mode == "box":
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False, return_logits=True)
            logits = masks[0]
        elif mode == "box_point":
            masks, scores, _ = predictor.predict(
                box=box[None, :],
                point_coords=point[None, :],
                point_labels=np.asarray([1], dtype=np.int32),
                multimask_output=False,
                return_logits=True,
            )
            logits = masks[0]
        elif mode == "iterative_box_point_mask":
            first_masks, _, first_low = predictor.predict(
                box=box[None, :],
                multimask_output=False,
                return_logits=True,
            )
            first_binary = first_masks[0] > mask_threshold
            refined_point = interior_point(first_binary, box)
            masks, scores, _ = predictor.predict(
                box=box[None, :],
                point_coords=refined_point[None, :],
                point_labels=np.asarray([1], dtype=np.int32),
                mask_input=first_low,
                multimask_output=False,
                return_logits=True,
            )
            logits = masks[0]
        else:
            raise ValueError(f"Unsupported prompt mode: {mode}")
        outputs.append(logits.astype(np.float32) if return_logits else logits > mask_threshold)
        qualities.append(float(scores[0]))
    return np.stack(outputs), np.asarray(qualities, dtype=np.float32)


def resize_targets_to_logits(target_masks: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    targets = target_masks[:, None].float()
    return F.interpolate(targets, size=logits.shape[-2:], mode="nearest")


def summarize_instance_ious(ious: list[float]) -> dict[str, float | int]:
    if not ious:
        return {"instances": 0, "mean_iou": 0.0, "median_iou": 0.0, "iou_at_0_5": 0.0, "iou_at_0_75": 0.0}
    values = np.asarray(ious, dtype=np.float64)
    return {
        "instances": int(len(values)),
        "mean_iou": float(values.mean()),
        "median_iou": float(np.median(values)),
        "iou_at_0_5": float((values >= 0.5).mean()),
        "iou_at_0_75": float((values >= 0.75).mean()),
    }
