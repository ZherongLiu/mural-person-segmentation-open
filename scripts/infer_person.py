from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam2_mural_common as common
import train_person_maskrcnn as base


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict person masks with Mask R-CNN boxes refined by SAM 2.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--checkpoint", default="weights/maskrcnn_person_latest.pth")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--sam2-checkpoint", default="weights/sam2_mural_best.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--prompt-mode", choices=["box", "box_point"], default="box_point")
    parser.add_argument("--box-expansion", type=float, default=0.02)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("TORCH_HOME", str(Path("models/torch_cache").resolve()))
    out_dir = Path(args.output_dir)
    overlay_dir = out_dir / "visualizations"
    mask_dir = out_dir / "masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(Path(args.input))
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No images found under {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maskrcnn = base.build_model(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    maskrcnn.load_state_dict(checkpoint["model_state"])
    maskrcnn.to(device)
    maskrcnn.eval()

    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)

    records = []
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths, 1):
            record = predict_one(
                image_path=image_path,
                maskrcnn=maskrcnn,
                predictor=predictor,
                device=device,
                score_threshold=args.score_threshold,
                prompt_mode=args.prompt_mode,
                box_expansion=args.box_expansion,
                mask_threshold=args.mask_threshold,
                overlay_dir=overlay_dir,
                mask_dir=mask_dir,
            )
            records.append(record)
            print(f"processed={idx}/{len(image_paths)} file={image_path.name} persons={len(record['instances'])}")

    payload = {
        "workflow": "Mask R-CNN person detection + SAM 2 boundary refinement",
        "checkpoint": args.checkpoint,
        "sam2_checkpoint": args.sam2_checkpoint,
        "score_threshold": args.score_threshold,
        "prompt_mode": args.prompt_mode,
        "box_expansion": args.box_expansion,
        "mask_threshold": args.mask_threshold,
        "image_count": len(records),
        "total_instances": sum(len(r["instances"]) for r in records),
        "images": records,
    }
    output_json = out_dir / "person_workflow_predictions.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_json={output_json}")
    print(f"visualizations={overlay_dir}")
    print(f"masks={mask_dir}")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def predict_one(
    image_path: Path,
    maskrcnn,
    predictor: SAM2ImagePredictor,
    device: torch.device,
    score_threshold: float,
    prompt_mode: str,
    box_expansion: float,
    mask_threshold: float,
    overlay_dir: Path,
    mask_dir: Path,
) -> dict[str, Any]:
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)
    image_tensor = torch.as_tensor(img_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    image_tensor = image_tensor.to(device)

    output = maskrcnn([image_tensor])[0]
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()
    boxes = output["boxes"].detach().cpu()
    keep = (labels == 1) & (scores >= score_threshold)
    kept_boxes = boxes[keep].numpy().astype(np.float32)
    kept_scores = scores[keep].numpy().astype(float)
    kept_mask_logits = output["masks"].detach().cpu()[keep, 0].numpy()
    seed_masks = kept_mask_logits >= 0.5
    refined_masks, sam_qualities = common.predict_prompt_masks(
        predictor,
        img_np,
        kept_boxes,
        prompt_mode,
        box_expansion=box_expansion,
        mask_threshold=mask_threshold,
        seed_masks=seed_masks if prompt_mode == "box_point" else None,
    )

    image_stem = image_path.stem
    instances = []
    overlay = img_np.copy()
    colors = make_colors(len(refined_masks))
    for i, (box, score, mask, quality) in enumerate(zip(kept_boxes, kept_scores, refined_masks, sam_qualities), 1):
        color = colors[i - 1]
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
        mask_png = mask_dir / f"{image_stem}_person_{i:03d}.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_png)
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        instances.append(
            {
                "instance_id": f"{image_stem}_person_{i:03d}",
                "label": "person",
                "score": float(score),
                "sam2_quality": float(quality),
                "bbox_xyxy": [round(float(v), 3) for v in box.tolist()],
                "mask_area": int(mask.sum()),
                "mask_png": str(mask_png),
                "segmentation_rle": {
                    "size": [int(v) for v in rle["size"]],
                    "counts": rle["counts"],
                },
                "prompt_mode": prompt_mode,
            }
        )

    out_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(out_img)
    for i, (box, score) in enumerate(zip(kept_boxes, kept_scores), 1):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        draw.text((x1, max(0, y1 - 16)), f"person {i} {score:.2f}", fill=(0, 255, 0))
    overlay_path = overlay_dir / f"{image_stem}_workflow_overlay.png"
    out_img.save(overlay_path)

    return {
        "image_path": str(image_path),
        "file_name": image_path.name,
        "width": pil_img.width,
        "height": pil_img.height,
        "overlay_png": str(overlay_path),
        "instances": instances,
    }


def make_colors(n: int) -> list[np.ndarray]:
    palette = [
        (64, 220, 96),
        (255, 170, 48),
        (64, 160, 255),
        (220, 80, 200),
        (255, 80, 80),
        (80, 220, 220),
    ]
    return [np.array(palette[i % len(palette)], dtype=np.uint8) for i in range(n)]


if __name__ == "__main__":
    main()
