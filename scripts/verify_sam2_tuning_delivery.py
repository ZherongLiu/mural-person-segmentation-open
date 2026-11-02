from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "15temple_sam2_tuned"
PACKAGE = ROOT / "outputs" / "deliverables" / "15temple_text_prompt_sam2_tuning_package"
EXPECTED_RCNN_SHA256 = "813131D3EA54835E3A13FBE5A7DD9EB457EE54587823057178CF1711D428BEF2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    summary = load(RUN / "complete_workflow_summary.json")
    base = summary["base_sam2_test"]
    tuned = summary["tuned_sam2_test"]
    predictions = load(RUN / "final_test_outputs" / "person_predictions.json")
    manifest = load(PACKAGE / "manifest.json")

    visualizations = list((RUN / "final_test_outputs" / "visualizations").glob("*.png"))
    masks = list((RUN / "final_test_outputs" / "masks").rglob("*.png"))
    structured_instances = sum(len(record["instances"]) for record in predictions["records"])
    report = PACKAGE / "report" / "SAM2领域微调与文本提示工作流报告.md"
    report_text = report.read_text(encoding="utf-8")
    asset_paths = [PACKAGE / "report" / value.split("(", 1)[1].split(")", 1)[0] for value in report_text.splitlines() if value.startswith("![")]

    assert summary["rcnn_weights_modified"] is False
    assert base["gt_instances"] == tuned["gt_instances"] == 422
    assert base["pred_instances"] == tuned["pred_instances"] == 495
    assert tuned["matched_instances"] > base["matched_instances"]
    assert tuned["f1_at_iou_0_5"] > base["f1_at_iou_0_5"]
    assert tuned["mean_matched_iou"] > base["mean_matched_iou"]
    assert len(visualizations) == predictions["images"] == 138
    assert len(masks) == predictions["predicted_instances"] == structured_instances == 496
    assert manifest["tuned_f1"] == tuned["f1_at_iou_0_5"]
    assert all(path.exists() and path.stat().st_size > 0 for path in asset_paths)
    assert sha256(RUN / "sam2_mural_best.pt") == sha256(PACKAGE / "model" / "sam2_mural_best.pt")
    rcnn = ROOT / "outputs" / "deliverables" / "15temple_person_segmentation_training_package" / "model" / "maskrcnn_person_latest.pth"
    assert sha256(rcnn) == EXPECTED_RCNN_SHA256

    print("verification=passed")
    print(f"quantitative_images={tuned['images']}")
    print(f"gt_instances={tuned['gt_instances']}")
    print(f"pred_instances_quantitative={tuned['pred_instances']}")
    print(f"matched_instances={tuned['matched_instances']}")
    print(f"tuned_f1={tuned['f1_at_iou_0_5']:.6f}")
    print(f"tuned_mean_iou={tuned['mean_matched_iou']:.6f}")
    print(f"visualizations={len(visualizations)} masks={len(masks)} structured_instances={structured_instances}")
    print(f"report_assets={len(asset_paths)}")
    print(f"rcnn_sha256={EXPECTED_RCNN_SHA256}")


if __name__ == "__main__":
    main()
