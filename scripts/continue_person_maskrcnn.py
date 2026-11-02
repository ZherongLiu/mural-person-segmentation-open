from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_person_maskrcnn as base


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue a Mask R-CNN person model from an existing checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(Path("models/torch_cache").resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    start_epoch = int(checkpoint.get("epoch", 0))

    model = base.build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    train_set = base.CocoPersonDataset(args.train_json, args.image_dir)
    test_set = base.CocoPersonDataset(args.test_json, args.image_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=base.collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=base.collate_fn)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history_path = out_dir / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []

    print(f"device={device} resume_epoch={start_epoch} train_images={len(train_set)} test_images={len(test_set)}")
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        started = time.time()
        stats = base.train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, epoch)
        scheduler.step()
        ckpt_path = out_dir / f"maskrcnn_person_epoch_{epoch:03d}.pth"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "train_stats": stats}, ckpt_path)
        row = {"epoch": epoch, "checkpoint": str(ckpt_path), "seconds": round(time.time() - started, 2), **stats}
        history.append(row)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False))

    metrics = base.evaluate_model(model, test_loader, device, args.score_threshold, args.iou_threshold)
    metrics["phase"] = f"continued_from_epoch_{start_epoch}"
    (out_dir / "test_metrics_epoch_020.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = out_dir / "maskrcnn_person_latest.pth"
    torch.save({"epoch": start_epoch + args.epochs, "model_state": model.state_dict(), "test_metrics": metrics}, latest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"latest_checkpoint={latest}")


if __name__ == "__main__":
    main()
