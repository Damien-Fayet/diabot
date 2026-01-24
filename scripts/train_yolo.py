#!/usr/bin/env python3
"""Minimal YOLO training script for Diablo vision.

Usage (examples):
  python scripts/train_yolo.py --data data/ml/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --batch 16

Requirements:
  pip install ultralytics
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for Diablo vision")
    parser.add_argument("--data", type=str, default="data/ml/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (e.g., yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory")
    parser.add_argument("--name", type=str, default="diablo-yolo", help="Run name inside project")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    model = YOLO(args.model)

    print("🚀 Starting training")
    print(f"  data   : {data_path}")
    print(f"  model  : {args.model}")
    print(f"  epochs : {args.epochs}")
    print(f"  imgsz  : {args.imgsz}")
    print(f"  batch  : {args.batch}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=args.workers,
    )

    print("✅ Training finished")
    print(f"Artifacts: {results.save_dir}")
    print("Tip: best weights are saved as best.pt inside the run directory")


if __name__ == "__main__":
    main()
