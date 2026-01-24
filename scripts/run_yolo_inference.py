#!/usr/bin/env python3
"""Run YOLO inference on a screenshot (optionally cropped to playfield).

Usage:
  python scripts/run_yolo_inference.py --model runs/train/diablo-yolo/weights/best.pt \
      --image data/screenshots/inputs/game.jpg \
      --output data/screenshots/outputs/yolo_inference.jpg \
      --conf 0.35

Requirements:
  pip install ultralytics
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from src.diabot.vision.screen_regions import ENVIRONMENT_REGIONS
except Exception:
    ENVIRONMENT_REGIONS = None
    # If import fails, fall back to full frame inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a Diablo screenshot")
    parser.add_argument("--model", required=True, type=str, help="Path to trained YOLO model (.pt or .onnx)")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="data/screenshots/outputs/yolo_inference.jpg",
                        help="Path to save annotated output")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--use_playfield", action="store_true", help="Crop to playfield region before inference")
    return parser.parse_args()


def crop_playfield(img):
    if ENVIRONMENT_REGIONS is None:
        return img
    region = ENVIRONMENT_REGIONS.get("playfield")
    if region is None:
        return img
    h, w = img.shape[:2]
    x, y, rw, rh = region.get_bounds(h, w)
    return img[y:y+rh, x:x+rw]


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    if args.use_playfield:
        img_infer = crop_playfield(img)
    else:
        img_infer = img

    print(f"🚀 Running inference: {model_path}")
    print(f"   Image: {image_path}")
    print(f"   Use playfield: {args.use_playfield}")

    model = YOLO(str(model_path))
    results = model.predict(source=img_infer, conf=args.conf, verbose=False)

    # Annotate
    annotated = results[0].plot()

    # If we cropped, paste back to original size for convenience
    if args.use_playfield and img_infer.shape != img.shape:
        x, y, rw, rh = ENVIRONMENT_REGIONS['playfield'].get_bounds(img.shape[0], img.shape[1])
        canvas = img.copy()
        canvas[y:y+rh, x:x+rw] = cv2.resize(annotated, (rw, rh))
        annotated = canvas

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    # Print detections
    print("✅ Inference done")
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
        print(f" - cls={cls_id} conf={conf:.2f} bbox={xyxy}")

    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()
