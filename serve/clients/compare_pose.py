"""Exercise 4b helper — in-process Ultralytics .pt / FP16 engine timings.

Run this in the Windows conda env that has ultralytics + TensorRT.
Triton HTTP/gRPC JSON files are written separately from WSL.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO,
    REPO_ROOT,
    add_sample,
    empty_totals,
    finalize_metrics,
    load_frames,
    now,
    print_metrics,
    write_metrics_json,
)

OPTIMIZE = REPO_ROOT / "optimize"
POSE_PT = OPTIMIZE / "models" / "yolo26m-pose.pt"
POSE_ENGINE = OPTIMIZE / "models" / "yolo26m-pose-fp16.engine"


def time_yolo(model, frames, warmup):
    import cv2
    import torch

    warmup_n = min(warmup, len(frames))
    for i in range(warmup_n):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        model(rgb, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    totals = empty_totals()
    fps_start = now()
    for frame in frames:
        pipe0 = now()
        pre0 = now()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pre1 = now()

        inf0 = now()
        results = model(rgb, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf1 = now()

        post0 = now()
        _ = results[0].keypoints if results else None
        fps_end = now()
        _fps = 1.0 / (fps_end - fps_start) if fps_end > fps_start else 0.0
        fps_start = fps_end
        post1 = now()
        pipe1 = now()
        add_sample(totals, pre1 - pre0, inf1 - inf0, post1 - post0, pipe1 - pipe0)
    return finalize_metrics(totals), warmup_n


def main():
    parser = argparse.ArgumentParser(
        description="Write pose_pt.json / pose_engine.json from in-process YOLO (conda)."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--pose-pt", type=Path, default=POSE_PT)
    parser.add_argument("--pose-engine", type=Path, default=POSE_ENGINE)
    parser.add_argument("--skip-pt", action="store_true")
    parser.add_argument("--skip-engine", action="store_true")
    args = parser.parse_args()

    os.chdir(OPTIMIZE)
    from ultralytics import YOLO

    frames = load_frames(args.video, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No frames read from video.")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_pt:
        model = YOLO(str(args.pose_pt))
        metrics, warmup_n = time_yolo(model, frames, args.warmup)
        print_metrics(f"YOLO in-process ({args.pose_pt.name}):", metrics)
        write_metrics_json(
            DEFAULT_OUTPUT_DIR / "pose_pt.json",
            metrics,
            extra={
                "path": "yolo_pt",
                "protocol": "in_process",
                "warmup": warmup_n,
                "video": str(args.video),
                "model": args.pose_pt.name,
            },
        )

    if not args.skip_engine:
        model = YOLO(str(args.pose_engine))
        metrics, warmup_n = time_yolo(model, frames, args.warmup)
        print_metrics(f"YOLO in-process ({args.pose_engine.name}):", metrics)
        write_metrics_json(
            DEFAULT_OUTPUT_DIR / "pose_engine.json",
            metrics,
            extra={
                "path": "yolo_fp16_engine",
                "protocol": "in_process",
                "warmup": warmup_n,
                "video": str(args.video),
                "model": args.pose_engine.name,
            },
        )


if __name__ == "__main__":
    main()
