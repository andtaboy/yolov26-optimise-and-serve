import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

colours = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (255, 102, 178),
    (102, 255, 102),
]

mask_color = np.random.randint(0, 256, size=3)


def draw_keypoints(image, keypoints, o_h, o_w):
    for i, colour in enumerate(colours):
        x, y = keypoints[i][0], keypoints[i][1]
        cx, cy = int(x * o_w), int(y * o_h)
        cv2.circle(image, (cx, cy), 5, colour, -1)
    return image


def load_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video at {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def load_yolo_if_exists(path):
    path = Path(path)
    if not path.exists():
        print(f"Skipping missing model: {path}")
        return None
    return YOLO(str(path))


def overlay_pose(frame_rgb, results, fps):
    if not results or results[0].keypoints is None:
        return frame_rgb
    keypoints = results[0].keypoints.xyn.cpu().numpy()
    if keypoints.size == 0:
        return frame_rgb
    keypoints = keypoints[0]
    frame_rgb = draw_keypoints(frame_rgb, keypoints, frame_rgb.shape[0], frame_rgb.shape[1])
    cv2.putText(
        frame_rgb,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)


def overlay_seg(frame_rgb, results):
    if not results or results[0].masks is None:
        return frame_rgb
    masks_xy = results[0].masks.xy
    classes = results[0].boxes.cls.cpu().numpy()
    contours = [
        np.round(masks_xy[i]).astype(np.int32).reshape((-1, 1, 2))
        for i in range(len(classes))
        if int(classes[i]) == 0 and len(masks_xy[i])
    ]
    if not contours:
        return frame_rgb
    mask = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, contours, 1)
    colored_mask = np.zeros_like(frame_rgb, dtype=np.uint8)
    colored_mask[mask == 1] = mask_color
    return cv2.addWeighted(frame_rgb, 0.5, colored_mask, 0.5, 0)


def run_inference(model, frames, task):
    total_inference_time = 0.0
    total_preprocess_time = 0.0
    total_postprocess_time = 0.0
    total_pipeline_time = 0.0
    total_frames = 0
    fps_start_time = 0.0

    for frame in frames:
        pipeline_start = time.perf_counter()
        pre_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pre_end = time.perf_counter()

        start = time.perf_counter()
        results = model(frame_rgb, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        post_start = time.perf_counter()
        fps_end_time = time.perf_counter()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff if time_diff > 0 else 0.0
        fps_start_time = fps_end_time

        if task == "pose":
            overlay_pose(frame_rgb, results, fps)
        else:
            overlay_seg(frame_rgb, results)
        post_end = time.perf_counter()
        pipeline_end = time.perf_counter()

        total_inference_time += end - start
        total_preprocess_time += pre_end - pre_start
        total_postprocess_time += post_end - post_start
        total_pipeline_time += pipeline_end - pipeline_start
        total_frames += 1

    n = max(total_frames, 1)
    return {
        "frames": total_frames,
        "infer_s": total_inference_time,
        "infer_ms": (total_inference_time / n) * 1000.0,
        "pre_s": total_preprocess_time,
        "pre_ms": (total_preprocess_time / n) * 1000.0,
        "post_s": total_postprocess_time,
        "post_ms": (total_postprocess_time / n) * 1000.0,
        "pipe_s": total_pipeline_time,
        "pipe_ms": (total_pipeline_time / n) * 1000.0,
        "infer_fps": total_frames / total_inference_time if total_inference_time > 0 else 0.0,
        "pipe_fps": total_frames / total_pipeline_time if total_pipeline_time > 0 else 0.0,
    }


def print_metrics(title, metrics):
    print(title)
    print(f"  Total frames processed: {metrics['frames']}")
    print(f"  Total inference time: {metrics['infer_s']:.2f} s")
    print(f"  Average inference time per frame: {metrics['infer_ms']:.2f} ms")
    print(f"  Total pre-processing time: {metrics['pre_s']:.2f} s")
    print(f"  Average pre-processing time per frame: {metrics['pre_ms']:.2f} ms")
    print(f"  Total post-processing time: {metrics['post_s']:.2f} s")
    print(f"  Average post-processing time per frame: {metrics['post_ms']:.2f} ms")
    print(f"  Total pipeline time: {metrics['pipe_s']:.2f} s")
    print(f"  Average pipeline time per frame: {metrics['pipe_ms']:.2f} ms")
    print(f"  Average FPS over video: {metrics['infer_fps']:.2f}")
    print(f"  Pipeline FPS over video: {metrics['pipe_fps']:.2f}")


def warmup(models, frame, rounds=3):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for _ in range(rounds):
        for model in models:
            model(rgb, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def compare_task(task, label, pt_path, engine_path, frames):
    print("=" * 60)
    print(f"{label}: pre-optimised PyTorch vs optimised TensorRT")
    print("=" * 60)

    pt_model = load_yolo_if_exists(pt_path)
    engine_model = load_yolo_if_exists(engine_path)
    if pt_model is None or engine_model is None:
        print(f"Skipping {label} comparison; need both {pt_path} and {engine_path}.")
        print("")
        return

    print(f"Warming up {label} models...")
    warmup([pt_model, engine_model], frames[0])

    print(f"Running TensorRT {label}...")
    engine_metrics = run_inference(engine_model, frames, task)
    print(f"Running PyTorch {label}...")
    pt_metrics = run_inference(pt_model, frames, task)

    print_metrics(f"TensorRT {label} ({engine_path}):", engine_metrics)
    print("")
    print_metrics(f"PyTorch {label} ({pt_path}):", pt_metrics)
    if pt_metrics["infer_ms"] > 0:
        speedup = pt_metrics["infer_ms"] / engine_metrics["infer_ms"]
        print("")
        print(f"  Inference speedup (pt / engine): {speedup:.2f}x")
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="media/002_air_squat_front_smartphone_one_rep.mp4")
    parser.add_argument("--pose_engine", type=str, default="models/yolo26m-pose-fp16.engine")
    parser.add_argument("--seg_engine", type=str, default="models/yolo26m-seg-int8-fp16.engine")
    parser.add_argument("--pose_pt", type=str, default="models/yolo26m-pose.pt")
    parser.add_argument("--seg_pt", type=str, default="models/yolo26m-seg.pt")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    frames = load_frames(args.video, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No frames read from video.")

    compare_task("pose", "POSE", args.pose_pt, args.pose_engine, frames)
    compare_task("seg", "SEG", args.seg_pt, args.seg_engine, frames)


if __name__ == "__main__":
    main()
