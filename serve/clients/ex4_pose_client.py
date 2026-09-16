"""Exercise 4c client — POST /pose to api/ex4_pose_app.py; overlay stays local."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import httpx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO,
    MODEL_NAME,
    add_sample,
    empty_totals,
    finalize_metrics,
    load_frames,
    now,
    overlay_pose_bgr,
    print_metrics,
    save_overlay,
    write_metrics_json,
)


def post_pose(client, url, frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    resp = client.post(url, files=files, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 4c client for api/ex4_pose_app.py (POST /pose)."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--save-overlay",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "pose_fastapi.jpg",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "pose_fastapi.json",
    )
    args = parser.parse_args()

    frames = load_frames(args.video, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No frames read from video.")

    pose_url = args.url.rstrip("/") + "/pose"
    health_url = args.url.rstrip("/") + "/health"

    with httpx.Client() as client:
        health = client.get(health_url, timeout=10.0)
        health.raise_for_status()
        print("GET /health", health.json())

        warmup_n = min(args.warmup, len(frames))
        for i in range(warmup_n):
            post_pose(client, pose_url, frames[i])

        totals = empty_totals()
        overlay_bgr = None
        fps_start = now()
        for frame in frames:
            pipe0 = now()
            pre0 = now()
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("JPEG encode failed")
            files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            pre1 = now()

            inf0 = now()
            resp = client.post(pose_url, files=files, timeout=30.0)
            resp.raise_for_status()
            body = resp.json()
            inf1 = now()

            post0 = now()
            people = []
            if body.get("keypoints"):
                people = [
                    {
                        "box": body["box"],
                        "conf": body["conf"],
                        "keypoints": body["keypoints"],
                    }
                ]
            fps_end = now()
            dt = fps_end - fps_start
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_start = fps_end
            annotated = overlay_pose_bgr(frame, people, fps=fps)
            if overlay_bgr is None and people:
                overlay_bgr = annotated
            post1 = now()
            pipe1 = now()
            add_sample(totals, pre1 - pre0, inf1 - inf0, post1 - post0, pipe1 - pipe0)

    if overlay_bgr is None:
        overlay_bgr = frames[0]

    metrics = finalize_metrics(totals)
    print_metrics(f"Exercise 4c FastAPI /pose ({MODEL_NAME}):", metrics)
    write_metrics_json(
        args.json_out,
        metrics,
        extra={
            "path": "fastapi_pose",
            "protocol": "fastapi_http",
            "exercise": "4c",
            "warmup": warmup_n,
            "video": str(args.video),
            "model": MODEL_NAME,
        },
    )
    save_overlay(args.save_overlay, overlay_bgr)


if __name__ == "__main__":
    main()
