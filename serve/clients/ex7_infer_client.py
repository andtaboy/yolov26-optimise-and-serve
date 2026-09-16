"""Exercise 7b — POST /infer (one squat frame) to api/ex7_infer_app.py."""

from __future__ import annotations

import argparse
import json
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
    load_frames,
    now,
    overlay_infer_bgr,
    save_overlay,
)


def post_infer(client, url, frame, timeout=60.0):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    resp = client.post(url, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def people_from_body(body):
    pose_people = []
    pose = body.get("pose") or {}
    if pose.get("keypoints"):
        pose_people = [
            {
                "box": pose.get("box"),
                "conf": pose.get("conf"),
                "keypoints": pose["keypoints"],
            }
        ]
    seg_people = []
    seg = body.get("seg") or {}
    if seg.get("polygons"):
        seg_people = [seg]
    return pose_people, seg_people


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 7b client for api/ex7_infer_app.py (POST /infer, one frame)."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--save-overlay",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ex7_infer.jpg",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ex7_infer.json",
    )
    args = parser.parse_args()

    frames = load_frames(args.video, max_frames=args.frame_index + 1)
    if not frames:
        raise RuntimeError("No frames read from video.")
    frame = frames[args.frame_index]

    infer_url = args.url.rstrip("/") + "/infer"
    health_url = args.url.rstrip("/") + "/health"

    with httpx.Client() as client:
        health = client.get(health_url, timeout=10.0)
        health.raise_for_status()
        print("GET /health", health.json())
        t0 = now()
        body = post_infer(client, infer_url, frame)
        wall_ms = (now() - t0) * 1000.0

    server_ms = body.get("server_ms") or {}
    print("Exercise 7b  POST /infer  (not /pose, not Triton :8000)")
    print(f"  n_pose={body.get('n_pose')}  n_seg_person={body.get('n_seg_person')}")
    print(f"  client POST wall     {wall_ms:.2f} ms")
    print(f"  server gather_ms     {server_ms.get('gather_ms', 0):.2f} ms")
    print(f"  server pose_grpc_ms  {server_ms.get('pose_grpc_ms', 0):.2f} ms")
    print(f"  server seg_grpc_ms   {server_ms.get('seg_grpc_ms', 0):.2f} ms")
    print(f"  server post_ms       {server_ms.get('post_ms', 0):.2f} ms")
    print("  Compare gather_ms to Exercise 6b parallel_ms (~93 ms on this 4090).")

    pose_people, seg_people = people_from_body(body)
    overlay = overlay_infer_bgr(frame, pose_people, seg_people, fps=None)
    save_overlay(args.save_overlay, overlay)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exercise": "7b",
        "path": "fastapi_infer",
        "client_post_ms": wall_ms,
        **body,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
