"""Exercise 7c — POST /infer per video frame to api/ex7_infer_app.py."""

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
    add_sample,
    empty_totals,
    finalize_metrics,
    load_frames,
    now,
    overlay_infer_bgr,
    print_metrics,
    save_overlay,
    write_metrics_json,
)
from ex7_infer_client import people_from_body, post_infer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 7c: client frame loop against api/ex7_infer_app.py. Not an mp4 upload."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--save-overlay",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ex7_video.jpg",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ex7_video.json",
    )
    args = parser.parse_args()

    frames = load_frames(args.video, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No frames read from video.")

    infer_url = args.url.rstrip("/") + "/infer"
    health_url = args.url.rstrip("/") + "/health"

    gather_ms = []
    with httpx.Client() as client:
        health = client.get(health_url, timeout=10.0)
        health.raise_for_status()
        print("GET /health", health.json())

        warmup_n = min(args.warmup, len(frames))
        for i in range(warmup_n):
            post_infer(client, infer_url, frames[i])

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
            resp = client.post(infer_url, files=files, timeout=60.0)
            resp.raise_for_status()
            body = resp.json()
            inf1 = now()

            post0 = now()
            pose_people, seg_people = people_from_body(body)
            fps_end = now()
            dt = fps_end - fps_start
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_start = fps_end
            annotated = overlay_infer_bgr(frame, pose_people, seg_people, fps=fps)
            if overlay_bgr is None and (pose_people or seg_people):
                overlay_bgr = annotated
            post1 = now()
            pipe1 = now()
            add_sample(totals, pre1 - pre0, inf1 - inf0, post1 - post0, pipe1 - pipe0)
            g = (body.get("server_ms") or {}).get("gather_ms")
            if g is not None:
                gather_ms.append(float(g))

    if overlay_bgr is None:
        overlay_bgr = frames[0]

    metrics = finalize_metrics(totals)
    mean_gather = sum(gather_ms) / len(gather_ms) if gather_ms else 0.0
    print_metrics("Exercise 7c FastAPI POST /infer (frame loop):", metrics)
    print(f"  mean server gather_ms: {mean_gather:.2f}  (compare to Exercise 6b ~93 ms)")
    print("  infer_ms here is client POST wall, not TensorRT.")
    write_metrics_json(
        args.json_out,
        metrics,
        extra={
            "path": "fastapi_infer_video",
            "protocol": "fastapi_http",
            "exercise": "7c",
            "warmup": warmup_n,
            "video": str(args.video),
            "mean_gather_ms": mean_gather,
        },
    )
    save_overlay(args.save_overlay, overlay_bgr)


if __name__ == "__main__":
    main()
