"""Exercise 4b — pose FP16 via Triton gRPC (:8001). Same pre/post as HTTP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tritonclient.grpc as grpcclient

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO,
    MODEL_NAME,
    add_sample,
    decode_output0,
    dump_output0_layout,
    empty_totals,
    finalize_metrics,
    letterbox,
    load_frames,
    now,
    overlay_pose_bgr,
    print_metrics,
    save_overlay,
    write_metrics_json,
)


def infer_one(client, tensor):
    inp = grpcclient.InferInput("images", list(tensor.shape), "FP32")
    inp.set_data_from_numpy(tensor)
    out = grpcclient.InferRequestedOutput("output0")
    result = client.infer(model_name=MODEL_NAME, inputs=[inp], outputs=[out])
    return result.as_numpy("output0")


def main():
    parser = argparse.ArgumentParser(description="Triton gRPC pose client (Exercise 4b).")
    parser.add_argument("--url", default="127.0.0.1:8001")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--save-overlay",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "pose_grpc.jpg",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "pose_grpc.json",
    )
    args = parser.parse_args()

    frames = load_frames(args.video, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No frames read from video.")

    client = grpcclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton gRPC not ready at {args.url}")

    warmup_n = min(args.warmup, len(frames))
    for i in range(warmup_n):
        tensor, meta = letterbox(frames[i])
        _ = infer_one(client, tensor)

    totals = empty_totals()
    overlay_bgr = None
    dumped = False
    fps_start = now()
    for frame in frames:
        pipe0 = now()
        pre0 = now()
        tensor, meta = letterbox(frame)
        pre1 = now()

        inf0 = now()
        output0 = infer_one(client, tensor)
        inf1 = now()

        post0 = now()
        if not dumped:
            dump_output0_layout(output0)
            dumped = True
        people = decode_output0(output0, meta, conf_thres=args.conf)
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
        tensor, meta = letterbox(frames[0])
        people = decode_output0(infer_one(client, tensor), meta, conf_thres=0.0)
        overlay_bgr = overlay_pose_bgr(frames[0], people, fps=0.0)

    metrics = finalize_metrics(totals)
    print_metrics(f"Triton gRPC pose ({MODEL_NAME}):", metrics)
    write_metrics_json(
        args.json_out,
        metrics,
        extra={
            "path": "triton_grpc",
            "protocol": "grpc",
            "warmup": warmup_n,
            "video": str(args.video),
            "model": MODEL_NAME,
        },
    )
    save_overlay(args.save_overlay, overlay_bgr)


if __name__ == "__main__":
    main()
