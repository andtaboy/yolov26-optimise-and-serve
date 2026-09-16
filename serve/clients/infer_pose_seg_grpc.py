"""Exercise 6b — serial vs concurrent gRPC pose + seg on one frame. Not FastAPI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import tritonclient.grpc.aio as grpcclient

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO,
    POSE_MODEL_NAME,
    SEG_MODEL_NAME,
    letterbox,
    load_frames,
    now,
)


def _input(tensor):
    inp = grpcclient.InferInput("images", list(tensor.shape), "FP32")
    inp.set_data_from_numpy(tensor)
    return inp


async def infer_pose(client, tensor):
    result = await client.infer(
        model_name=POSE_MODEL_NAME,
        inputs=[_input(tensor)],
        outputs=[grpcclient.InferRequestedOutput("output0")],
    )
    return {"output0": result.as_numpy("output0")}


async def infer_seg(client, tensor):
    result = await client.infer(
        model_name=SEG_MODEL_NAME,
        inputs=[_input(tensor)],
        outputs=[
            grpcclient.InferRequestedOutput("output0"),
            grpcclient.InferRequestedOutput("output1"),
        ],
    )
    return {
        "output0": result.as_numpy("output0"),
        "output1": result.as_numpy("output1"),
    }


def _shapes(prefix, tensors):
    for name, arr in tensors.items():
        print(f"  {prefix} {name}: shape={arr.shape} dtype={arr.dtype}")


async def run(url, tensor, warmup):
    client = grpcclient.InferenceServerClient(url=url)
    if not await client.is_server_ready():
        raise RuntimeError(f"Triton gRPC not ready at {url}")
    if not await client.is_model_ready(POSE_MODEL_NAME):
        raise RuntimeError(f"{POSE_MODEL_NAME} not READY")
    if not await client.is_model_ready(SEG_MODEL_NAME):
        raise RuntimeError(f"{SEG_MODEL_NAME} not READY")

    for _ in range(warmup):
        await infer_pose(client, tensor)
        await infer_seg(client, tensor)

    pose_s = seg_s = 0.0
    t0 = now()
    t_pose = now()
    pose_serial = await infer_pose(client, tensor)
    pose_s = now() - t_pose
    t_seg = now()
    seg_serial = await infer_seg(client, tensor)
    seg_s = now() - t_seg
    serial_s = now() - t0

    t0 = now()
    pose_p_s = seg_p_s = 0.0

    async def timed_pose():
        nonlocal pose_p_s
        t = now()
        out = await infer_pose(client, tensor)
        pose_p_s = now() - t
        return out

    async def timed_seg():
        nonlocal seg_p_s
        t = now()
        out = await infer_seg(client, tensor)
        seg_p_s = now() - t
        return out

    pose_par, seg_par = await asyncio.gather(timed_pose(), timed_seg())
    parallel_s = now() - t0

    _shapes("serial pose", pose_serial)
    _shapes("serial seg", seg_serial)
    _shapes("parallel pose", pose_par)
    _shapes("parallel seg", seg_par)

    return {
        "pose_serial_ms": pose_s * 1000.0,
        "seg_serial_ms": seg_s * 1000.0,
        "serial_ms": serial_s * 1000.0,
        "pose_parallel_ms": pose_p_s * 1000.0,
        "seg_parallel_ms": seg_p_s * 1000.0,
        "parallel_ms": parallel_s * 1000.0,
        "speedup": (serial_s / parallel_s) if parallel_s > 0 else 0.0,
        "pose_output0": list(pose_par["output0"].shape),
        "seg_output0": list(seg_par["output0"].shape),
        "seg_output1": list(seg_par["output1"].shape),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 6b: serial pose();seg() vs asyncio.gather of both on one frame."
    )
    parser.add_argument("--url", default="127.0.0.1:8001")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "pose_seg_grpc.json",
    )
    args = parser.parse_args()

    frames = load_frames(args.video, max_frames=args.frame_index + 1)
    if not frames:
        raise RuntimeError("No frames read from video.")
    frame = frames[args.frame_index]
    tensor, _meta = letterbox(frame)

    metrics = asyncio.run(run(args.url, tensor, args.warmup))
    print("")
    print("Exercise 6b  one squat frame  gRPC :8001  (not FastAPI)")
    print(f"  serial   pose then seg     {metrics['serial_ms']:.2f} ms")
    print(f"           pose              {metrics['pose_serial_ms']:.2f} ms")
    print(f"           seg               {metrics['seg_serial_ms']:.2f} ms")
    print(f"  parallel asyncio.gather    {metrics['parallel_ms']:.2f} ms")
    print(f"           pose (in flight)  {metrics['pose_parallel_ms']:.2f} ms")
    print(f"           seg  (in flight)  {metrics['seg_parallel_ms']:.2f} ms")
    print(f"  speedup  serial/parallel   {metrics['speedup']:.2f}x")
    print("  If parallel ~= max(pose, seg) the GPU overlapped. If ~= serial, it did not.")
    print("  FastAPI dual /infer is Exercise 7.")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": "triton_grpc_pose_seg",
        "protocol": "grpc_aio",
        "url": args.url,
        "video": str(args.video),
        "frame_index": args.frame_index,
        "warmup": args.warmup,
        "pose_model": POSE_MODEL_NAME,
        "seg_model": SEG_MODEL_NAME,
        **metrics,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
