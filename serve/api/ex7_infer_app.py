"""Exercise 7 — FastAPI product API: one JPEG, concurrent pose + seg via Triton gRPC.

Bind :8080, never :8000. Exercise 4c is api/ex4_pose_app.py (pose only). Do not run
both on :8080 at once. Overlay stays on the client.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc.aio as grpcclient
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

CLIENTS = Path(__file__).resolve().parents[1] / "clients"
if str(CLIENTS) not in sys.path:
    sys.path.insert(0, str(CLIENTS))

from common import (  # noqa: E402
    POSE_MODEL_NAME,
    SEG_MODEL_NAME,
    decode_output0,
    decode_seg_person,
    letterbox,
    now,
)

TRITON_GRPC = "127.0.0.1:8001"

app = FastAPI(title="Exercise 7 pose+seg product API", version="0.1.0")

_client = None


def _input(tensor):
    inp = grpcclient.InferInput("images", list(tensor.shape), "FP32")
    inp.set_data_from_numpy(tensor)
    return inp


async def _get_client():
    global _client
    if _client is None:
        _client = grpcclient.InferenceServerClient(url=TRITON_GRPC)
    return _client


async def _infer_pose(client, tensor):
    t0 = now()
    result = await client.infer(
        model_name=POSE_MODEL_NAME,
        inputs=[_input(tensor)],
        outputs=[grpcclient.InferRequestedOutput("output0")],
    )
    return result.as_numpy("output0"), now() - t0


async def _infer_seg(client, tensor):
    t0 = now()
    result = await client.infer(
        model_name=SEG_MODEL_NAME,
        inputs=[_input(tensor)],
        outputs=[
            grpcclient.InferRequestedOutput("output0"),
            grpcclient.InferRequestedOutput("output1"),
        ],
    )
    return result.as_numpy("output0"), result.as_numpy("output1"), now() - t0


@app.get("/health")
async def health():
    try:
        client = await _get_client()
        ready = bool(await client.is_server_ready())
        pose_ready = bool(await client.is_model_ready(POSE_MODEL_NAME))
        seg_ready = bool(await client.is_model_ready(SEG_MODEL_NAME))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not ready or not pose_ready or not seg_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "server_ready": ready,
                "pose_ready": pose_ready,
                "seg_ready": seg_ready,
            },
        )
    return {
        "exercise": "7",
        "server_ready": True,
        "pose_ready": True,
        "seg_ready": True,
        "pose_model": POSE_MODEL_NAME,
        "seg_model": SEG_MODEL_NAME,
    }


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    raw = await file.read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    t_pre = now()
    tensor, meta = letterbox(frame)
    letterbox_s = now() - t_pre

    try:
        client = await _get_client()
        t_g = now()
        (pose_out, pose_s), (seg0, seg1, seg_s) = await _gather_both(client, tensor)
        gather_s = now() - t_g
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    t_post = now()
    pose_people = decode_output0(pose_out, meta, conf_thres=0.25)
    seg_people = decode_seg_person(seg0, seg1, meta, conf_thres=0.25)
    post_s = now() - t_post

    pose_top = pose_people[0] if pose_people else None
    seg_top = seg_people[0] if seg_people else None
    body = {
        "exercise": "7",
        "orig_hw": [meta["orig_h"], meta["orig_w"]],
        "n_pose": len(pose_people),
        "n_seg_person": len(seg_people),
        "pose": {
            "box": pose_top["box"] if pose_top else None,
            "conf": pose_top["conf"] if pose_top else None,
            "keypoints": pose_top["keypoints"] if pose_top else [],
        },
        "seg": {
            "cls": seg_top["cls"] if seg_top else None,
            "class_name": seg_top["class_name"] if seg_top else "person",
            "conf": seg_top["conf"] if seg_top else None,
            "box": seg_top["box"] if seg_top else None,
            "polygons": seg_top["polygons"] if seg_top else [],
        },
        "server_ms": {
            "letterbox_ms": letterbox_s * 1000.0,
            "gather_ms": gather_s * 1000.0,
            "pose_grpc_ms": pose_s * 1000.0,
            "seg_grpc_ms": seg_s * 1000.0,
            "post_ms": post_s * 1000.0,
        },
    }
    return JSONResponse(body)


async def _gather_both(client, tensor):
    return await asyncio.gather(_infer_pose(client, tensor), _infer_seg(client, tensor))


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Exercise 7 FastAPI product API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
