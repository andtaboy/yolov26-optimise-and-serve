"""Exercise 4c — pose-only FastAPI gateway to Triton gRPC. Bind :8080, never 8000.

This is the training-wheels product API (pose only). Exercise 7 is api/ex7_infer_app.py
(pose + seg on POST /infer). Do not run both on :8080 at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

CLIENTS = Path(__file__).resolve().parents[1] / "clients"
if str(CLIENTS) not in sys.path:
    sys.path.insert(0, str(CLIENTS))

from common import MODEL_NAME, decode_output0, letterbox, now  # noqa: E402

TRITON_GRPC = "127.0.0.1:8001"

app = FastAPI(title="Exercise 4c pose-only gateway", version="0.1.0")


def _client():
    return grpcclient.InferenceServerClient(url=TRITON_GRPC)


def _grpc_infer(tensor):
    client = _client()
    inp = grpcclient.InferInput("images", list(tensor.shape), "FP32")
    inp.set_data_from_numpy(tensor)
    out = grpcclient.InferRequestedOutput("output0")
    t0 = now()
    result = client.infer(model_name=MODEL_NAME, inputs=[inp], outputs=[out])
    grpc_s = now() - t0
    return result.as_numpy("output0"), grpc_s


@app.get("/health")
def health():
    try:
        client = _client()
        ready = bool(client.is_server_ready())
        model_ready = bool(client.is_model_ready(MODEL_NAME))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not ready or not model_ready:
        raise HTTPException(
            status_code=503,
            detail={"server_ready": ready, "model_ready": model_ready},
        )
    return {
        "exercise": "4c",
        "server_ready": True,
        "model_ready": True,
        "model": MODEL_NAME,
    }


@app.post("/pose")
async def pose(file: UploadFile = File(...)):
    raw = await file.read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    tensor, meta = letterbox(frame)
    try:
        output0, grpc_s = _grpc_infer(tensor)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    people = decode_output0(output0, meta, conf_thres=0.25)
    top = people[0] if people else None
    body = {
        "exercise": "4c",
        "model": MODEL_NAME,
        "n_dets": len(people),
        "box": top["box"] if top else None,
        "conf": top["conf"] if top else None,
        "keypoints": top["keypoints"] if top else [],
        "server_grpc_ms": grpc_s * 1000.0,
        "orig_hw": [meta["orig_h"], meta["orig_w"]],
    }
    return JSONResponse(body)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Exercise 4c pose-only FastAPI gateway.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
