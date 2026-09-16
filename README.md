# yolov26-optimise-and-serve

YOLO26m **pose** and **segmentation**: compile and validate TensorRT engines on Windows, then serve them with **NVIDIA Triton Inference Server** (Linux container) and a thin FastAPI product API.

This repository does **not** run Triton. The server lives in a WSL workspace (`/home/andre/triton`) mounted into Docker. Python here is either Ultralytics/TensorRT (optimize) or `tritonclient` / FastAPI (serve).

**Serve decision (measured on an RTX 4090 Laptop, Ultralytics `model.val()`, `imgsz=640`, `batch=1`):**

| Task | Artifact to serve | Quality vs `.pt` | Val inference |
|---|---|---|---|
| Pose | `yolo26m-pose-fp16.engine` | pose mAP50-95 **0.6953** (delta 0) | 10.0 ms → 3.0 ms |
| Seg | `yolo26m-seg-int8-fp16.engine` | mask mAP50-95 0.4252 → **0.3963** (−0.0289) | 11.7 ms → 3.3 ms |
| Pose INT8 / mixed | do **not** serve | pose mAP50-95 collapses to ~0.41 | faster, unusable OKS |

Filename trap: `models/yolo26m-pose.engine` is **not** FP16. It is (or was) the mixed INT8 engine.

Triton does not make a single-stream loop faster than in-process `YOLO("*.engine")`. Its job is isolation, two models on one request, and multi-client scheduling.

## Layout

| Path | Role | Runtime |
|---|---|---|
| [`optimize/`](optimize/README.md) | Export, COCO val, in-process video FPS | Windows conda `model_perfromance_eval` (Python 3.8 + TensorRT) |
| [`serve/`](serve/README.md) | Clients and FastAPI in front of Triton | WSL Python 3.10+ with `tritonclient[all]` |
| `/home/andre/triton` (**not in git**) | Triton mount: `model_repository/` + `build/` | Docker `nvcr.io/nvidia/tritonserver:26.08-py3` |

Do not put a `model_repository` inside this Windows tree. TensorRT plans are OS- and version-locked: a Windows `.engine` will not load in Linux Triton 11.x. Rebuild ONNX → plan **inside** the same NGC image that runs `tritonserver`.

Weights (`.pt`, `.engine`, `.onnx`), videos, COCO datasets, and local notes under `docs/` are gitignored.

## Requirements

**Optimize (Windows)**

```text
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe
```

Keep `D:` mounted if that is where COCO lives. Details: [`optimize/README.md`](optimize/README.md).

**Serve (WSL)**

```bash
cd /mnt/d/SMARTCAM/python_projects/yolov26-optimise-and-serve/serve
python3 -m pip install -r requirements.txt
```

**Triton (WSL + Docker GPU)**

- Docker Desktop (or Docker Engine) with NVIDIA GPU
- Image: `nvcr.io/nvidia/tritonserver:26.08-py3` (not `*-vllm-python-py3`)
- WSL workspace (example):

```text
/home/andre/triton/
  model_repository/          # --model-repository (only real models)
    yolo26m_pose/
      config.pbtxt
      1/model.plan
    yolo26m_seg/
      config.pbtxt
      1/model.plan
  build/                     # ONNX + trtexec scratch; never inside model_repository
```

## 1. Optimize (compile and val)

From `optimize/`:

```text
python export_pose.py --fp16          → models/yolo26m-pose-fp16.engine
python export_seg.py                  → models/yolo26m-seg-int8-fp16.engine
python val_compare.py                 → COCO mAP table (accuracy gate)
python video_compare.py               → squat-clip FPS only (no mAP)
```

Accuracy is COCO `model.val()`, not video FPS. Pose and seg use **different** label formats; pointing seg val at pose keypoints yields mAP 0 and “corrupt” labels. Full notes in [`optimize/README.md`](optimize/README.md).

Copy **ONNX** (not the Windows `.engine`) into the Triton `build/` folder when you need a Linux plan:

```bash
cp /mnt/d/SMARTCAM/python_projects/yolov26-optimise-and-serve/optimize/models/yolo26m-pose.onnx \
   /home/andre/triton/build/yolo26m-pose.onnx
cp /mnt/d/SMARTCAM/python_projects/yolov26-optimise-and-serve/optimize/models/yolo26m-seg.onnx \
   /home/andre/triton/build/yolo26m-seg.onnx
```

## 2. Triton workspace

Create the skeleton:

```bash
mkdir -p /home/andre/triton/model_repository/yolo26m_pose/1
mkdir -p /home/andre/triton/model_repository/yolo26m_seg/1
mkdir -p /home/andre/triton/build
```

Start the server (named container, published ports — `--net=host` is unreliable on Docker Desktop):

```bash
docker run --gpus all --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/andre/triton:/triton \
  nvcr.io/nvidia/tritonserver:26.08-py3 \
  tritonserver --model-repository=/triton/model_repository
```

| Port | What |
|---|---|
| 8000 | Triton HTTP (KServe `/v2`, stats) |
| 8001 | Triton gRPC |
| 8002 | Prometheus metrics |
| 8080 | FastAPI in this repo (never put FastAPI on 8000) |

Later: `docker start triton` / `docker stop triton`.

Do **not** pass `/triton` as `--model-repository`. Triton treats every top-level directory under that path as a model. `build/` must stay a **sibling** of `model_repository`. A scratch dir inside the scanned repo (`_build`, etc.) causes `failed to load all models` and the process exits.

### Build plans in the container

`trtexec` does not create parent directories. Save to `build/` first, then copy to `1/model.plan`. Ultralytics `.engine` files have a JSON metadata prefix; Triton wants raw `model.plan`.

```bash
docker exec triton trtexec \
  --onnx=/triton/build/yolo26m-pose.onnx \
  --saveEngine=/triton/build/yolo26m-pose-triton.plan

docker exec triton trtexec --loadEngine=/triton/build/yolo26m-pose-triton.plan
# expect &&&& PASSED and binding names/shapes

cp /home/andre/triton/build/yolo26m-pose-triton.plan \
   /home/andre/triton/model_repository/yolo26m_pose/1/model.plan
```

Same pattern for seg (`yolo26m-seg.onnx` → `yolo26m_seg/1/model.plan`). Seg has **two** outputs. `Cannot write to FileStreamWriter` means the output path’s parent dir does not exist, not that ONNX failed to compile.

### `config.pbtxt`

`max_batch_size: 0` because these plans already include a leading batch of 1. Do not enable `dynamic_batching` without exporting a batched ONNX.

**Pose** (`yolo26m_pose`) — measured bindings: `images` `[1,3,640,640]` fp32, `output0` `[1,300,57]` fp32 (xyxy + conf + cls + 17×3 keypoints).

**Seg** (`yolo26m_seg`) — `images` `[1,3,640,640]` fp32, `output0` `[1,300,38]` fp32, `output1` `[1,32,160,160]` fp32. Two `output { }` blocks in `output [ ... ]` **must be comma-separated**. A missing comma is `Expected ",", found "{"` and **unloads every model** (default strict readiness).

Model folder names use underscores (`yolo26m_pose`, `yolo26m_seg`), matching `name:` in the config.

Restart Triton after adding a model. Success looks like:

```text
| yolo26m_pose | 1 | READY |
| yolo26m_seg  | 1 | READY |
```

```bash
curl -s http://127.0.0.1:8000/v2/health/ready
curl -s http://127.0.0.1:8000/v2/models/yolo26m_pose
curl -s http://127.0.0.1:8000/v2/models/yolo26m_seg/config
```

## 3. Serve clients (`serve/`)

Work from `serve/`. These scripts do not load `.pt` or `.engine`. Letterbox matches Ultralytics (640, pad 114). JSON timings go to `serve/outputs/run_1/` (gitignored).

Do not run `ex4_pose_app.py` and `ex7_infer_app.py` on `:8080` at the same time.

| Script | What it does |
|---|---|
| `clients/compare_pose.py` | In-process `.pt` vs FP16 engine (Windows conda) |
| `clients/infer_http.py` | Pose via Triton HTTP `:8000` |
| `clients/infer_grpc.py` | Pose via Triton gRPC `:8001` |
| `clients/print_table.py` | Sequential timings + load table |
| `api/ex4_pose_app.py` | FastAPI `:8080`, `POST /pose` only |
| `clients/ex4_pose_client.py` | JPEG loop against `/pose`; overlay on the client |
| `clients/load_pose.py` | Concurrent `POST /pose` (product hop) |
| `clients/load_triton.py` | Concurrent native HTTP/gRPC load + `/v2` queue/compute stats |
| `clients/infer_pose_seg_grpc.py` | One frame: serial pose+seg vs `asyncio.gather` |
| `api/ex7_infer_app.py` | FastAPI `:8080`, `POST /infer` (pose + person polygons) |
| `clients/ex7_infer_client.py` | One squat frame → `/infer` |
| `clients/ex7_video_client.py` | Video = POST **frames**, not an mp4 blob |

`GET /health` on the FastAPI apps includes `"exercise": "4c"` or `"exercise": "7"` so you know which process is bound.

```bash
python3 clients/infer_http.py
python3 clients/infer_grpc.py
python3 api/ex4_pose_app.py          # other terminal:
python3 clients/ex4_pose_client.py
python3 clients/print_table.py
```

```bash
python3 clients/load_pose.py --concurrency 1 --instance-group 1
python3 clients/load_triton.py --protocol both --concurrency 8 --instance-group 1
python3 clients/infer_pose_seg_grpc.py
```

```bash
python3 api/ex7_infer_app.py         # stop the 4c app first
python3 clients/ex7_infer_client.py
python3 clients/ex7_video_client.py --max-frames 96
```

`POST /infer` letterboxes **once**, then `asyncio.gather`s pose and seg over gRPC. The JSON is keypoints plus **person-class** polygons (COCO class 0). Overlay stays on the client. `gather_ms` should sit near the native dual-gRPC time; extra POST latency is JPEG, Uvicorn, mask decode, and JSON.

### How to read load numbers

Sequential `infer_fps` (one client) is not the same as load **requests per second** (many in-flight calls). Do not mix them on one table.

On this 4090, FastAPI `/pose` saturates near **13 requests/s** (`queue_ms` ~0 — wait is in FastAPI, not Triton). Native Triton HTTP reaches ~**35 requests/s**. `instance_group count: 2` does not fix the FastAPI hop. `inf/exec = 1.00` means no dynamic batching (expected for these batch-1 plans).

| Header | Meaning |
|---|---|
| conc | concurrency (in-flight requests from that script) |
| inst | `instance_group.count` you set in `config.pbtxt` |
| infer_ms | mean client wait for **that hop** |
| rps | requests per second (completed / wall clock) |
| queue_ms | Triton `/v2` stats: wait inside Triton |
| compute_ms | Triton `/v2` stats: GPU `compute_infer` (not in-process ~5 ms) |

## Pitfalls

- NGC tag `*-vllm-python-py3` is the wrong image; use `26.08-py3`.
- Host TensorRT (e.g. 10.15 on Windows) ≠ container TensorRT 11.2.1. Rebuild in-container; do not rebuild again just to change mounts if `model.plan` already loads READY on that tag.
- One bad `config.pbtxt` takes the **whole** server down.
- FastAPI is not Triton. Clients of the product talk to `:8080`; FastAPI talks to `:8001`.
- A “video service” is a loop of `1x3x640x640` frames. Do not POST an mp4 as one infer.

## License / data

Model weights and the squat clip are not in git. Obtain Ultralytics YOLO26m `.pt` files yourself and run `optimize/` exports locally. The Triton workspace stays on the machine that has the GPU.