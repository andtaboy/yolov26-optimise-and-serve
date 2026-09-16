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

`POST /infer` letterboxes **once**, then `asyncio.gather`s pose and seg over gRPC. The JSON is keypoints plus **person-class** polygons (COCO class 0). Overlay stays on the client. `gather_ms` should sit near the native dual-gRPC time; extra POST latency is JPEG, Uvicorn, mask decode, and JSON.

## 4. What to run, in order (and which table you get)

All serve numbers below are **one RTX 4090 Laptop**, squat clip `optimize/media/002_air_squat_front_smartphone_one_rep.mp4`, 96 frames unless noted, `imgsz=640`, batch 1. JSON lands in `serve/outputs/run_1/` (gitignored). Do not put load **requests per second** on the one-stream **frames per second** table.

| Order | You run | Writes | Table |
|---|---|---|---|
| 0 | Triton READY (`yolo26m_pose`, later also `yolo26m_seg`) | server logs | — |
| 1 | `optimize/val_compare.py` then `--skip-seg --pose-engine models/yolo26m-pose-fp16.engine` | `optimize/runs/` | **A** COCO accuracy |
| 2 | `serve/clients/compare_pose.py` (Windows conda) | `pose_pt.json`, `pose_engine.json` | **B** |
| 3 | `clients/infer_http.py` then `infer_grpc.py` | `pose_http.json`, `pose_grpc.json` | **B** |
| 4 | `api/ex4_pose_app.py` + `clients/ex4_pose_client.py` | `pose_fastapi.json` | **B** |
| 5 | `clients/print_table.py` | stdout | **B** (and **C** if load JSON exists) |
| 6 | keep `ex4_pose_app.py`; `load_pose.py --concurrency 1` then `8 --instance-group 1` | `pose_load_fastapi_*.json` | **C** |
| 7 | `load_triton.py --protocol both --concurrency 1` then `8 --instance-group 1` | `pose_load_http_*.json`, `pose_load_grpc_*.json` | **C** |
| 8 | optional: `instance_group count: 2`, restart Triton, repeat 6–7 with `--instance-group 2` | `*_ig2.json` | **C** |
| 9 | both models READY; `clients/infer_pose_seg_grpc.py` | `pose_seg_grpc.json` | **D** |
| 10 | **stop** `ex4_pose_app.py`; start `api/ex7_infer_app.py` | OpenAPI `/docs` | — |
| 11 | `clients/ex7_infer_client.py` | `ex7_infer.json` | **E** (first POST is often cold) |
| 12 | `clients/ex7_video_client.py --max-frames 96` | `ex7_video.json` | **F** (use this as the product loop) |

```bash
# from serve/, Triton already READY
python3 clients/infer_http.py
python3 clients/infer_grpc.py
python3 api/ex4_pose_app.py                 # leave running
python3 clients/ex4_pose_client.py
python3 clients/load_pose.py --concurrency 1 --instance-group 1
python3 clients/load_pose.py --concurrency 8 --instance-group 1
python3 clients/load_triton.py --protocol both --concurrency 1 --instance-group 1
python3 clients/load_triton.py --protocol both --concurrency 8 --instance-group 1
python3 clients/print_table.py
python3 clients/infer_pose_seg_grpc.py
# stop ex4_pose_app.py, then:
python3 api/ex7_infer_app.py
python3 clients/ex7_infer_client.py
python3 clients/ex7_video_client.py --max-frames 96
```

In-process pose rows (step 2) on Windows:

```text
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe serve\clients\compare_pose.py
```

## 5. Example results (this 4090)

### A — COCO val (accuracy gate)

`optimize/val_compare.py`. Val `Speed:` inference milliseconds, not video FPS. Serve pose **FP16**, seg **INT8+FP16**.

**Pose** (2346 images):

| model | box mAP50-95 | pose mAP50-95 | inf ms | serve? |
|---|---:|---:|---:|---|
| `yolo26m-pose.pt` | 0.7476 | **0.6953** | 10.0 | baseline |
| `yolo26m-pose.engine` INT8-only | 0.6320 | **0.4080** | 2.4 | no |
| `yolo26m-pose-int8-fp16.engine` | 0.6324 | **0.4104** | 2.3 | no |
| `yolo26m-pose-fp16.engine` | 0.7476 | **0.6953** | 3.0 | **yes** |

**Seg** (4952 images):

| model | box mAP50-95 | mask mAP50-95 | inf ms | serve? |
|---|---:|---:|---:|---|
| `yolo26m-seg.pt` | 0.5220 | **0.4252** | 11.7 | baseline |
| `yolo26m-seg-int8-fp16.engine` | 0.4768 | **0.3963** (−0.0289) | 3.3 | **yes** |

### B — One sequential stream (pose only)

96 frames. `infer_ms` is that hop’s mean wait. `infer_fps` = 1000 / `infer_ms` for **this one client**, not aggregate capacity.

| path | frames | pre_ms | infer_ms | post_ms | pipe_ms | infer_fps | pipe_fps |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO `.pt` in-process | 96 | 1.11 | **11.63** | 0.00 | 12.74 | 86.0 | 78.5 |
| YOLO FP16 engine in-process | 96 | 1.10 | **5.15** | 0.00 | 6.25 | 194.2 | 159.9 |
| Triton HTTP `:8000` | 96 | 2.70 | **42.29** | 1.64 | 46.63 | 23.6 | 21.4 |
| Triton gRPC `:8001` | 96 | 2.69 | **55.16** | 1.68 | 59.52 | 18.1 | 16.8 |
| FastAPI `POST /pose` `:8080` | 96 | 3.59 | **84.07** | 1.68 | 89.34 | 11.9 | 11.2 |

Compile win is `.pt` → engine (~11.6 → 5.2 ms). Serving tax is engine → Triton (~5 → 42–55 ms) and FastAPI (~84 ms). On this Windows/Desktop path HTTP beat gRPC at concurrency 1.

### C — Concurrent load (same pose model, many in-flight calls)

`print_table.py` after steps 6–8. **rps** = requests per second (completed / wall). `inf/exec` stayed **1.00** (no dynamic batching).

| part | hop | conc | inst | infer_ms | p50_ms | p99_ms | requests/s | queue_ms | compute_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5a | FastAPI `/pose` | 1 | 1 | 77 | 73 | 110 | **13.0** | 0.07 | 30 |
| 5a | FastAPI `/pose` | 8 | 1 | 629 | 642 | 832 | **12.3** | 0.07 | 29 |
| 5a | FastAPI `/pose` | 8 | 2 | 665 | 694 | 869 | **11.7** | 0.08 | 29 |
| 5b | Triton gRPC | 1 | 1 | 62 | 58 | 93 | 16.2 | 0.08 | 27 |
| 5b | Triton gRPC | 8 | 1 | 276 | 295 | 394 | 27.2 | 1.43 | 13 |
| 5b | Triton gRPC | 8 | 2 | 237 | 217 | 583 | 32.1 | 0.14 | 11 |
| 5b | Triton HTTP | 1 | 1 | 52 | 50 | 111 | 19.3 | 0.05 | 20 |
| 5b | Triton HTTP | 8 | 1 | 237 | 226 | 482 | 33.0 | **14.9** | 13 |
| 5b | Triton HTTP | 8 | 2 | 219 | 215 | 398 | **35.5** | 11.7 | 13 |

FastAPI stays ~13 requests/s and `queue_ms` ~0: wait is in the product hop, not Triton. Native HTTP reaches ~35 requests/s. `count: 2` does not fix 5a.

| Header | Meaning |
|---|---|
| conc | concurrency (in-flight requests from that script) |
| inst | `instance_group.count` in `config.pbtxt` |
| infer_ms | mean client wait for **that hop** (milliseconds) |
| p50_ms / p99_ms | 50th / 99th percentile latency (milliseconds) |
| requests/s | completed calls / wall-clock time — capacity, not table B frames per second |
| queue_ms | Triton `/v2` stats: wait **inside Triton** (milliseconds) |
| compute_ms | Triton `/v2` stats: GPU `compute_infer` (milliseconds). Not in-process ~5 ms |

### D — Two models, one frame, native gRPC

`clients/infer_pose_seg_grpc.py` (no FastAPI). Shapes: pose `1×300×57`, seg `1×300×38` + `1×32×160×160`.

| | milliseconds |
|---|---:|
| serial pose then seg | 194.1 |
| pose (serial) | 81.1 |
| seg (serial) | 112.9 |
| parallel `asyncio.gather` | **138.5** |
| speedup serial / parallel | 1.40× |

A warmer repeat on this box was ~108 ms serial / ~93 ms parallel. Use **table F** `mean gather_ms` (~126 ms) as the product-side compare, not a single cold POST.

### E — Product `POST /infer`, one frame (often cold)

`clients/ex7_infer_client.py`. First call after app start is slower than the 96-frame mean.

| | |
|---|---|
| detections | `n_pose=1`, `n_seg_person=1` |
| client POST wall | 257.5 ms |
| server gather_ms | 236.9 ms |
| server pose_grpc_ms | 121.4 ms |
| server seg_grpc_ms | 233.8 ms |
| server post_ms (decode) | 6.1 ms |

### F — Product video loop (96 frames)

`clients/ex7_video_client.py --max-frames 96`. `infer_ms` is client POST wall, not TensorRT.

| | |
|---|---|
| frames | 96 |
| mean POST (`infer_ms`) | **147.8 ms** |
| pre_ms (JPEG encode) | 3.4 |
| post_ms (local overlay) | 11.7 |
| pipe_ms | 163.0 |
| infer frames/s | 6.76 |
| pipeline frames/s | 6.14 |
| mean server gather_ms | **125.9** |

### G — Same clip, which hop you timed

| Hop | Typical wait | Script |
|---|---|---|
| In-process pose FP16 | 5.2 ms / ~194 fps | `compare_pose.py` |
| Triton HTTP pose | 42 ms / ~24 fps | `infer_http.py` |
| FastAPI pose only | 84 ms / ~12 fps | `ex4_pose_client.py` |
| Native pose+seg gather | ~139 ms (one frame) | `infer_pose_seg_grpc.py` |
| FastAPI pose+seg, 96-frame mean | 148 ms POST, 126 ms gather / ~6.8 fps | `ex7_video_client.py` |
| FastAPI pose load | ~13 requests/s, flat from conc 1→8 | `load_pose.py` |
| Triton HTTP load | ~35 requests/s at conc 8, `count: 2` | `load_triton.py` |

The product loop is slower than in-process TensorRT. That is serving tax (JPEG, two models, JSON, Uvicorn), not a failed compile. `gather_ms` ~126 vs native ~139 is the same two gRPC calls behind FastAPI.

## Pitfalls

- NGC tag `*-vllm-python-py3` is the wrong image; use `26.08-py3`.
- Host TensorRT (e.g. 10.15 on Windows) ≠ container TensorRT 11.2.1. Rebuild in-container; do not rebuild again just to change mounts if `model.plan` already loads READY on that tag.
- One bad `config.pbtxt` takes the **whole** server down.
- FastAPI is not Triton. Clients of the product talk to `:8080`; FastAPI talks to `:8001`.
- A “video service” is a loop of `1x3x640x640` frames. Do not POST an mp4 as one infer.

## License / data

Model weights and the squat clip are not in git. Obtain Ultralytics YOLO26m `.pt` files yourself and run `optimize/` exports locally. The Triton workspace stays on the machine that has the GPU.