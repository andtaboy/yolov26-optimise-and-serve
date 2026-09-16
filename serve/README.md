# serve — clients in front of Triton

This package does **not** load `.pt` or `.engine`. It sends tensors to the Triton process on WSL.

**Server (not in this repo):**

```text
Host:  /home/andre/triton          → container /triton
Scan:  /triton/model_repository
Build: /triton/build               (ONNX / trtexec scratch)
```

```bash
docker run --gpus all --rm --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/andre/triton:/triton \
  nvcr.io/nvidia/tritonserver:26.08-py3 \
  tritonserver --model-repository=/triton/model_repository
```

| Port | Protocol |
|---|---|
| 8000 | Triton HTTP |
| 8001 | Triton gRPC |
| 8002 | Prometheus metrics |
| 8080 | FastAPI product API (pose-only; never 8000) |

**Copy ONNX for a plan rebuild** (from WSL):

```bash
cp /mnt/d/SMARTCAM/python_projects/yolov26-optimise-and-serve/optimize/models/yolo26m-pose.onnx \
   /home/andre/triton/build/yolo26m-pose.onnx
```

Install clients in **WSL**, not the Windows 3.8 conda env:

```bash
cd /mnt/d/SMARTCAM/python_projects/yolov26-optimise-and-serve/serve
python3 -m pip install -r requirements.txt
```

Triton must already show `yolo26m_pose` READY. Use the same `--max-frames` everywhere if you want a short first pass.

### 4a HTTP (`:8000`)

```bash
python3 clients/infer_http.py
# optional: python3 clients/infer_http.py --max-frames 10
```

Also prints KServe `/v2/models/yolo26m_pose` and `/config`. Writes `outputs/pose_http.json` and `outputs/pose_http.jpg`.

### 4b gRPC (`:8001`)

```bash
python3 clients/infer_grpc.py
```

Writes `outputs/pose_grpc.json` and `outputs/pose_grpc.jpg`. `curl` to port 8001 should still fail (not HTTP).

In-process Ultralytics rows (Windows conda, TensorRT):

```text
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe serve\clients\compare_pose.py
```

Print the table after JSON files exist:

```bash
python3 clients/print_table.py
```

### 4c FastAPI — `api/ex4_pose_app.py` (`:8080` → Triton gRPC)

```bash
python3 api/ex4_pose_app.py
# OpenAPI: http://127.0.0.1:8080/docs
```

Second terminal:

```bash
python3 clients/ex4_pose_client.py
```

Writes `outputs/pose_fastapi.json` and `outputs/pose_fastapi.jpg`. `infer_ms` here is the full `POST /pose` round trip.

### 5 Load: product hop (5a) vs Triton hop (5b)

Exercise 4 asked “how fast is one sequential stream?” Exercise 5 asks “where does extra concurrency wait, and which config is worth changing?” **Do not** add these rows to the Exercise 4 table.

Keep FastAPI Exercise 4c (`python3 api/ex4_pose_app.py`) and Triton READY. `--instance-group` only **records** the `count` you set in `config.pbtxt`; it does not change Triton.

**5a — product API** (JPEG → FastAPI `:8080` → letterbox → gRPC → Triton):

```bash
python3 clients/load_pose.py --concurrency 1 --instance-group 1
python3 clients/load_pose.py --concurrency 8 --instance-group 1
```

**5b — native Triton** (letterbox on the client, then only `Infer()` to `:8000` / `:8001`). This is the student stand-in for NVIDIA `perf_analyzer` — same question, readable Python, same JSON as 5a:

```bash
python3 clients/load_triton.py --protocol both --concurrency 1 --instance-group 1
python3 clients/load_triton.py --protocol both --concurrency 8 --instance-group 1
# change config instance_group count to 2, restart Triton, then:
python3 clients/load_pose.py --concurrency 8 --instance-group 2
python3 clients/load_triton.py --protocol both --concurrency 8 --instance-group 2
python3 clients/print_table.py
```

Writes `outputs/run_1/pose_load_fastapi_*.json` and `pose_load_http_*.json` / `pose_load_grpc_*.json`. `print_table.py` prints **one** Exercise 5 table plus a reading guide.

**Worked example — one RTX 4090** (96 requests; `inf/exec = 1.00` on every row):

| part | hop | conc | inst | infer_ms | rps | queue_ms | compute_ms |
|---|---|---|---|---|---|---|---|
| 5a | FastAPI | 1 | 1 | 77 | **13.0** | 0.07 | 30 |
| 5a | FastAPI | 8 | 1 | 629 | **12.3** | 0.07 | 29 |
| 5a | FastAPI | 8 | 2 | 665 | **11.7** | 0.08 | 29 |
| 5b | gRPC | 1 | 1 | 62 | 16.2 | 0.08 | 27 |
| 5b | gRPC | 8 | 1 | 276 | 27.2 | 1.43 | 13 |
| 5b | gRPC | 8 | 2 | 237 | 32.1 | 0.14 | 11 |
| 5b | HTTP | 1 | 1 | 52 | 19.3 | 0.05 | 20 |
| 5b | HTTP | 8 | 1 | 237 | 33.0 | **14.9** | 13 |
| 5b | HTTP | 8 | 2 | 219 | **35.5** | 11.7 | 13 |

How to read it: **rps** is requests per second (completed calls / wall-clock time). 5a requests per second is flat at ~13 and `queue_ms` stays ~0, so wait is in FastAPI, not Triton — `count: 2` cannot help 5a. 5b HTTP reaches ~35 requests per second, so the 4090 had more to give. HTTP `queue_ms` 0.05 → 15 is Triton starting to queue; `count: 2` is a small 5b win (33 → 35.5), not 2x. Client `infer_ms` (237) is still much larger than `queue+compute` (~28) — protocol/Python tax. In-process FP16 was ~5 ms / ~194 frames per second; do not treat `compute_ms` ~13 as that number. `inf/exec = 1.00` → do not enable `dynamic_batching` on this plan.

### 6 Second model + concurrent gRPC (not FastAPI)

**6a** is Exercise 3 again for `yolo26m_seg`: ONNX in `build/`, `trtexec` save to `build/` then `cp` to `1/model.plan`, two-output `config.pbtxt` with a **comma** between outputs, restart, both READY. A parse error in seg’s config exits the **whole** server.

**6b** — one squat frame, serial `pose(); seg()` vs `asyncio.gather` on gRPC `:8001`:

```bash
python3 clients/infer_pose_seg_grpc.py
```

Writes `outputs/run_1/pose_seg_grpc.json`. If parallel wall ≈ max(pose, seg), the 4090 overlapped the two models. If ≈ serial, it did not — still a valid measurement. Dual-model FastAPI is Exercise 7.

### 7 Capstone — `api/ex7_infer_app.py` (stop `ex4_pose_app.py` first; both bind `:8080`)

Product API, not Triton’s ports. `POST /infer` letterboxes once, `asyncio.gather`s pose + seg gRPC (the 6b calls), returns keypoints + **person** polygons. Overlay stays on the client. Video = client POSTs frames, not an mp4 blob.

```bash
python3 api/ex7_infer_app.py
# other terminal:
python3 clients/ex7_infer_client.py
python3 clients/ex7_video_client.py --max-frames 96
```

| Path | Role |
|---|---|
| `api/ex4_pose_app.py` | **4c** pose-only `POST /pose` |
| `clients/ex4_pose_client.py` | **4c** frame client |
| `api/ex7_infer_app.py` | **7** `GET /health` (both models) + `POST /infer` |
| `clients/ex7_infer_client.py` | **7b** one squat frame |
| `clients/ex7_video_client.py` | **7c** frame loop (`--max-frames`) |

Do not return a rendered video. Do not bind FastAPI on `:8000`.

| Path | Exercise |
|---|---|
| `clients/common.py` | letterbox / decode / metrics / Triton `/v2` stats |
| `clients/infer_http.py` | 4a HTTP `:8000` |
| `clients/infer_grpc.py` | 4b gRPC `:8001` |
| `clients/compare_pose.py` | 4b in-process `.pt` / FP16 JSON |
| `clients/print_table.py` | Exercise 4 table + Exercise 5 5a/5b table |
| `api/ex4_pose_app.py` | **4c** FastAPI pose-only `:8080` |
| `clients/ex4_pose_client.py` | **4c** frame client |
| `clients/load_pose.py` | **5a** concurrent POST `/pose` (needs ex4 app) |
| `clients/load_triton.py` | **5b** concurrent Triton HTTP/gRPC |
| `clients/infer_pose_seg_grpc.py` | **6b** serial vs `asyncio.gather` pose+seg gRPC |
| `api/ex7_infer_app.py` | **7** FastAPI pose+seg `POST /infer` |
| `clients/ex7_infer_client.py` | **7b** one-frame client |
| `clients/ex7_video_client.py` | **7c** frame POST loop |
