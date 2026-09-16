# YOLO26m TensorRT evaluation (`optimize/`)

This folder is **Ultralytics export + COCO val only**. Triton serving lives outside this tree (WSL `/home/andre/triton` and repo `serve/`).

These are the working notes for the data scientist taking this over. The job was not “export some engines.” It was: **can we serve INT8 TensorRT copies of YOLO26m pose and YOLO26m seg instead of the PyTorch `.pt` files, claiming roughly 2× speed for a slight accuracy drop?**

Short answer, from measured val on this machine (RTX 4090 Laptop, Ultralytics 8.4.12, Python 3.8.20, torch 2.4.1+cu121):

- **Seg INT8: yes, with a modest drop.** Mask mAP50-95 fell 0.4252 → 0.3963 (−0.0289, about 7% relative). Inference went 11.7 ms → 3.3 ms (~3.5×). That is the “slight drop, more than double the speed” story.
- **Pose INT8: no.** Pose mAP50-95 fell 0.6953 → 0.4080 on INT8-only and 0.4104 with INT8+FP16 mixed. The val path is honest: a plain FP16 engine matches the `.pt` (pose mAP50-95 **0.6953**, delta 0.0000) at 10.0 ms → 3.0 ms (~3.3×). Serve **pose FP16**, not INT8.

The rest of this file is how we got those numbers, which files to run, what they actually do, and the dataset traps that already wasted a full val cycle.

---

## 1. What this project is for

We compare **the same architecture, same input size (640), same batch (1)** in two runtimes:

| artifact | meaning |
|---|---|
| `models/yolo26m-pose.pt` / `models/yolo26m-seg.pt` | official Ultralytics PyTorch weights (FP32 graph, GPU) |
| `models/yolo26m-pose-int8-fp16.engine` | pose TensorRT mixed INT8+FP16 (not servable; large OKS drop) |
| `models/yolo26m-pose-fp16.engine` | **serve pose** — TensorRT FP16; matches `.pt` mAP |
| `models/yolo26m-seg-int8-fp16.engine` | **serve seg** — TensorRT mixed INT8+FP16; modest mask drop |

Ultralytics always writes `models/yolo26m-pose.engine` / `yolo26m-seg.engine` from the `.pt` stem. Export scripts **rename** immediately to the names above. Do not leave an unnamed `.engine` in `models/`.

Accuracy is **not** FPS on a squat video. Accuracy is Ultralytics `model.val()` on COCO val2017 with the **correct label format** for each task:

- Pose → box mAP + **keypoint mAP (OKS)**
- Seg → box mAP + **mask mAP**

Speed numbers quoted below are Ultralytics val `Speed:` lines (preprocess / inference / postprocess per image). That is model-only timing, not a full product pipeline.

Python to use (do not invent a new venv):

```text
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe
```

Keep the dataset drive mounted as **D:**. INT8 export and val both fail if the disk disappears mid-run.

---

## 2. Current pipeline (run in this order)

```text
1. export_pose.py          → models/yolo26m-pose-int8-fp16.engine
2. export_pose.py --fp16   → models/yolo26m-pose-fp16.engine
3. export_seg.py           → models/yolo26m-seg-int8-fp16.engine
4. val_compare.py          → PT vs named engines
5. video_compare.py           → optional video FPS (defaults: pose FP16, seg INT8+FP16)
```

Engines already exist on disk (built 13 Sep 2026). Re-run 1–2 only if you change calib images, TensorRT, Ultralytics, or GPU. Re-run 3 after any new engine.

### 2.1 `export_pose.py`

**Intention:** produce `models/yolo26m-pose-int8-fp16.engine` (mixed) and, with `--fp16`, `models/yolo26m-pose-fp16.engine`.

What it does:

1. Ensures `datasets/coco/images/val2017` (COCO val JPEGs) and `datasets/coco/labels/val2017` (pose labels: class + xywh + 17×3 keypoints).
2. Samples **300 labeled** pose images (not random unlabeled JPEGs). Copies JPEG + matching `.txt` into `datasets/coco/images/calib` and `datasets/coco/labels/calib`.
3. Writes `coco-calib-pose.yaml` (`train`/`val` both `images/calib`, `kpt_shape: [17, 3]`, one class `person`).
4. `YOLO(...).export(format="engine", int8=True, ...)`. Default also applies the TensorRT INT8+FP16 builder-flag patch, then **renames** the Ultralytics output to `models/yolo26m-pose-int8-fp16.engine`. `python export_pose.py --fp16` builds plain FP16 (`half=True`, no INT8) as `models/yolo26m-pose-fp16.engine` and does not overwrite the mixed file.

**Outcomes we already have:**

- First export calibrated on 300 images with **0 labels** (Ultralytics reported 300 backgrounds). Engine was valid but scales were not person-conditioned.
- After the copy-labels fix, scan was **300 images, 0 backgrounds**. Labeled INT8-only engine (~22.8 MB, 13 Sep 21:19) lived at the Ultralytics default name `yolo26m-pose.engine`. **That file is gone** (overwritten by mixed, then renamed).
- 14 Sep: same 300-image calib, INT8+FP16 mixed builder flags. Now `models/yolo26m-pose-int8-fp16.engine`. **Pose mAP did not recover**.
- Plain FP16 (`export_pose.py --fp16`) is `models/yolo26m-pose-fp16.engine` (~44.9 MB). Smoke load returned keypoints. Full val matches `.pt`.

TensorRT INT8 here is **MinMax calibration**. It does **not** train and does **not** read boxes or keypoints. The calibrator only needs images (`batch["img"] / 255`). We still copy labels so the dataloader does not treat every frame as background and so the 300-frame set is actually the pose distribution (people), not random COCO clutter.

`name=yolo26m_pose_int8` is ignored. Ultralytics writes `yolo26m-pose.engine`; we rename to `yolo26m-pose-int8-fp16.engine` or `yolo26m-pose-fp16.engine`.

Some decode/NMS ops (`TopK`, `Tile`, …) stay non-INT8. That is normal.

### 2.2 `export_seg.py`

**Intention:** produce `models/yolo26m-seg-int8-fp16.engine` without poisoning pose calib, and without using pose labels as polygons.

What it does:

1. Reuses `datasets/coco/images/val2017` for JPEGs. **Does not copy pose labels.**
2. Puts polygon labels under `datasets/coco-seg/labels/val2017` (from `coco2017labels-segments.zip`, val2017 members only — it does not unpack 118k train files).
3. Builds its **own** 300-image calib under `datasets/coco-seg/images/calib` + `datasets/coco-seg/labels/calib`.
4. Writes `coco-calib-seg.yaml` with all **80** COCO names.
5. Exports INT8, but patches TensorRT so **FP16 is also allowed**. YOLO-seg’s proto/cv3 `Conv+SiLU` has no INT8 kernel on this GPU; INT8-only build dies with TensorRT Error Code 10. The patch sets `BuilderFlag.FP16` after INT8 and disables JIT convolutions.

**Outcome:** `models/yolo26m-seg-int8-fp16.engine` (~26.4 MB, 13 Sep 22:15), cache `models/yolo26m-seg.cache`. You should see `300 images, 0 backgrounds` during export. Do not reuse `datasets/coco/labels/calib` (those are pose keypoints).

### 2.3 `val_compare.py` — the accuracy source of truth

**Intention:** same val set, imgsz, batch, device for `.pt` vs `.engine`. Print a small delta table. Do not trust FPS scripts for accuracy.

```text
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe val_compare.py
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe val_compare.py --skip-pose
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe val_compare.py --skip-seg
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe val_compare.py --skip-seg --pose-engine models/yolo26m-pose-fp16.engine
```

Flags: `--imgsz 640` (default), `--batch 1`, `--device 0`, `--workers 2`, `--fraction` (Ultralytics val fraction; **do not assume 0.05 actually subsamples** — we already saw a “quick” run still walk the full split).

What it does before each task:

1. **Pose YAML** (`coco-val-pose.yaml`): list file `datasets/coco/val2017-pose.txt` of **only images that have a pose `.txt`**. Paths must start with `./` (`./images/val2017/<name>`). YAML `path:` is the **absolute** `datasets/coco`. Includes `kpt_shape` and `flip_idx`.
2. **Seg YAML** (`coco-val-seg.yaml`): list file `datasets/coco-seg/val2017-seg.txt`. YAML `path:` is absolute `datasets/coco-seg`. Class names copied from `coco-calib-seg.yaml`. JPEGs are looked up in the **real** `datasets/coco/images/val2017`, not via `Path.resolve()` on the Windows junction.
3. Deletes stale `*.cache` files so a previous wrong scan cannot poison the next val.
4. Creates a Windows **junction** `datasets/coco-seg/images/val2017` → `datasets/coco/images/val2017` so JPEGs are not duplicated. The junction is for image **access**. The YAML `val:` must **not** be that directory.

Then `YOLO(path).val(...)` twice per task (`int8=False` for `.pt`, `int8=True` for `.engine`). Results land under `runs/val-compare/<stem>/`.

**A healthy pose scan looks like:** `2346 images, 0 backgrounds, 0 corrupt`.  
**A healthy seg scan looks like:** `4952 images, …, 0 corrupt`, tens of thousands of instances.

If you see thousands of “corrupt” labels or `cannot reshape array of size 55 into shape (2)`, you are feeding **pose keypoints** to the **seg** parser. Stop. Do not interpret mAP=0 as a bad engine.

### 2.4 `video_compare.py` — speed only

Times pose and seg separately on `media/002_air_squat_front_smartphone_one_rep.mp4`: `.pt` vs `.engine`, with warmup and CUDA sync. Prints inference / pre / post / pipeline ms and a speedup. **No mAP.** Use this when you care about video FPS, not when you decide whether INT8 is accurate enough.

---

## 3. Dataset layout (read this twice)

COCO val JPEGs are **shared**. Labels are **not**.

```text
datasets/coco/images/val2017/          5000 JPEGs (canonical images)
datasets/coco/labels/val2017/          2346 pose files  (class + box + 17 kpts)
datasets/coco/images/calib/            300 pose calib JPEGs
datasets/coco/labels/calib/            300 matching pose labels

datasets/coco-seg/labels/val2017/      4952 polygon files (80 classes)
datasets/coco-seg/images/calib/        300 seg calib JPEGs (copies)
datasets/coco-seg/labels/calib/        300 matching polygon labels
datasets/coco-seg/images/val2017/      Windows junction → coco/images/val2017
```

Pose line after the class id: 4 box numbers + 51 keypoint numbers = **55 numbers** (odd).  
Seg line after the class id: polygon `x,y` pairs (even length).

If the seg loader is pointed at pose labels, every file is “corrupt” (`reshape … 55 into shape (2)`), instance count is 0, and both `.pt` and `.engine` report mAP 0. That already happened. It was **not** a broken seg model.

### The junction trap (the actual bug)

Ultralytics `img2label_paths()` swaps `\images\` → `\labels\` on the **resolved** image path.

Windows `Path.resolve()` **follows** `datasets/coco-seg/images/val2017` into `datasets/coco/images/val2017`. Labels then become `datasets/coco/labels/val2017` — pose keypoints. The cache can even overwrite `datasets/coco/labels/val2017.cache`.

**Rule:** keep the junction so we do not duplicate 5000 JPEGs. Write YAML `val:` as a **list file** under `datasets/coco-seg` (`val2017-seg.txt` with `./images/val2017/...`). Never set `val: images/val2017` for coco-seg on this Windows layout.

**Never `rmtree` the junction.** That can delete the real coco images. If you must remove it, `rmdir` (junction only).

### List-file `./` prefix

Without `./`, Ultralytics treats `images/val2017/....jpg` as relative to the **cwd**, not the YAML `path:`. Result: 2346 “corrupt” pose files and a crash. Prefix is mandatory.

---

## 4. Measured results (use these, not folklore)

Hardware: NVIDIA GeForce RTX 4090 Laptop GPU, 16376 MiB. `imgsz=640`, `batch=1`.

### Pose — 2346 labeled val images, 6352 instances, 0 corrupt

Same `.pt` and val split for every engine row. INT8-only = labeled 300-image MinMax, TensorRT `INT8` flag only (13 Sep; **file not on disk**). INT8+FP16 mixed = `yolo26m-pose-int8-fp16.engine`. FP16 = `yolo26m-pose-fp16.engine`.

| model | box mAP50-95 | box mAP50 | pose mAP50-95 | pose mAP50 | inf ms |
|---|---:|---:|---:|---:|---:|
| `yolo26m-pose.pt` | 0.7476 | 0.9308 | 0.6953 | 0.8896 | 10.0 |
| `yolo26m-pose.engine` INT8-only (deleted) | 0.6320 | 0.8604 | 0.4080 | 0.7532 | 2.4 |
| delta (INT8-only − pt) | −0.1157 | −0.0704 | **−0.2872** | −0.1364 | ~4.2× |
| `yolo26m-pose-int8-fp16.engine` | 0.6324 | 0.8583 | 0.4104 | 0.7578 | 2.3 |
| delta (INT8+FP16 − pt) | −0.1152 | −0.0725 | **−0.2848** | −0.1318 | ~4.3× |
| `yolo26m-pose-fp16.engine` | 0.7476 | 0.9309 | 0.6953 | 0.8896 | 3.0 |
| delta (FP16 − pt) | +0.0000 | +0.0001 | **+0.0000** | +0.0000 | ~3.3× |

Pose mAP50-95 is the metric that matters for joints. INT8 losing ~29 points is real quantization: the same val split and `model.val()` path give **identical** 0.6953 for `.pt` and FP16. Mixed INT8+FP16 did not move it (0.4080 → 0.4104). Pose mAP50 (0.89 → ~0.76 on INT8) is milder because it only asks for coarse OKS≥0.50.

`.pt` inference on this GPU has been 7.9–10.2 ms across repeats; use **10.0 ms** from the FP16 compare run. INT8 ~2.3 ms is faster than FP16 ~3.0 ms, at the cost of the OKS collapse.

### Seg — 4952 labeled val images, 36335 instances, 0 corrupt

(After the list-file / junction fix. Ignore any earlier table of all zeros.)

| model | box mAP50-95 | box mAP50 | mask mAP50-95 | mask mAP50 | inf ms |
|---|---:|---:|---:|---:|---:|
| `yolo26m-seg.pt` | 0.5220 | 0.6891 | 0.4252 | 0.6568 | 11.7 |
| `yolo26m-seg-int8-fp16.engine` | 0.4768 | 0.6539 | 0.3963 | 0.6221 | 3.3 |
| delta (engine − pt) | −0.0452 | −0.0352 | **−0.0289** | −0.0348 | ~3.5× |

Seg inference times are from val `Speed:` lines on this GPU. The all-zero run still executed the models, so those ms are usable; the mAP row is from the fixed 4952-image val.

---

## 5. Serve decision

**Seg INT8+FP16 — serve.** Artifact: `models/yolo26m-seg-int8-fp16.engine`. Accuracy drop is in the range people mean by “slight.” Speed more than doubles. Document the ~3 mask-mAP50-95 points. Watch person-class mask quality in-app; COCO 80-class mAP can hide a person-only regression, so spot-check SMARTCAM footage.

**Pose FP16 — serve.** Artifact: `models/yolo26m-pose-fp16.engine`. Matches the `.pt` on COCO pose val (pose mAP50-95 0.6953, box mAP50-95 0.7476). Inference 10.0 ms → 3.0 ms.

**Pose INT8 / INT8+FP16 — do not serve.** `models/yolo26m-pose-int8-fp16.engine` is on disk for the record; pose mAP50-95 stayed at 0.41. Pure INT8-only was overwritten. INT8 is faster (~2.3 ms) only because it destroys OKS. If INT8 is mandatory: QAT in a **new** env. If the product only needs a person box and crude keypoints, pose mAP50 ~0.76 might be acceptable — that is a product call, not an ML “slight drop.”

Do not quote “double the speed, slight drop” for **both INT8 models**. It is true for `yolo26m-seg-int8-fp16.engine`. Pose should be `yolo26m-pose-fp16.engine`.

---

## 6. How to re-run from scratch (if engines are lost)

```text
cd <this-repo>\optimize

C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe export_pose.py
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe export_pose.py --fp16
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe export_seg.py
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe val_compare.py
C:\Users\andta\miniconda3\envs\model_perfromance_eval\python.exe val_compare.py --skip-seg --pose-engine models/yolo26m-pose-fp16.engine
```

Export success looks like `TensorRT: export success` and exit code 0. Calibration progress is not the engine; wait for the file timestamp on `models/*.engine`.

If you change the 300 calib images and want new INT8 scales, delete **that model’s** `models/<stem>.cache` (e.g. `yolo26m-pose.cache`). Do not delete `datasets/coco/labels/calib.cache` unless you rebuilt the pose calib folder — that file is a dataset index, not TensorRT scales. Pose and seg caches are separate; they do not cross-read.

---

## 7. Files that matter vs leftover experiments

### Live (keep)

| file | role |
|---|---|
| `export_pose.py` | download pose data, 300 labeled calib, INT8 or `--fp16` engine |
| `export_seg.py` | download seg labels, 300 labeled calib, INT8+FP16 engine |
| `val_compare.py` | PT vs named engines (`--pose-engine` for FP16) |
| `video_compare.py` | video FPS; defaults pose FP16 + seg INT8+FP16 |
| `coco-calib-pose.yaml` / `coco-calib-seg.yaml` | generated by export; required for INT8 calib |
| `coco-val-pose.yaml` / `coco-val-seg.yaml` | generated by val compare; do not hand-edit `val:` onto the junction |
| `requirements.txt` | `ultralytics` + `tensorrt>7.0.0,!=10.1.0` |
| `models/yolo26m-pose-fp16.engine` | **servable pose** |
| `models/yolo26m-seg-int8-fp16.engine` | **servable seg** |
| `models/yolo26m-pose-int8-fp16.engine` | pose mixed INT8+FP16 eval artifact, not for serve |

### Removed (March yolo26s demo)

Deleted from this repo: `yolo-rt.py`, `yolo-rt-optimized.py`, `yolo-rt-benchmark.py`, `yolo-webcam.py`, all `models/yolo26s-*`, `outputs/`, and leaked Triton `*.plan` files that belonged on WSL (`/home/andre/triton/build`), not here.

---

## 8. Operational gotchas

- Python 3.8 `urllib` does not follow HTTP 308. Pose/seg zips must go through Ultralytics `safe_download`, which the export scripts already use.
- `--fraction` on val was not a reliable subsample in our run. Budget time for full 2346 / 4952.
- Val overwrites `runs/val-compare/<stem>/`. Pose and seg use different stems, so they do not clobber each other; repeating the same model does.
- Serving the `.engine` on a **different GPU / TensorRT version** usually requires a rebuild. Engines are not portable the way `.pt` files are.
- Product images are gym/phone video, not COCO. COCO mAP is the gate we used; a small person-only set with OKS / IoU on SMARTCAM frames is the next check before production, especially for pose.
