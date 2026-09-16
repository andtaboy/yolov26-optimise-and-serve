"""Shared letterbox, decode, overlay, and metrics for Triton pose clients."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = REPO_ROOT / "optimize" / "media" / "002_air_squat_front_smartphone_one_rep.mp4"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "run_1"
POSE_MODEL_NAME = "yolo26m_pose"
SEG_MODEL_NAME = "yolo26m_seg"
MODEL_NAME = POSE_MODEL_NAME
IMGSZ = 640
NUM_KPTS = 17
# end-to-end pose: xyxy(4) + conf(1) + cls(1) + 17*3 kpts = 57
BOX_SLICE = slice(0, 4)
CONF_IDX = 4
KPT_SLICE = slice(6, 57)

COLOURS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (255, 102, 178),
    (102, 255, 102),
]


def load_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video at {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def letterbox(image_bgr, new_shape=IMGSZ, padding_value=114, stride=32):
    """Ultralytics LetterBox: auto=False, scaleup=True, center=True, pad 114."""
    del stride  # kept in signature to match the teaching note; auto=False so unused
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    shape = image_bgr.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        image_bgr = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image_bgr = cv2.copyMakeBorder(
        image_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(padding_value,) * 3
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    batched = np.expand_dims(chw, 0)
    meta = {"ratio": r, "padw": left, "padh": top, "orig_h": shape[0], "orig_w": shape[1]}
    return batched, meta


def _scale_xy(x, y, meta):
    x = (x - meta["padw"]) / meta["ratio"]
    y = (y - meta["padh"]) / meta["ratio"]
    x = float(np.clip(x, 0, meta["orig_w"] - 1))
    y = float(np.clip(y, 0, meta["orig_h"] - 1))
    return x, y


def decode_output0(output0, meta, conf_thres=0.25):
    """Decode Triton output0 [1,300,57] to detections in original-frame pixels."""
    dets = np.asarray(output0)
    if dets.ndim == 3:
        dets = dets[0]
    people = []
    for row in dets:
        conf = float(row[CONF_IDX])
        if conf < conf_thres:
            continue
        box = row[BOX_SLICE]
        x1, y1 = _scale_xy(box[0], box[1], meta)
        x2, y2 = _scale_xy(box[2], box[3], meta)
        kpt_raw = row[KPT_SLICE].reshape(NUM_KPTS, 3)
        keypoints = []
        for kx, ky, kc in kpt_raw:
            px, py = _scale_xy(kx, ky, meta)
            keypoints.append(
                {
                    "x": px,
                    "y": py,
                    "xn": px / meta["orig_w"],
                    "yn": py / meta["orig_h"],
                    "conf": float(kc),
                }
            )
        people.append(
            {
                "box": [x1, y1, x2, y2],
                "conf": conf,
                "cls": float(row[5]),
                "keypoints": keypoints,
            }
        )
    people.sort(key=lambda p: p["conf"], reverse=True)
    return people


def decode_seg_person(output0, output1, meta, conf_thres=0.25, person_cls=0):
    """Decode seg output0 [1,300,38] + output1 [1,32,160,160] to person polygons.

    Boxes/masks are mapped back to original-frame pixels with the same letterbox meta
    as pose. COCO person is class 0. Not an 80-class dump.
    """
    dets = np.asarray(output0)
    proto = np.asarray(output1)
    if dets.ndim == 3:
        dets = dets[0]
    if proto.ndim == 4:
        proto = proto[0]
    channels, mask_h, mask_w = proto.shape
    proto_flat = proto.reshape(channels, -1)
    orig_h, orig_w = meta["orig_h"], meta["orig_w"]
    unpad_w = int(round(orig_w * meta["ratio"]))
    unpad_h = int(round(orig_h * meta["ratio"]))
    pad_x, pad_y = int(meta["padw"]), int(meta["padh"])
    people = []
    for row in dets:
        conf = float(row[CONF_IDX])
        cls = int(row[5])
        if conf < conf_thres or cls != person_cls:
            continue
        coeffs = row[6 : 6 + channels].astype(np.float32)
        mask_low = 1.0 / (1.0 + np.exp(-np.clip(coeffs @ proto_flat, -30.0, 30.0)))
        mask_low = mask_low.reshape(mask_h, mask_w)
        scale = mask_h / float(IMGSZ)
        x1 = int(np.clip(float(row[0]) * scale, 0, mask_w - 1))
        y1 = int(np.clip(float(row[1]) * scale, 0, mask_h - 1))
        x2 = int(np.clip(float(row[2]) * scale, 0, mask_w))
        y2 = int(np.clip(float(row[3]) * scale, 0, mask_h))
        if x2 <= x1 or y2 <= y1:
            continue
        cropped = np.zeros((mask_h, mask_w), dtype=np.float32)
        cropped[y1:y2, x1:x2] = mask_low[y1:y2, x1:x2]
        mask_lb = cv2.resize(cropped, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        y_end = min(pad_y + unpad_h, IMGSZ)
        x_end = min(pad_x + unpad_w, IMGSZ)
        crop = mask_lb[pad_y:y_end, pad_x:x_end]
        if crop.size == 0:
            continue
        mask_orig = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (mask_orig > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 32:
                continue
            pts = contour.reshape(-1, 2)
            if len(pts) < 3:
                continue
            polygons.append([[float(px), float(py)] for px, py in pts])
        if not polygons:
            continue
        bx1, by1 = _scale_xy(row[0], row[1], meta)
        bx2, by2 = _scale_xy(row[2], row[3], meta)
        people.append(
            {
                "cls": cls,
                "class_name": "person",
                "conf": conf,
                "box": [bx1, by1, bx2, by2],
                "polygons": polygons,
            }
        )
    people.sort(key=lambda p: p["conf"], reverse=True)
    return people


def dump_output0_layout(output0):
    dets = np.asarray(output0)
    print(f"output0 shape: {dets.shape} dtype={dets.dtype}")
    if dets.ndim == 3:
        dets = dets[0]
    if dets.size == 0:
        return
    row = dets[0]
    print(
        "first-row layout probe: "
        f"cols0-5 min/max=({row[:6].min():.4f},{row[:6].max():.4f}) "
        f"last-51 min/max=({row[6:].min():.4f},{row[6:].max():.4f}) "
        f"conf={row[CONF_IDX]:.4f}"
    )


def draw_keypoints(image, keypoints, o_h, o_w):
    for i, colour in enumerate(COLOURS):
        if i >= len(keypoints):
            break
        kpt = keypoints[i]
        if isinstance(kpt, dict):
            x, y = kpt["xn"], kpt["yn"]
        else:
            x, y = kpt[0], kpt[1]
        cx, cy = int(x * o_w), int(y * o_h)
        cv2.circle(image, (cx, cy), 5, colour, -1)
    return image


def overlay_pose_bgr(frame_bgr, people, fps=None):
    image = frame_bgr.copy()
    o_h, o_w = image.shape[:2]
    if people:
        draw_keypoints(image, people[0]["keypoints"], o_h, o_w)
        x1, y1, x2, y2 = [int(v) for v in people[0]["box"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if fps is not None:
        cv2.putText(
            image,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return image


def overlay_seg_polygons_bgr(frame_bgr, seg_people):
    image = frame_bgr.copy()
    if not seg_people:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for person in seg_people:
        for poly in person.get("polygons") or []:
            pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
    tint = np.zeros_like(image)
    tint[mask == 1] = (0, 255, 255)
    return cv2.addWeighted(image, 0.65, tint, 0.35, 0)


def overlay_infer_bgr(frame_bgr, pose_people, seg_people, fps=None):
    image = overlay_seg_polygons_bgr(frame_bgr, seg_people)
    return overlay_pose_bgr(image, pose_people, fps=fps)


def empty_totals():
    return {
        "infer_s": 0.0,
        "pre_s": 0.0,
        "post_s": 0.0,
        "pipe_s": 0.0,
        "frames": 0,
    }


def add_sample(totals, pre_s, infer_s, post_s, pipe_s):
    totals["pre_s"] += pre_s
    totals["infer_s"] += infer_s
    totals["post_s"] += post_s
    totals["pipe_s"] += pipe_s
    totals["frames"] += 1


def finalize_metrics(totals):
    n = max(totals["frames"], 1)
    infer_s = totals["infer_s"]
    pre_s = totals["pre_s"]
    post_s = totals["post_s"]
    pipe_s = totals["pipe_s"]
    frames = totals["frames"]
    return {
        "frames": frames,
        "infer_s": infer_s,
        "infer_ms": (infer_s / n) * 1000.0,
        "pre_s": pre_s,
        "pre_ms": (pre_s / n) * 1000.0,
        "post_s": post_s,
        "post_ms": (post_s / n) * 1000.0,
        "pipe_s": pipe_s,
        "pipe_ms": (pipe_s / n) * 1000.0,
        "infer_fps": frames / infer_s if infer_s > 0 else 0.0,
        "pipe_fps": frames / pipe_s if pipe_s > 0 else 0.0,
    }


def print_metrics(title, metrics):
    print(title)
    print(f"  Total frames processed: {metrics['frames']}")
    print(f"  Total inference time: {metrics['infer_s']:.2f} s")
    print(f"  Average inference time per frame: {metrics['infer_ms']:.2f} ms")
    print(f"  Total pre-processing time: {metrics['pre_s']:.2f} s")
    print(f"  Average pre-processing time per frame: {metrics['pre_ms']:.2f} ms")
    print(f"  Total post-processing time: {metrics['post_s']:.2f} s")
    print(f"  Average post-processing time per frame: {metrics['post_ms']:.2f} ms")
    print(f"  Total pipeline time: {metrics['pipe_s']:.2f} s")
    print(f"  Average pipeline time per frame: {metrics['pipe_ms']:.2f} ms")
    print(f"  Average FPS over video: {metrics['infer_fps']:.2f}")
    print(f"  Pipeline FPS over video: {metrics['pipe_fps']:.2f}")


def write_metrics_json(path, metrics, extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frames": metrics["frames"],
        "infer_ms": metrics["infer_ms"],
        "pre_ms": metrics["pre_ms"],
        "post_ms": metrics["post_ms"],
        "pipe_ms": metrics["pipe_ms"],
        "infer_fps": metrics["infer_fps"],
        "pipe_fps": metrics["pipe_fps"],
        "infer_s": metrics["infer_s"],
        "pre_s": metrics["pre_s"],
        "post_s": metrics["post_s"],
        "pipe_s": metrics["pipe_s"],
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def save_overlay(path, image_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
    print(f"Wrote overlay {path}")


def now():
    return time.perf_counter()


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _stats_host(http_url):
    host = http_url.strip()
    for prefix in ("http://", "https://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return host.rstrip("/")


def fetch_model_stats(http_url="127.0.0.1:8000", model_name=MODEL_NAME, timeout=10):
    """GET Triton's /v2/models/{name}/stats (HTTP :8000 even if you loaded over gRPC)."""
    url = f"http://{_stats_host(http_url)}/v2/models/{model_name}/stats"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _stat_pair(block, key):
    item = (block or {}).get(key) or {}
    return int(item.get("count") or 0), int(item.get("ns") or 0)


def summarize_model_stats(stats):
    models = (stats or {}).get("model_stats") or []
    if not models:
        return None
    model = models[0]
    inf = model.get("inference_stats") or {}
    out = {
        "inference_count": int(model.get("inference_count") or 0),
        "execution_count": int(model.get("execution_count") or 0),
    }
    for key in (
        "success",
        "fail",
        "queue",
        "compute_input",
        "compute_infer",
        "compute_output",
    ):
        count, ns = _stat_pair(inf, key)
        out[f"{key}_count"] = count
        out[f"{key}_ns"] = ns
        out[f"{key}_ms"] = (ns / count / 1e6) if count else 0.0
    return out


def stats_window(before, after):
    """Mean ms for events between two /v2 stats snapshots (Triton's view, not the client RTT)."""
    if not before or not after:
        return {}
    out = {}
    for key in ("queue", "compute_input", "compute_infer", "compute_output", "success"):
        dcount = after[f"{key}_count"] - before[f"{key}_count"]
        dns = after[f"{key}_ns"] - before[f"{key}_ns"]
        out[f"{key}_count_delta"] = dcount
        out[f"{key}_ms"] = (dns / dcount / 1e6) if dcount > 0 else None
    inf = after["inference_count"] - before["inference_count"]
    exe = after["execution_count"] - before["execution_count"]
    out["inference_count_delta"] = inf
    out["execution_count_delta"] = exe
    out["inferences_per_execution"] = (inf / exe) if exe else None
    return out


def snapshot_stats(http_url="127.0.0.1:8000", model_name=MODEL_NAME):
    try:
        return summarize_model_stats(fetch_model_stats(http_url, model_name=model_name))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        print(f"warning: Triton stats unavailable at {http_url}: {exc}")
        return None


def stats_fields(window):
    """Flat JSON/table fields. Missing stats → None (printed as '-')."""
    if not window:
        return {
            "queue_ms": None,
            "compute_input_ms": None,
            "compute_ms": None,
            "compute_output_ms": None,
            "inferences_per_execution": None,
            "inference_count_delta": None,
            "execution_count_delta": None,
        }
    return {
        "queue_ms": window.get("queue_ms"),
        "compute_input_ms": window.get("compute_input_ms"),
        "compute_ms": window.get("compute_infer_ms"),
        "compute_output_ms": window.get("compute_output_ms"),
        "inferences_per_execution": window.get("inferences_per_execution"),
        "inference_count_delta": window.get("inference_count_delta"),
        "execution_count_delta": window.get("execution_count_delta"),
    }
