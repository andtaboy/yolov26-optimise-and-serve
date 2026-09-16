"""Exercise 5a — concurrent FastAPI POST /pose (product hop under load)."""

from __future__ import annotations

import argparse
import asyncio
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
    MODEL_NAME,
    load_frames,
    now,
    percentile,
    print_metrics,
    snapshot_stats,
    stats_fields,
    stats_window,
    write_metrics_json,
)


def encode_jpegs(frames):
    blobs = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        blobs.append(buf.tobytes())
    return blobs


async def run_load(url, blobs, concurrency, requests, warmup, stats_url):
    pose_url = url.rstrip("/") + "/pose"
    health_url = url.rstrip("/") + "/health"
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await client.get(health_url)
        health.raise_for_status()
        print("GET /health", health.json())

        for i in range(min(warmup, len(blobs))):
            resp = await client.post(
                pose_url,
                files={"file": ("frame.jpg", blobs[i], "image/jpeg")},
            )
            resp.raise_for_status()

        before = snapshot_stats(stats_url)
        next_i = 0
        lock = asyncio.Lock()
        latencies = []
        errors = 0
        pre_s = 0.0
        post_s = 0.0

        async def worker():
            nonlocal next_i, errors, pre_s, post_s
            while True:
                async with lock:
                    if next_i >= requests:
                        return
                    idx = next_i
                    next_i += 1
                jpeg = blobs[idx % len(blobs)]
                pre0 = now()
                files = {"file": ("frame.jpg", jpeg, "image/jpeg")}
                pre1 = now()
                inf0 = now()
                try:
                    resp = await client.post(pose_url, files=files)
                    resp.raise_for_status()
                    inf1 = now()
                    post0 = now()
                    _ = resp.json()
                    post1 = now()
                except Exception as exc:
                    errors += 1
                    print(f"request {idx} failed: {exc}")
                    continue
                latencies.append(inf1 - inf0)
                pre_s += pre1 - pre0
                post_s += post1 - post0

        wall0 = now()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = now() - wall0

    window = stats_window(before, snapshot_stats(stats_url))
    completed = len(latencies)
    infer_s = sum(latencies)
    n = max(completed, 1)
    ms = [x * 1000.0 for x in latencies]
    ms_sorted = sorted(ms)
    throughput = completed / wall if wall > 0 else 0.0
    metrics = {
        "frames": completed,
        "infer_s": infer_s,
        "infer_ms": (infer_s / n) * 1000.0,
        "pre_s": pre_s,
        "pre_ms": (pre_s / n) * 1000.0,
        "post_s": post_s,
        "post_ms": (post_s / n) * 1000.0,
        "pipe_s": wall,
        "pipe_ms": (wall / n) * 1000.0,
        # Per-request mean FPS would be 1000/infer_ms; do NOT use aggregate here.
        "infer_fps": (n / infer_s) if infer_s > 0 else 0.0,
        "pipe_fps": throughput,
        "concurrency": concurrency,
        "errors": errors,
        "p50_ms": percentile(ms_sorted, 50),
        "p99_ms": percentile(ms_sorted, 99),
        "throughput_rps": throughput,
        "wall_s": wall,
        **stats_fields(window),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 5a: N concurrent POST /pose (FastAPI product hop). Not Triton-native load."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--stats-url", default="127.0.0.1:8000", help="Triton HTTP for /v2 stats")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--requests", type=int, default=None, help="Total POSTs (default: max-frames)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--instance-group",
        type=int,
        default=None,
        help="Record Triton's instance_group count in JSON (you set this in config.pbtxt).",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    frames = load_frames(args.video, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No frames read from video.")
    blobs = encode_jpegs(frames)
    n_req = args.requests if args.requests is not None else len(frames)

    metrics = asyncio.run(
        run_load(args.url, blobs, args.concurrency, n_req, args.warmup, args.stats_url)
    )

    ig = args.instance_group
    tag = f"c{args.concurrency}"
    if ig is not None:
        tag += f"_ig{ig}"
    json_out = args.json_out or (DEFAULT_OUTPUT_DIR / f"pose_load_fastapi_{tag}.json")

    print_metrics(
        f"FastAPI load /pose  concurrency={args.concurrency}  instance_group={ig!s}:",
        metrics,
    )
    print(f"  Completed: {metrics['frames']}  errors: {metrics['errors']}")
    print(f"  Latency p50: {metrics['p50_ms']:.2f} ms  p99: {metrics['p99_ms']:.2f} ms")
    print(f"  Aggregate throughput: {metrics['throughput_rps']:.2f} req/s")
    q = metrics.get("queue_ms")
    c = metrics.get("compute_ms")
    print(
        "  Triton stats window: "
        f"queue_ms={q if q is not None else '-'}  "
        f"compute_ms={c if c is not None else '-'}"
    )
    print("  infer_ms is mean POST /pose RTT (JPEG + FastAPI + gRPC + JSON), not TensorRT alone.")

    extra = {
        "path": f"fastapi_load_{tag}",
        "protocol": "fastapi_http_load",
        "exercise_part": "5a",
        "warmup": args.warmup,
        "video": str(args.video),
        "model": MODEL_NAME,
        "concurrency": args.concurrency,
        "errors": metrics["errors"],
        "p50_ms": metrics["p50_ms"],
        "p99_ms": metrics["p99_ms"],
        "throughput_rps": metrics["throughput_rps"],
        "wall_s": metrics["wall_s"],
        "instance_group": ig,
        "queue_ms": metrics.get("queue_ms"),
        "compute_input_ms": metrics.get("compute_input_ms"),
        "compute_ms": metrics.get("compute_ms"),
        "compute_output_ms": metrics.get("compute_output_ms"),
        "inferences_per_execution": metrics.get("inferences_per_execution"),
        "inference_count_delta": metrics.get("inference_count_delta"),
        "execution_count_delta": metrics.get("execution_count_delta"),
    }
    write_metrics_json(json_out, metrics, extra=extra)


if __name__ == "__main__":
    main()
