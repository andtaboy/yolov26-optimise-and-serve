"""Exercise 5b — concurrent load on native Triton HTTP/gRPC (Python stand-in for perf_analyzer).

This is not NVIDIA's C++ perf_analyzer binary. Students can read every hop:
letterbox on the client, then only time Infer() to :8000 or :8001, then read
Triton's /v2/models/.../stats for queue vs compute_infer.

Part 5a (load_pose.py) times FastAPI POST /pose. Part 5b times Triton itself.
Same JSON keys so print_table.py puts both on one table.
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO,
    MODEL_NAME,
    letterbox,
    load_frames,
    now,
    percentile,
    print_metrics,
    snapshot_stats,
    stats_fields,
    stats_window,
    write_metrics_json,
)


def infer_http(client, tensor):
    inp = httpclient.InferInput("images", list(tensor.shape), "FP32")
    inp.set_data_from_numpy(tensor)
    out = httpclient.InferRequestedOutput("output0")
    result = client.infer(model_name=MODEL_NAME, inputs=[inp], outputs=[out])
    return result.as_numpy("output0")


def infer_grpc(client, tensor):
    inp = grpcclient.InferInput("images", list(tensor.shape), "FP32")
    inp.set_data_from_numpy(tensor)
    out = grpcclient.InferRequestedOutput("output0")
    result = client.infer(model_name=MODEL_NAME, inputs=[inp], outputs=[out])
    return result.as_numpy("output0")


def make_client(protocol, url):
    if protocol == "http":
        client = httpclient.InferenceServerClient(url=url)
        if not client.is_server_ready():
            raise RuntimeError(f"Triton HTTP not ready at {url}")
        return client
    client = grpcclient.InferenceServerClient(url=url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton gRPC not ready at {url}")
    return client


def infer_one(protocol, client, tensor):
    if protocol == "http":
        return infer_http(client, tensor)
    return infer_grpc(client, tensor)


def default_url(protocol):
    return "127.0.0.1:8000" if protocol == "http" else "127.0.0.1:8001"


def run_load(protocol, url, tensors, concurrency, requests, warmup, stats_url):
    probe = make_client(protocol, url)
    for i in range(min(warmup, len(tensors))):
        infer_one(protocol, probe, tensors[i])

    before = snapshot_stats(stats_url)
    next_i = 0
    lock = threading.Lock()
    latencies = []
    errors = 0

    def worker():
        nonlocal next_i, errors
        client = make_client(protocol, url)
        while True:
            with lock:
                if next_i >= requests:
                    return
                idx = next_i
                next_i += 1
            tensor = tensors[idx % len(tensors)]
            t0 = now()
            try:
                _ = infer_one(protocol, client, tensor)
                t1 = now()
            except Exception as exc:
                with lock:
                    errors += 1
                print(f"request {idx} failed: {exc}")
                continue
            with lock:
                latencies.append(t1 - t0)

    wall0 = now()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker) for _ in range(concurrency)]
        for fut in futs:
            fut.result()
    wall = now() - wall0
    after = snapshot_stats(stats_url)
    window = stats_window(before, after)

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
        "pre_s": 0.0,
        "pre_ms": 0.0,
        "post_s": 0.0,
        "post_ms": 0.0,
        "pipe_s": wall,
        "pipe_ms": (wall / n) * 1000.0,
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


def write_one(protocol, url, tensors, pre_ms, args, n_req):
    metrics = run_load(
        protocol,
        url,
        tensors,
        args.concurrency,
        n_req,
        args.warmup,
        args.stats_url,
    )
    metrics["pre_ms"] = pre_ms
    metrics["pre_s"] = (pre_ms / 1000.0) * max(metrics["frames"], 1)

    ig = args.instance_group
    tag = f"{protocol}_c{args.concurrency}"
    if ig is not None:
        tag += f"_ig{ig}"
    json_out = args.json_out if args.json_out and protocol != "both" else (
        DEFAULT_OUTPUT_DIR / f"pose_load_{tag}.json"
    )

    print_metrics(
        f"Triton {protocol} load  concurrency={args.concurrency}  instance_group={ig!s}:",
        metrics,
    )
    print(f"  Completed: {metrics['frames']}  errors: {metrics['errors']}")
    print(f"  Latency p50: {metrics['p50_ms']:.2f} ms  p99: {metrics['p99_ms']:.2f} ms")
    print(f"  Aggregate throughput: {metrics['throughput_rps']:.2f} req/s")
    q = metrics.get("queue_ms")
    c = metrics.get("compute_ms")
    batch = metrics.get("inferences_per_execution")
    print(
        "  Triton stats window: "
        f"queue_ms={q if q is not None else '-'}  "
        f"compute_ms={c if c is not None else '-'}  "
        f"inferences/execution={batch if batch is not None else '-'}"
    )
    print("  infer_ms is client Infer() RTT (tensor already letterboxed; no JPEG, no FastAPI).")

    extra = {
        "path": f"triton_{protocol}_load_c{args.concurrency}"
        + (f"_ig{ig}" if ig is not None else ""),
        "protocol": f"triton_{protocol}_load",
        "exercise_part": "5b",
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
        "url": url,
        "queue_ms": metrics.get("queue_ms"),
        "compute_input_ms": metrics.get("compute_input_ms"),
        "compute_ms": metrics.get("compute_ms"),
        "compute_output_ms": metrics.get("compute_output_ms"),
        "inferences_per_execution": metrics.get("inferences_per_execution"),
        "inference_count_delta": metrics.get("inference_count_delta"),
        "execution_count_delta": metrics.get("execution_count_delta"),
    }
    write_metrics_json(json_out, metrics, extra=extra)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Exercise 5b: N concurrent Triton Infer() calls (HTTP or gRPC). "
            "Not FastAPI. Not NVIDIA perf_analyzer. Same JSON as load_pose.py."
        )
    )
    parser.add_argument("--protocol", choices=("http", "grpc", "both"), default="http")
    parser.add_argument("--url", default=None, help="Override (default :8000 HTTP / :8001 gRPC)")
    parser.add_argument("--stats-url", default="127.0.0.1:8000", help="Triton HTTP for /v2 stats")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--requests", type=int, default=None, help="Total infers (default: max-frames)")
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
    if args.json_out is not None and args.protocol == "both":
        raise SystemExit("--json-out cannot be used with --protocol both")

    frames = load_frames(args.video, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No frames read from video.")

    tensors = []
    pre0 = now()
    for frame in frames:
        tensor, _meta = letterbox(frame)
        tensors.append(tensor)
    pre_ms = ((now() - pre0) / max(len(frames), 1)) * 1000.0
    print(f"letterbox (not in infer_ms): {pre_ms:.2f} ms/frame, n={len(tensors)}")

    n_req = args.requests if args.requests is not None else len(frames)
    protocols = ("http", "grpc") if args.protocol == "both" else (args.protocol,)
    for protocol in protocols:
        url = args.url or default_url(protocol)
        write_one(protocol, url, tensors, pre_ms, args, n_req)


if __name__ == "__main__":
    main()
