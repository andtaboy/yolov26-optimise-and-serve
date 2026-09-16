"""Print Exercise 4 (one stream) and Exercise 5 (5a FastAPI + 5b Triton) load tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import DEFAULT_OUTPUT_DIR  # noqa: E402

ROWS = [
    ("pose_pt.json", "YOLO .pt in-process"),
    ("pose_engine.json", "YOLO FP16 engine in-process"),
    ("pose_http.json", "Triton HTTP :8000"),
    ("pose_grpc.json", "Triton gRPC :8001"),
    ("pose_fastapi.json", "FastAPI :8080 -> Triton gRPC"),
]

READ_GUIDE = """
How to read the Exercise 5 table
--------------------------------
Exercise 4 (above) is ONE sequential client. Its infer_fps is that stream's
frames per second. Exercise 5 is MANY in-flight requests. rps is requests per
second (completed calls / wall-clock time) -- aggregate capacity.
Do not put a 5th/6th row on the Exercise 4 table and call it "faster Triton".

Each row is one hop x concurrency x instance_group.count (you set the count
in config.pbtxt; the script only records it):

  5a  FastAPI POST /pose   JPEG in, JSON out; FastAPI letterbox + gRPC to Triton
  5b  native Triton HTTP :8000 or gRPC :8001
      NCHW tensor already letterboxed on the client; no JPEG, no FastAPI

Column headers (short name -> full name):
  part         exercise part (5a product hop vs 5b native Triton hop)
  path         which client path (FastAPI, Triton HTTP, or Triton gRPC)
  conc         concurrency (in-flight requests from this script)
  inst         instance group count (instance_group.count in config.pbtxt)
  ok           successful requests
  err          failed requests
  infer_ms     mean inference latency in milliseconds (client wait for THAT hop)
  p50_ms       50th percentile latency in milliseconds (half of requests faster)
  p99_ms       99th percentile latency in milliseconds (tail; >> p50 usually means queueing)
  rps          requests per second (completed requests / wall-clock time) -- capacity
  queue_ms     queue time in milliseconds (Triton /v2 stats: wait inside Triton)
  compute_ms   compute time in milliseconds (Triton /v2 stats: GPU compute_infer)
  inf/exec     inferences per execution (1.00 = one execution per request, no dynamic batching)

Decision shortcuts (what to try next):
  requests/sec flat, infer_ms ~= concurrency x (c=1 latency)
                                 -> saturated; extra clients only wait
  queue_ms grows, compute_ms flat
                                 -> wait for an instance, kernels not slower
  5a infer_ms >> 5b infer_ms     -> product/JPEG/Python tax, not TensorRT
  5b infer_ms >> compute_ms      -> client / network / protocol tax
  5b HTTP vs 5b gRPC             -> protocol tax at Triton's boundary
  instance group 2 does not raise requests/sec
                                 -> second instance unused, or GPU already full
  inf/exec stays 1.00            -> expected here (batch-1 plan, max_batch_size: 0)
  Do not turn on dynamic_batching without rebuilding ONNX with a batch axis.

Compare 5b concurrency=1 infer_ms to Exercise 4 HTTP/gRPC infer_ms -- same hop, should be close.
Compare 5a concurrency=1 infer_ms to Exercise 4 FastAPI infer_ms -- same product hop.
""".strip()


def _print_grid(title, header, body_rows):
    if not body_rows:
        print(title)
        print("  (no rows)")
        return
    rows = [header] + body_rows
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    print(title)
    for i, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        print(line)
        if i == 0:
            print("  ".join("-" * w for w in widths))


def _fmt(value, digits=2):
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _seq_row(label, data):
    return [
        label,
        str(data.get("frames", "")),
        f"{data['pre_ms']:.2f}",
        f"{data['infer_ms']:.2f}",
        f"{data['post_ms']:.2f}",
        f"{data['pipe_ms']:.2f}",
        f"{data['infer_fps']:.2f}",
        f"{data['pipe_fps']:.2f}",
    ]


def _part(data):
    part = data.get("exercise_part")
    if part:
        return str(part)
    proto = str(data.get("protocol") or "")
    path = str(data.get("path") or "")
    if "fastapi" in proto or path.startswith("fastapi"):
        return "5a"
    if proto.startswith("triton_") or "http_load" in proto or "grpc_load" in proto:
        return "5b"
    return "5a"


def _load_row(data):
    ig = data.get("instance_group")
    ig_s = "-" if ig is None else str(ig)
    return [
        _part(data),
        data.get("path", "load"),
        str(data.get("concurrency", "")),
        ig_s,
        str(data.get("frames", "")),
        str(data.get("errors", "")),
        _fmt(data.get("infer_ms")),
        _fmt(data.get("p50_ms")),
        _fmt(data.get("p99_ms")),
        _fmt(data.get("throughput_rps", data.get("pipe_fps"))),
        _fmt(data.get("queue_ms")),
        _fmt(data.get("compute_ms")),
        _fmt(data.get("inferences_per_execution")),
    ]


def _sort_load(data):
    part = _part(data)
    proto = str(data.get("protocol") or "")
    conc = int(data.get("concurrency") or 0)
    ig = data.get("instance_group")
    ig_n = -1 if ig is None else int(ig)
    return (part, proto, conc, ig_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    seq = []
    for filename, label in ROWS:
        path = args.dir / filename
        if not path.exists():
            print(f"missing {path} -- skip {label}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        seq.append(_seq_row(label, data))

    _print_grid(
        "Exercise 4 pose timings (one client, sequential frames; infer_ms is not COCO val inf ms)",
        ["path", "frames", "pre_ms", "infer_ms", "post_ms", "pipe_ms", "infer_fps", "pipe_fps"],
        seq,
    )
    print("")
    print("Exercise 4 = infer_fps of one stream. ")
    print("")
    print("Exercise 5 = throughput of many overlapping calls. Different questions.")
    print("")

    load_files = sorted(args.dir.glob("pose_load*.json"))
    parsed = []
    for path in load_files:
        parsed.append(json.loads(path.read_text(encoding="utf-8")))
    parsed.sort(key=_sort_load)
    load_rows = [_load_row(data) for data in parsed]

    _print_grid(
        "Exercise 5 load (5a FastAPI product hop + 5b native Triton). Same columns, different hops.",
        [
            "part",
            "path",
            "conc",
            "inst",
            "ok",
            "err",
            "infer_ms",
            "p50_ms",
            "p99_ms",
            "rps",
            "queue_ms",
            "compute_ms",
            "inf/exec",
        ],
        load_rows,
    )
    if not load_rows:
        print("  5a: python3 clients/load_pose.py --concurrency 1 --instance-group 1")
        print("      python3 clients/load_pose.py --concurrency 8 --instance-group 1")
        print("  5b: python3 clients/load_triton.py --protocol both --concurrency 1 --instance-group 1")
        print("      python3 clients/load_triton.py --protocol both --concurrency 8 --instance-group 1")
        print("  then instance_group count: 2, restart Triton, repeat with --instance-group 2.")
    print("")
    print(READ_GUIDE)


if __name__ == "__main__":
    main()
