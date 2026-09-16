import argparse
import os
import subprocess
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

POSE_PT = "models/yolo26m-pose.pt"
POSE_ENGINE = "models/yolo26m-pose-int8-fp16.engine"
SEG_PT = "models/yolo26m-seg.pt"
SEG_ENGINE = "models/yolo26m-seg-int8-fp16.engine"

POSE_YAML = "coco-val-pose.yaml"
SEG_YAML = "coco-val-seg.yaml"
SEG_NAMES_SRC = "coco-calib-seg.yaml"


def _abs_posix(path):
    return Path(path).resolve().as_posix()


def _find_image(folder, stem):
    folder = Path(folder)
    for ext in (".jpg", ".jpeg", ".png"):
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def write_pose_yaml():
    dataset = Path("datasets/coco").resolve()
    images = dataset / "images" / "val2017"
    labels = dataset / "labels" / "val2017"
    list_path = dataset / "val2017-pose.txt"
    lines = []
    for label in sorted(labels.glob("*.txt")):
        image = _find_image(images, label.stem)
        if image is not None:
            # Ultralytics only joins list-file lines to the dataset root if they start with ./
            lines.append(f"./images/val2017/{image.name}")
    if not lines:
        raise RuntimeError("No pose-labeled val images found.")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    Path(POSE_YAML).write_text(
        "\n".join(
            [
                f"path: {_abs_posix(dataset)}",
                "train: images/calib",
                f"val: {list_path.name}",
                "",
                "kpt_shape: [17, 3]",
                "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]",
                "",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {POSE_YAML} with {len(lines)} labeled pose val images.")


def _ensure_seg_image_junction():
    target = Path("datasets/coco/images/val2017").resolve()
    link = Path("datasets/coco-seg/images/val2017")
    if not target.is_dir():
        raise RuntimeError(f"Missing COCO images at {target}")
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not link {link} -> {target}: {result.stdout} {result.stderr}"
            )
    else:
        link.symlink_to(target, target_is_directory=True)


def write_seg_yaml():
    _ensure_seg_image_junction()
    names_src = Path(SEG_NAMES_SRC)
    if not names_src.exists():
        raise RuntimeError(f"Missing {SEG_NAMES_SRC} for COCO class names.")
    text = names_src.read_text(encoding="utf-8")
    names_block = text.split("names:", 1)[1].strip("\n")

    dataset = Path("datasets/coco-seg").resolve()
    # Look up JPEGs in the real coco folder. Do not Path.resolve() the
    # coco-seg/images/val2017 junction — Windows follows it to coco/images,
    # and Ultralytics then maps labels to datasets/coco/labels (pose keypoints).
    images = Path("datasets/coco/images/val2017")
    labels = dataset / "labels" / "val2017"
    list_path = dataset / "val2017-seg.txt"
    lines = []
    for label in sorted(labels.glob("*.txt")):
        image = _find_image(images, label.stem)
        if image is not None:
            lines.append(f"./images/val2017/{image.name}")
    if not lines:
        raise RuntimeError("No coco-seg labeled val images found.")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    Path(SEG_YAML).write_text(
        "\n".join(
            [
                f"path: {_abs_posix(dataset)}",
                "train: images/calib",
                f"val: {list_path.name}",
                "",
                "names:",
                names_block,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {SEG_YAML} with {len(lines)} labeled coco-seg val images "
        "(list file, so labels stay under datasets/coco-seg)."
    )


def extract_metrics(metrics, task):
    row = {
        "box_mAP50-95": float(metrics.box.map),
        "box_mAP50": float(metrics.box.map50),
    }
    if task == "pose":
        row["pose_mAP50-95"] = float(metrics.pose.map)
        row["pose_mAP50"] = float(metrics.pose.map50)
    else:
        row["mask_mAP50-95"] = float(metrics.seg.map)
        row["mask_mAP50"] = float(metrics.seg.map50)
    return row


def _clear_stale_val_caches():
    for stale in (
        Path("images/val2017.cache"),
        Path("datasets/coco/images/val2017.cache"),
        Path("datasets/coco-seg/images/val2017.cache"),
        Path("datasets/coco/labels/val2017.cache"),
        Path("datasets/coco-seg/labels/val2017.cache"),
        Path("datasets/coco/val2017-pose.cache"),
        Path("datasets/coco-seg/val2017-seg.cache"),
    ):
        if stale.exists():
            stale.unlink()


def run_val(model_path, data, int8, args):
    _clear_stale_val_caches()
    model = YOLO(model_path)
    return model.val(
        data=data,
        imgsz=args.imgsz,
        batch=args.batch,
        rect=False,
        device=args.device,
        plots=False,
        verbose=True,
        int8=int8,
        fraction=args.fraction,
        workers=args.workers,
        project="runs/val-compare",
        name=Path(model_path).stem,
        exist_ok=True,
    )


def print_table(title, columns, rows):
    print("")
    print(title)
    header = ["model"] + columns
    widths = [max(len(header[0]), max((len(r[0]) for r in rows), default=0))]
    for i, col in enumerate(columns):
        width = len(col)
        for row in rows:
            width = max(width, len(row[i + 1]))
        widths.append(width)

    def fmt(cells):
        return "  ".join(str(cell).ljust(w) for cell, w in zip(cells, widths))

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _engine_precision_label(engine_path):
    stem = Path(engine_path).stem.lower()
    has_int8 = "int8" in stem
    has_fp16 = "fp16" in stem
    if has_int8 and has_fp16:
        return "INT8+FP16", True
    if has_int8:
        return "INT8", True
    if has_fp16:
        return "FP16", False
    return "engine", True


def compare_task(task, label, pt_path, engine_path, data, columns, args):
    precision, engine_int8 = _engine_precision_label(engine_path)
    print("=" * 72)
    print(f"{label} accuracy: PyTorch vs TensorRT {precision}")
    print("=" * 72)

    results = []
    for path, int8 in ((pt_path, False), (engine_path, engine_int8)):
        if not Path(path).exists():
            print(f"Skipping missing model: {path}")
            continue
        print(f"Validating {path} ...")
        metrics = run_val(path, data, int8, args)
        row = extract_metrics(metrics, task)
        results.append((Path(path).name, row))

    if len(results) < 1:
        print(f"No {label} models found.")
        return

    table_rows = []
    for name, row in results:
        table_rows.append([name] + [f"{row[c]:.4f}" for c in columns])
    if len(results) == 2:
        delta = []
        for col in columns:
            delta.append(f"{results[1][1][col] - results[0][1][col]:+.4f}")
        table_rows.append(["delta (engine - pt)"] + delta)
    print_table(f"{label} mAP", columns, table_rows)
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs TensorRT INT8 accuracy with Ultralytics val()."
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the val set (1.0 = full COCO val). Use 0.1 for a quicker check.",
    )
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument(
        "--pose-engine",
        default=POSE_ENGINE,
        help="Pose TensorRT engine vs the .pt. Default is mixed INT8+FP16; pass models/yolo26m-pose-fp16.engine for FP16 (paths relative to optimize/).",
    )
    args = parser.parse_args()

    if not args.skip_pose:
        write_pose_yaml()
        compare_task(
            "pose",
            "POSE",
            POSE_PT,
            args.pose_engine,
            POSE_YAML,
            ["box_mAP50-95", "box_mAP50", "pose_mAP50-95", "pose_mAP50"],
            args,
        )
    if not args.skip_seg:
        write_seg_yaml()
        compare_task(
            "seg",
            "SEG",
            SEG_PT,
            SEG_ENGINE,
            SEG_YAML,
            ["box_mAP50-95", "box_mAP50", "mask_mAP50-95", "mask_mAP50"],
            args,
        )


if __name__ == "__main__":
    main()
