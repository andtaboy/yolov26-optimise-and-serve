import argparse
import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import safe_download, unzip_file

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "models/yolo26m-pose.pt"
DATASET_DIR = "datasets/coco"
CALIB_IMAGES = 300
OUTPUT_NAME = "yolo26m_pose_int8"
WORKSPACE = 4
ULTRALYTICS_ENGINE = Path("models/yolo26m-pose.engine")
POSE_INT8_FP16_ENGINE = Path("models/yolo26m-pose-int8-fp16.engine")
POSE_FP16_ENGINE = Path("models/yolo26m-pose-fp16.engine")


def _flatten_extracted(extracted, dest):
    """Move zip contents up one level if they unpacked into a nested folder."""
    extracted = Path(extracted)
    dest = Path(dest)
    if not extracted.is_dir() or extracted.resolve() == dest.resolve():
        return
    for item in extracted.iterdir():
        target = dest / item.name
        if not target.exists():
            shutil.move(str(item), str(target))
    shutil.rmtree(extracted, ignore_errors=True)


def _download_and_unzip(url, dest_dir):
    """Download a zip with redirect-aware Ultralytics helper, then unzip it."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / Path(url).name
    if not zip_path.exists():
        zip_path = Path(
            safe_download(url, dir=dest_dir, unzip=False, delete=False)
        )
    extracted = unzip_file(file=zip_path, path=dest_dir)
    zip_path.unlink(missing_ok=True)
    return extracted


def download_val2017():
    dataset_dir = Path(DATASET_DIR)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Download COCO val2017 images
    # -----------------------------
    if not (images_dir / "val2017").exists():
        print("Downloading COCO val2017 images...")
        _download_and_unzip(
            "https://images.cocodataset.org/zips/val2017.zip",
            images_dir,
        )

    # -----------------------------
    # Download COCO pose labels
    # -----------------------------
    if not (dataset_dir / "labels").exists():
        print("Downloading COCO pose labels...")
        extracted = _download_and_unzip(
            f"{ASSETS_URL}/coco2017labels-pose.zip",
            dataset_dir,
        )
        _flatten_extracted(extracted, dataset_dir)

def _find_image(folder, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _labeled_calib_count(dst_img, dst_lbl):
    if not dst_img.is_dir() or not dst_lbl.is_dir():
        return 0
    return sum(
        1
        for img in dst_img.iterdir()
        if img.is_file() and (dst_lbl / f"{img.stem}.txt").exists()
    )


def create_calibration_subset():
    # -----------------------------
    # Create calibration subset from labeled pose images only
    # -----------------------------
    src_img = Path(DATASET_DIR) / "images" / "val2017"
    src_lbl = Path(DATASET_DIR) / "labels" / "val2017"
    dst_img = Path(DATASET_DIR) / "images" / "calib"
    dst_lbl = Path(DATASET_DIR) / "labels" / "calib"
    cache = Path(DATASET_DIR) / "labels" / "calib.cache"

    labeled = []
    for label in src_lbl.glob("*.txt"):
        image = _find_image(src_img, label.stem)
        if image is not None:
            labeled.append((image, label))

    if len(labeled) < CALIB_IMAGES:
        raise RuntimeError(
            f"Need {CALIB_IMAGES} labeled pose images, found {len(labeled)} in {src_lbl}"
        )

    if _labeled_calib_count(dst_img, dst_lbl) >= CALIB_IMAGES:
        print(f"Calibration subset already has {CALIB_IMAGES} labeled images.")
        return

    print(f"Creating {CALIB_IMAGES}-image labeled calibration subset...")
    for folder in (dst_img, dst_lbl):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        cache.unlink()

    for image, label in random.sample(labeled, CALIB_IMAGES):
        shutil.copy(image, dst_img / image.name)
        shutil.copy(label, dst_lbl / label.name)

def create_yaml():
    # -----------------------------
    # Create YAML file
    # -----------------------------
    yaml_content = f"""
path: {DATASET_DIR}
train: images/calib
val: images/calib

kpt_shape: [17, 3]

names:
  0: person
"""

    with open("coco-calib-pose.yaml", "w") as f:
        f.write(yaml_content)


def _patch_tensorrt_int8_fp16_fallback():
    """Let layers without an INT8 kernel fall back to FP16 instead of FP32.

    Ultralytics 8.4.12 sets BuilderFlag.INT8 and skips FP16 (half=True is
    cleared when int8=True). TensorRT then cannot keep the pose head in
    half precision. Setting both flags is the same mixed path as seg.
    """
    import tensorrt as trt

    original = trt.IBuilderConfig.set_flag

    def set_flag(self, flag):
        original(self, flag)
        if int(flag) != int(trt.BuilderFlag.INT8):
            return
        original(self, trt.BuilderFlag.FP16)
        try:
            jit = trt.TacticSource.JIT_CONVOLUTIONS
            self.set_tactic_sources(self.get_tactic_sources() & ~(1 << int(jit)))
        except Exception:
            pass

    trt.IBuilderConfig.set_flag = set_flag
    return original


def _move_ultralytics_engine(dest):
    """Ultralytics always writes models/<pt-stem>.engine. Move it to the named artifact."""
    if not ULTRALYTICS_ENGINE.exists():
        raise RuntimeError(f"Export finished but {ULTRALYTICS_ENGINE} is missing.")
    if dest.exists():
        dest.unlink()
    ULTRALYTICS_ENGINE.rename(dest)
    print(f"Engine saved as {dest}")


def export_int8():
    print("Exporting INT8 TensorRT engine...")
    print("Using INT8+FP16 fallback so the pose head can stay FP16.")

    # Force a new MinMax pass; the existing cache is from the INT8-only engine.
    cache = Path("models/yolo26m-pose.cache")
    if cache.exists():
        cache.unlink()
        print(f"Deleted {cache} so TensorRT recalibrates with mixed precision.")

    original_set_flag = _patch_tensorrt_int8_fp16_fallback()
    try:
        model = YOLO(MODEL_PATH)
        model.export(
            format="engine",
            int8=True,
            half=True,
            imgsz=640,
            batch=1,
            dynamic=False,
            data="coco-calib-pose.yaml",
            name=OUTPUT_NAME,
            device=0,
            workspace=WORKSPACE,
        )
    finally:
        import tensorrt as trt

        trt.IBuilderConfig.set_flag = original_set_flag

    print("Export complete.")
    _move_ultralytics_engine(POSE_INT8_FP16_ENGINE)


def export_fp16():
    """Standard TensorRT FP16. No INT8, no mixed-precision patch."""
    print("Exporting FP16 TensorRT engine (half=True only)...")
    if ULTRALYTICS_ENGINE.exists():
        ULTRALYTICS_ENGINE.unlink()

    model = YOLO(MODEL_PATH)
    model.export(
        format="engine",
        half=True,
        int8=False,
        imgsz=640,
        batch=1,
        dynamic=False,
        device=0,
        workspace=WORKSPACE,
    )
    print("Export complete.")
    _move_ultralytics_engine(POSE_FP16_ENGINE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export a standard FP16 engine (no INT8). Does not overwrite the INT8 engine.",
    )
    args = parser.parse_args()

    download_val2017()
    if args.fp16:
        export_fp16()
    else:
        create_calibration_subset()
        create_yaml()
        export_int8()
