import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import safe_download, unzip_file

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# =============================
# USER SETTINGS
# =============================
MODEL_PATH = "models/yolo26m-seg.pt"
IMAGE_SOURCE = Path("datasets/coco/images/val2017")
DATASET_DIR = Path("datasets/coco-seg")
CALIB_IMAGES = 300
OUTPUT_NAME = "yolo26m_seg_int8"
WORKSPACE = 4
ULTRALYTICS_ENGINE = Path("models/yolo26m-seg.engine")
SEG_INT8_FP16_ENGINE = Path("models/yolo26m-seg-int8-fp16.engine")
# =============================

COCO_NAMES = """
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""


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


def _extract_zip_subset(zip_path, dest_dir, needle):
    """Extract only zip members whose path contains needle (e.g. labels/val2017)."""
    from zipfile import ZipFile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    needle = needle.replace("\\", "/")
    with ZipFile(zip_path) as zf:
        for member in zf.namelist():
            normalized = member.replace("\\", "/")
            if needle not in normalized:
                continue
            name = Path(normalized).name
            if not name:
                continue
            target = dest_dir / name
            if target.exists():
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def download_data():
    IMAGE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_SOURCE.exists():
        print("Downloading COCO val2017 images...")
        _download_and_unzip(
            "https://images.cocodataset.org/zips/val2017.zip",
            IMAGE_SOURCE.parent,
        )

    dest_labels = DATASET_DIR / "labels" / "val2017"
    if dest_labels.exists() and any(dest_labels.glob("*.txt")):
        return

    print("Downloading COCO segmentation labels...")
    dest_labels.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "coco2017labels-segments.zip"
    if not zip_path.exists():
        zip_path = Path(
            safe_download(
                f"{ASSETS_URL}/coco2017labels-segments.zip",
                dir=DATASET_DIR,
                unzip=False,
                delete=False,
            )
        )
    _extract_zip_subset(zip_path, dest_labels, "labels/val2017/")
    zip_path.unlink(missing_ok=True)
    if not any(dest_labels.glob("*.txt")):
        raise RuntimeError(f"No val2017 polygon labels extracted to {dest_labels}")


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
    src_img = IMAGE_SOURCE
    src_lbl = DATASET_DIR / "labels" / "val2017"
    dst_img = DATASET_DIR / "images" / "calib"
    dst_lbl = DATASET_DIR / "labels" / "calib"
    cache = DATASET_DIR / "labels" / "calib.cache"

    labeled = []
    for label in src_lbl.glob("*.txt"):
        image = _find_image(src_img, label.stem)
        if image is not None:
            labeled.append((image, label))

    if len(labeled) < CALIB_IMAGES:
        raise RuntimeError(
            f"Need {CALIB_IMAGES} labeled seg images, found {len(labeled)} in {src_lbl}"
        )

    if _labeled_calib_count(dst_img, dst_lbl) >= CALIB_IMAGES:
        print(f"Calibration subset already has {CALIB_IMAGES} labeled images.")
        return

    print(f"Creating {CALIB_IMAGES}-image labeled segmentation calibration subset...")
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
    names_block = COCO_NAMES.strip("\n")
    yaml_content = (
        "path: {path}\n"
        "train: images/calib\n"
        "val: images/calib\n"
        "\n"
        "names:\n"
        "{names}\n"
    ).format(path=DATASET_DIR.as_posix(), names=names_block)
    with open("coco-calib-seg.yaml", "w") as f:
        f.write(yaml_content)


def _patch_tensorrt_int8_fp16_fallback():
    """Let YOLO-seg proto/cv3 Conv+SiLU fall back to FP16 when INT8 has no kernel.

    Ultralytics sets BuilderFlag.INT8 and skips FP16. TensorRT then fuses
    proto/cv3 into Conv+PWN(Sigmoid,Mul) and fails with Error Code 10.
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
    if not ULTRALYTICS_ENGINE.exists():
        raise RuntimeError(f"Export finished but {ULTRALYTICS_ENGINE} is missing.")
    if dest.exists():
        dest.unlink()
    ULTRALYTICS_ENGINE.rename(dest)
    print(f"Engine saved as {dest}")


def export_int8():
    print("Exporting INT8 TensorRT engine...")
    print("Using INT8+FP16 fallback for the segmentation proto head.")

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
            data="coco-calib-seg.yaml",
            device=0,
            workspace=WORKSPACE,
            name=OUTPUT_NAME,
        )
    finally:
        import tensorrt as trt

        trt.IBuilderConfig.set_flag = original_set_flag

    print("Export complete.")
    _move_ultralytics_engine(SEG_INT8_FP16_ENGINE)


if __name__ == "__main__":
    download_data()
    create_calibration_subset()
    create_yaml()
    export_int8()
