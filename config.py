from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
CROPS_DIR = OUTPUT_DIR / "crops"

MODEL_NAME = str(BASE_DIR / "yolov8s-worldv2.pt")

YOLO_WORLD_CLASSES = [
    "retail product",
    "packaged product",
    "product package",

    "rectangular product package",
    "square product package",
    "rectangular box",
    "square box",
    "cardboard box",
    "food box",
    "snack box",
    "biscuit box",
    "cookie box",

    "chips packet",
    "snack packet",
    "food packet",
    "biscuit packet",
    "cookie packet",
    "flat packet",
    "standing pouch",
    "pouch",
    "chips bag",
    "snack bag",

    "juice carton",
    "milk carton",
    "tetra pack",
    "beverage carton",
    "rectangular carton",
    "square carton",

    "bottle",
    "plastic bottle",
    "soft drink bottle",
    "juice bottle",
    "milk bottle",
    "drink can",
    "energy drink can",
    "tin can",

    "dairy product",
    "yogurt cup",
    "curd cup",
    "dairy cup",
    "plastic cup",
]

CONFIDENCE_THRESHOLD = 0.12
IOU_THRESHOLD = 0.18

# Duplicate cleaning only
DUPLICATE_IOU_THRESHOLD = 0.45
NESTED_BOX_THRESHOLD = 0.70
MIN_BOX_WIDTH = 40
MIN_BOX_HEIGHT = 60

PRODUCT_LIKE_CLASSES = set(YOLO_WORLD_CLASSES)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
