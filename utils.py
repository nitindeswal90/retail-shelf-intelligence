from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import CROPS_DIR, IMAGE_EXTENSIONS, INPUT_DIR, OUTPUT_DIR, RESULTS_DIR
from detector import Detection
from price_ocr import PriceTag


def ensure_directories() -> None:
    # Create necessary output directories (input_images, outputs, results, crops)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)


def list_input_images() -> list[Path]:
    # List all image files in input_images directory (jpg, png, webp, bmp, etc.)
    if not INPUT_DIR.exists():
        return []
    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_path_for(image_path: Path) -> Path:
    return OUTPUT_DIR / f"{image_path.stem}_detected.jpg"


def results_path_for(image_path: Path) -> Path:
    return RESULTS_DIR / f"{image_path.stem}.json"


def crops_dir_for(image_path: Path) -> Path:
    return CROPS_DIR / image_path.name


def draw_detections(
    image_path: Path,
    detections: list[Detection],
    output_path: Path,
    price_tags: list[PriceTag] | None = None,
    segmentation_masks: dict[int, np.ndarray] | None = None,
) -> Path:
    # Draw product detection boxes and price tag boxes on image with labels
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image for annotation: {image_path}")

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (35, 180, 70)
        final_label = getattr(detection, "final_label", detection.label)
        label = _limit_label_text(final_label)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        _draw_label(image, label, x1, y1, color)

    for price_tag in price_tags or []:
        x1, y1, x2, y2 = price_tag.bbox
        color = (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        _draw_label(image, price_tag.price, x1, y1, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Unable to save annotated image: {output_path}")
    return output_path


def save_results_json(
    image_path: Path,
    detections: list[Detection],
    output_path: Path,
    price_tags: list[PriceTag] | None = None,
    shelf_space_percent: dict[str, float] | None = None,
    detection_areas: list[int] | None = None,
) -> Path:
    # Save comprehensive results to JSON: products, brands, shelf metrics, OCR labels, prices
    output_path.parent.mkdir(parents=True, exist_ok=True)
    brands: dict[str, int] = {}
    for detection in detections:
        final_label = getattr(detection, "final_label", "Other")
        brands[final_label] = brands.get(final_label, 0) + 1

    payload = {
        "image_name": image_path.name,
        "total_products": len(detections),
        "brands": dict(sorted(brands.items())),
        "shelf_space_percent": shelf_space_percent or {},
        "ocr_labels": _unique_prices(price_tags or []),
        "detections": [
            {
                "final_label": getattr(detection, "final_label", "Other"),
                "bbox": list(detection.bbox),
                "ocr_text": getattr(detection, "ocr_text", []),
                "linked_price": getattr(detection, "linked_price", None),
                "area": detection_areas[index] if detection_areas and index < len(detection_areas) else _bbox_area(detection.bbox),
                "area_source": "bbox",
            }
            for index, detection in enumerate(detections)
        ],
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _draw_label(image, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = _load_label_font()
    label = _limit_label_text(label)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    label_width = text_bbox[2] - text_bbox[0] + 8
    label_height = text_bbox[3] - text_bbox[1] + 8
    image_width = image.shape[1]
    x = min(max(0, x), max(0, image_width - label_width))
    label_top = max(0, y - label_height)

    rgb_color = (color[2], color[1], color[0])
    draw.rectangle((x, label_top, x + label_width, label_top + label_height), fill=rgb_color)
    draw.text((x + 4, label_top + 3), label, font=font, fill=(255, 255, 255))

    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _limit_label_text(label: str, max_length: int = 25) -> str:
    label = " ".join(str(label).split())
    if len(label) <= max_length:
        return label
    return label[: max_length - 3].rstrip() + "..."


def _load_label_font():
    font_candidates = [
        "C:/Windows/Fonts/seguisym.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, 14)
    return ImageFont.load_default()


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _unique_prices(price_tags: list[PriceTag]) -> list[str]:
    prices: list[str] = []
    seen: set[str] = set()
    for price_tag in price_tags:
        if price_tag.price not in seen:
            seen.add(price_tag.price)
            prices.append(price_tag.price)
    return prices
