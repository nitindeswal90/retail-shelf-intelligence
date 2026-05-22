from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2


@dataclass(frozen=True)
class PriceTag:
    price: str
    tag_text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRTextBox:
    text: str
    bbox: tuple[int, int, int, int]


class PriceOCRService:
    # Main service class for detecting and extracting price tags from shelf images
    def __init__(self, ocr_service) -> None:
        self.ocr_service = ocr_service

    def detect_price_tags(
        self,
        image_path: str | Path,
        product_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[PriceTag]:
        """
        Detect price tags from an image by extracting OCR text and filtering for valid prices.
        Removes overlapping products and groups nearby text boxes to build price tag information.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image for price OCR: {image_path}")

        text_boxes = [
            OCRTextBox(text=text, bbox=bbox)
            for bbox, text in self.ocr_service.read_text_from_image(image)
            if text.strip()
        ]

        price_tags: list[PriceTag] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for box in text_boxes:
            price = clean_price_text(box.text)
            if price is None:
                continue

            if _overlaps_product(box.bbox, product_bboxes or []):
                continue

            grouped_boxes = _nearby_tag_boxes(box, text_boxes)
            tag_text = _build_tag_text(grouped_boxes)
            tag_bbox = box.bbox
            if not _is_valid_price_bbox(tag_bbox, image.shape):
                continue

            key = (price, tag_bbox)
            if key in seen:
                continue

            seen.add(key)
            price_tags.append(PriceTag(price=price, tag_text=tag_text, bbox=tag_bbox))

        return sorted(price_tags, key=lambda item: (item.bbox[1], item.bbox[0]))


def clean_price_text(text: str | None) -> str | None:
    """
    Extract and normalize price values from OCR text using regex patterns.
    Handles various Indian price formats (₹, Rs., INR, MRP) and filters out size tokens.
    """
    if not text:
        return None

    normalized = str(text).strip().lower()
    patterns = [
        r"(?:\u20b9|rs\.?|inr|mrp|price)\s*[:=.-]?\s*(\d{1,5})",
        r"(\d{1,5})\s*(?:/-|rs\.?|\u20b9)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return f"\u20b9{match.group(1)}"

    token = re.sub(r"[^a-z0-9.]+", "", normalized)
    if _looks_like_size_token(token):
        return None

    if re.fullmatch(r"\d{1,4}", token):
        return f"\u20b9{int(token)}"

    return None


def _looks_like_size_token(token: str) -> bool:
    # Check if token is a size/weight unit (e.g., "250g", "1.5L") to exclude from price detection
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:g|gm|kg|ml|l|ltr|lt|oz)", token))


def _is_valid_price_bbox(bbox: tuple[int, int, int, int], image_shape) -> bool:
    # Validate bounding box dimensions to ensure it's a realistic price tag (not too large or too small)
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False
    if width > image_width * 0.18:
        return False
    if height > image_height * 0.10:
        return False
    return True


def _overlaps_product(
    bbox: tuple[int, int, int, int],
    product_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    # Check if detected text overlaps significantly with product regions to avoid false positives
    for product_bbox in product_bboxes:
        if _overlap_ratio(bbox, product_bbox) > 0.05:
            return True
    return False


def _overlap_ratio(
    bbox: tuple[int, int, int, int],
    other_bbox: tuple[int, int, int, int],
) -> float:
    # Calculate intersection-over-union (IoU) ratio between two bounding boxes
    x1 = max(bbox[0], other_bbox[0])
    y1 = max(bbox[1], other_bbox[1])
    x2 = min(bbox[2], other_bbox[2])
    y2 = min(bbox[3], other_bbox[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return intersection / area


def _nearby_tag_boxes(price_box: OCRTextBox, text_boxes: list[OCRTextBox]) -> list[OCRTextBox]:
    # Group text boxes that are spatially close to the price box (e.g., price labels, discounts)
    px1, py1, px2, py2 = price_box.bbox
    price_center_x = (px1 + px2) / 2
    price_center_y = (py1 + py2) / 2
    price_height = max(1, py2 - py1)

    grouped = [price_box]
    for box in text_boxes:
        if box is price_box:
            continue

        x1, y1, x2, y2 = box.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        vertical_close = abs(center_y - price_center_y) <= max(70, price_height * 3)
        horizontal_close = abs(center_x - price_center_x) <= 450
        if vertical_close and horizontal_close:
            grouped.append(box)

    return sorted(grouped, key=lambda item: (item.bbox[1], item.bbox[0]))


def _build_tag_text(text_boxes: list[OCRTextBox]) -> str:
    # Build combined text from grouped text boxes, removing redundant price tokens
    tokens = []
    for box in text_boxes:
        token = _remove_price_tokens(box.text)
        if token:
            tokens.append(token)
    return " ".join(tokens)


def _remove_price_tokens(text: str) -> str:
    # Filter out price-related tokens and size tokens to keep only descriptive text
    kept_tokens = []
    for token in str(text).split():
        normalized = re.sub(r"[^a-z0-9.]+", "", token.lower())
        if clean_price_text(token) is not None and not _looks_like_size_token(normalized):
            continue
        kept_tokens.append(token)
    return " ".join(kept_tokens)
