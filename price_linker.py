from __future__ import annotations

from dataclasses import replace
import re

from ocr_service import OCRDetection
from price_ocr import PriceTag


def link_prices_to_products(
    detections: list[OCRDetection],
    price_tags: list[PriceTag],
) -> list[OCRDetection]:
    # Link detected price tags to products using text matching and spatial proximity rules
    linked_detections: list[OCRDetection] = []

    for detection in detections:
        price_tag = _match_by_text(detection, price_tags)
        match_method = "text_match" if price_tag else None

        if price_tag is None:
            price_tag = _nearest_price_below(detection.bbox, price_tags)
            match_method = "nearest_below" if price_tag else None

        linked_detections.append(
            replace(
                detection,
                linked_price=price_tag.price if price_tag else None,
                price_match_method=match_method,
            )
        )

    return linked_detections


def _match_by_text(detection: OCRDetection, price_tags: list[PriceTag]) -> PriceTag | None:
    # Match product to price by checking if brand name appears in nearby price tag text
    product_text = _normalize_for_match(detection.final_label)
    product_compact = _compact(product_text)
    if not product_compact or product_compact == "other":
        return None

    for tag in price_tags:
        tag_text = _normalize_for_match(tag.tag_text)
        tag_compact = _compact(tag_text)
        if product_text in tag_text or product_compact in tag_compact:
            return tag

    return None


def _nearest_price_below(
    product_bbox: tuple[int, int, int, int],
    price_tags: list[PriceTag],
) -> PriceTag | None:
    # Find the nearest price tag spatially positioned below/near the product (fallback matching)
    px1, py1, px2, py2 = product_bbox
    product_center_x = (px1 + px2) / 2
    product_bottom_y = py2
    product_width = max(1, px2 - px1)

    candidates: list[tuple[float, float, PriceTag]] = []
    for tag in price_tags:
        tx1, ty1, tx2, ty2 = tag.bbox
        tag_center_x = (tx1 + tx2) / 2
        tag_center_y = (ty1 + ty2) / 2
        if tag_center_y < product_bottom_y:
            continue

        horizontal_distance = abs(tag_center_x - product_center_x)
        if horizontal_distance > max(160, product_width * 0.85):
            continue

        vertical_distance = tag_center_y - product_bottom_y
        candidates.append((vertical_distance, horizontal_distance, tag))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _normalize_for_match(value: str | None) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value)
