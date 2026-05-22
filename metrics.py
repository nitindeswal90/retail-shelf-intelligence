from __future__ import annotations

from pathlib import Path

import numpy as np

from detector import Detection


def calculate_shelf_metrics(
    image_path: str | Path,
    detections: list[Detection],
    segmentation_model: str | None = None,
) -> tuple[dict[str, float], list[int], dict[int, np.ndarray]]:
    """
    Calculate shelf space percentage per brand using bounding box areas as product region approximation.

    YOLO-World detections remain the source of product boxes. Shelf space is
    estimated from product bounding-box area as a lightweight product-region
    approximation.
    """

    detection_areas = _bbox_areas(detections)
    total_area = sum(detection_areas)
    if total_area <= 0:
        return {}, detection_areas, {}

    brand_areas: dict[str, int] = {}
    for detection, area in zip(detections, detection_areas):
        final_label = getattr(detection, "final_label", "Other")
        brand_areas[final_label] = brand_areas.get(final_label, 0) + area

    shelf_space_percent = {
        brand: round((area / total_area) * 100, 1)
        for brand, area in sorted(brand_areas.items())
    }
    return shelf_space_percent, detection_areas, {}


def _bbox_areas(detections: list[Detection]) -> list[int]:
    # Calculate pixel areas for all detection bounding boxes
    return [_bbox_area(detection.bbox) for detection in detections]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)
