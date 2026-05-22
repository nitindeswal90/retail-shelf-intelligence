from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CONFIDENCE_THRESHOLD,
    DUPLICATE_IOU_THRESHOLD,
    IOU_THRESHOLD,
    MIN_BOX_HEIGHT,
    MIN_BOX_WIDTH,
    MODEL_NAME,
    NESTED_BOX_THRESHOLD,
    PRODUCT_LIKE_CLASSES,
    YOLO_WORLD_CLASSES,
)


@dataclass(frozen=True)
class Detection:
    """Single YOLO detection result."""

    bbox: tuple[int, int, int, int]
    confidence: float
    label: str


class ProductDetector:
    """YOLO-World object detector with light duplicate-box cleaning."""
    # Detects retail products in shelf images using YOLO-World with custom product classes

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        classes: list[str] | tuple[str, ...] = YOLO_WORLD_CLASSES,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self.model = YOLO(model_name)

        if classes:
            if not hasattr(self.model, "set_classes"):
                raise ValueError("Custom prompt classes require a YOLO-World model.")
            self.model.set_classes(list(classes))

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_path: str | Path) -> list[Detection]:
        # Main detection method: load image and run YOLO inference to find all products
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        return self._detect_with_yolo(image)

    def _detect_with_yolo(self, image: np.ndarray) -> list[Detection]:
        # Run YOLO model inference and filter detections by confidence, size, and class type
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        if not results:
            return []

        detections: list[Detection] = []
        names = results[0].names

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id)).lower().strip()

            if PRODUCT_LIKE_CLASSES and label not in PRODUCT_LIKE_CLASSES:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

            if x2 <= x1 or y2 <= y1:
                continue

            if (x2 - x1) < MIN_BOX_WIDTH or (y2 - y1) < MIN_BOX_HEIGHT:
                continue

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=round(float(box.conf[0]), 4),
                    label=label,
                )
            )

        detections = self._remove_duplicate_boxes(detections)

        return sorted(detections, key=lambda item: (item.bbox[1], item.bbox[0]))

    @staticmethod
    def _area(box: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _intersection(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> int:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0

        return (ix2 - ix1) * (iy2 - iy1)

    @classmethod
    def _iou(
        cls,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        inter = cls._intersection(box_a, box_b)
        area_a = cls._area(box_a)
        area_b = cls._area(box_b)

        union = area_a + area_b - inter
        if union <= 0:
            return 0.0

        return inter / union

    @classmethod
    def _remove_duplicate_boxes(cls, detections: list[Detection]) -> list[Detection]:
        """
        Remove duplicate and overlapping detections (box-inside-box cases).
        Keeps the highest confidence detection and removes nearby low-confidence duplicates.
        """

        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        for det in detections:
            box = det.bbox
            box_area = cls._area(box)

            keep = True

            for old in kept:
                old_box = old.bbox
                old_area = cls._area(old_box)

                inter = cls._intersection(box, old_box)

                if inter <= 0:
                    continue

                iou = cls._iou(box, old_box)

                # If one box is mostly inside another, remove duplicate.
                smaller_overlap = inter / max(1, min(box_area, old_area))

                if iou >= DUPLICATE_IOU_THRESHOLD:
                    keep = False
                    break

                if smaller_overlap >= NESTED_BOX_THRESHOLD:
                    keep = False
                    break

            if keep:
                kept.append(det)

        return kept
