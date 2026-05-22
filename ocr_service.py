from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from brand_mapper import map_brand
from detector import Detection


@dataclass(frozen=True)
class OCRDetection:
    """YOLO detection enriched with OCR text and final product label."""

    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    final_label: str
    ocr_text: list[str]
    linked_price: str | None = None
    price_match_method: str | None = None
    crop_path: str | None = None


class OCRService:
    # OCR service for extracting text from product crops using EasyOCR
    def __init__(
        self,
        languages: list[str] | None = None,
        crop_padding: int = 5,
        gpu: bool = False,
    ) -> None:
        if crop_padding < 0:
            raise ValueError("crop_padding must be zero or greater.")

        self.backend = "easyocr"
        self.reader = self._init_reader(languages or ["en"], gpu)
        self.crop_padding = crop_padding

    def label_detections(
        self,
        image_path: str | Path,
        detections: list[Detection],
    ) -> list[OCRDetection]:
        # Extract OCR text from product crops and map to brand using brand mapper
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image for OCR: {image_path}")

        labeled_detections: list[OCRDetection] = []
        for detection in detections:
            crop = self._crop_with_padding(image, detection.bbox)
            ocr_text = self.read_text_from_crop(crop) if crop.size else []
            final_label = map_brand(ocr_text)

            labeled_detections.append(
                OCRDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    label=detection.label,
                    final_label=final_label,
                    ocr_text=ocr_text,
                )
            )

        return labeled_detections

    def save_product_crops(
        self,
        image_path: str | Path,
        detections: list[OCRDetection],
        crops_dir: Path,
    ) -> list[OCRDetection]:
        # Save cropped product images to disk with safe filenames based on brand labels
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image for crop saving: {image_path}")

        crops_dir.mkdir(parents=True, exist_ok=True)
        for old_crop in crops_dir.glob("*.jpg"):
            old_crop.unlink()

        saved_detections: list[OCRDetection] = []
        label_counts: dict[str, int] = {}
        for detection in detections:
            crop = self._crop_with_padding(image, detection.bbox)
            safe_label = _safe_filename(detection.final_label)
            label_counts[safe_label] = label_counts.get(safe_label, 0) + 1
            crop_path = crops_dir / f"{safe_label}_{label_counts[safe_label]:03d}.jpg"

            if crop.size:
                cv2.imwrite(str(crop_path), crop)

            saved_detections.append(
                OCRDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    label=detection.label,
                    final_label=detection.final_label,
                    ocr_text=detection.ocr_text,
                    linked_price=detection.linked_price,
                    price_match_method=detection.price_match_method,
                    crop_path=crop_path.as_posix(),
                )
            )

        return saved_detections

    def _crop_with_padding(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        padding = self.crop_padding

        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(width, x2 + padding)
        crop_y2 = min(height, y2 + padding)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return np.empty((0, 0, 3), dtype=image.dtype)

        return image[crop_y1:crop_y2, crop_x1:crop_x2]

    def read_text_from_crop(self, crop: np.ndarray) -> list[str]:
        """Extract and return all OCR text lines from a product crop image."""
        return [text for _, text in self.read_text_from_image(crop)]

    def read_text_from_image(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
        """Run EasyOCR on image and return bounding boxes with recognized text lines."""
        if image.size == 0:
            return []

        try:
            result = self.reader.readtext(image, detail=1, paragraph=False)
            return _parse_easyocr_result(result)
        except Exception:
            return []

    def _init_reader(self, languages: list[str], gpu: bool):
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError(
                "EasyOCR is required for OCR. Install dependencies with: pip install -r requirements.txt"
            ) from error

        return easyocr.Reader(languages, gpu=gpu)


def _safe_filename(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_-]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_-")
    return value or "Other"


def _parse_easyocr_result(result) -> list[tuple[tuple[int, int, int, int], str]]:
    parsed: list[tuple[tuple[int, int, int, int], str]] = []
    for item in result or []:
        try:
            points, text = item[0], item[1]
            xs = [int(point[0]) for point in points]
            ys = [int(point[1]) for point in points]
        except (IndexError, TypeError, ValueError):
            continue

        clean_text = str(text).strip()
        if clean_text:
            parsed.append(((min(xs), min(ys), max(xs), max(ys)), clean_text))

    return parsed
