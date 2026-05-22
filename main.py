from __future__ import annotations

from config import CONFIDENCE_THRESHOLD
from detector import ProductDetector
from metrics import calculate_shelf_metrics
from ocr_service import OCRService
from price_linker import link_prices_to_products
from price_ocr import PriceOCRService
from utils import (
    crops_dir_for,
    draw_detections,
    ensure_directories,
    list_input_images,
    output_path_for,
    results_path_for,
    save_results_json,
)


def main() -> None:
    # Main pipeline: process all input images through detection, OCR, price linking, and result saving
    ensure_directories()

    image_paths = list_input_images()
    if not image_paths:
        print("No images found in input_images/. Add .jpg, .jpeg, .png, .webp, or .bmp files.")
        return

    detector = ProductDetector(confidence_threshold=CONFIDENCE_THRESHOLD)
    ocr_service = OCRService()
    price_ocr_service = PriceOCRService(ocr_service)

    print("retail-shelf-intelligence")
    print("-" * 32)
    for image_path in image_paths:
        try:
            detections = detector.detect(image_path)
            labeled_detections = ocr_service.label_detections(
                image_path=image_path,
                detections=detections,
            )
            price_tags = price_ocr_service.detect_price_tags(
                image_path,
                product_bboxes=[detection.bbox for detection in labeled_detections],
            )
            priced_detections = link_prices_to_products(labeled_detections, price_tags)
            final_detections = ocr_service.save_product_crops(
                image_path=image_path,
                detections=priced_detections,
                crops_dir=crops_dir_for(image_path),
            )
            shelf_space_percent, detection_areas, _ = calculate_shelf_metrics(image_path, final_detections)
            output_path = draw_detections(
                image_path=image_path,
                detections=final_detections,
                output_path=output_path_for(image_path),
                price_tags=price_tags,
            )
            results_path = save_results_json(
                image_path=image_path,
                detections=final_detections,
                output_path=results_path_for(image_path),
                price_tags=price_tags,
                shelf_space_percent=shelf_space_percent,
                detection_areas=detection_areas,
            )

            print(f"Image: {image_path.name}")
            print(f"Total products detected: {len(final_detections)}")
            print(f"Price tags detected: {len(price_tags)}")
            print(f"Output image path: {output_path.as_posix()}")
            print(f"Results JSON path: {results_path.as_posix()}")
            print("-" * 32)
        except Exception as error:
            print(f"Image: {image_path.name}")
            print(f"Error: {error}")
            print("-" * 32)


if __name__ == "__main__":
    main()
