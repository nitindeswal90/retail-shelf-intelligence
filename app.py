from __future__ import annotations

import json
from html import escape
from pathlib import Path

import streamlit as st

from config import CONFIDENCE_THRESHOLD, OUTPUT_DIR
from detector import ProductDetector
from metrics import calculate_shelf_metrics
from ocr_service import OCRService
from price_linker import link_prices_to_products
from price_ocr import PriceOCRService
from utils import (
    crops_dir_for,
    draw_detections,
    ensure_directories,
    output_path_for,
    results_path_for,
    save_results_json,
)


UPLOAD_DIR = OUTPUT_DIR / "streamlit_uploads"


@st.cache_resource
def load_detector() -> ProductDetector:
    # Load and cache the YOLO-World product detector model for efficient reuse
    return ProductDetector(confidence_threshold=CONFIDENCE_THRESHOLD)


@st.cache_resource
def load_ocr_service() -> OCRService:
    # Load and cache the EasyOCR service for text recognition from product crops
    return OCRService()


def _inject_card_styles() -> None:
    # Inject custom CSS styles for product card display in Streamlit UI
    st.markdown(
        """
        <style>
        .product-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.38rem 0.45rem;
            margin: 0.25rem 0 0.8rem;
            background: #ffffff;
        }
        .product-name {
            color: #111827;
            font-size: 0.82rem;
            font-weight: 650;
            line-height: 1.2;
            margin: 0;
        }
        .product-price {
            color: #374151;
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_product_cards(detections: list[dict]) -> None:
    # Render detected products as visual cards with crop images, labels, and prices
    st.subheader("Product cards")
    if not detections:
        st.info("No products detected.")
        return

    columns = st.columns(6)
    for index, item in enumerate(detections):
        with columns[index % 4]:
            crop_path = Path(item.get("crop_path") or "")
            if crop_path.is_file():
                st.image(str(crop_path), use_container_width=True)
            product_name = escape(str(item.get("final_label") or "Unknown"))
            product_price = escape(str(item.get("linked_price") or "unknown"))
            st.markdown(
                f"""
                <div class="product-card">
                    <div class="product-name">{product_name}</div>
                    <div class="product-price">Price: {product_price}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    # Main Streamlit app: upload shelf image, run full pipeline (detection, OCR, price matching)
    st.set_page_config(page_title="retail-shelf-intelligence", layout="wide")
    _inject_card_styles()
    st.title("retail-shelf-intelligence")

    uploaded_file = st.file_uploader(
        "Upload a shelf image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    run_analysis = st.button("Run analysis", type="primary", disabled=uploaded_file is None)
    if uploaded_file is None:
        st.info("Upload an image, then run analysis to detect products, OCR labels, and link prices.")
        return

    ensure_directories()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_path = UPLOAD_DIR / Path(uploaded_file.name).name
    uploaded_bytes = uploaded_file.getvalue()
    uploaded_path.write_bytes(uploaded_bytes)

    st.subheader("Original image")
    st.image(uploaded_bytes, use_container_width=True)

    if not run_analysis:
        return

    detector = load_detector()
    ocr_service = load_ocr_service()
    price_ocr_service = PriceOCRService(ocr_service)

    with st.spinner("Running detection, OCR, price extraction, and text matching..."):
        detections = detector.detect(uploaded_path)
        labeled_detections = ocr_service.label_detections(
            image_path=uploaded_path,
            detections=detections,
        )
        price_tags = price_ocr_service.detect_price_tags(
            uploaded_path,
            product_bboxes=[detection.bbox for detection in labeled_detections],
        )
        priced_detections = link_prices_to_products(labeled_detections, price_tags)
        final_detections = ocr_service.save_product_crops(
            image_path=uploaded_path,
            detections=priced_detections,
            crops_dir=crops_dir_for(uploaded_path),
        )
        shelf_space_percent, detection_areas, _ = calculate_shelf_metrics(uploaded_path, final_detections)
        annotated_path = draw_detections(
            image_path=uploaded_path,
            detections=final_detections,
            output_path=output_path_for(uploaded_path),
            price_tags=price_tags,
        )
        json_path = save_results_json(
            image_path=uploaded_path,
            detections=final_detections,
            output_path=results_path_for(uploaded_path),
            price_tags=price_tags,
            shelf_space_percent=shelf_space_percent,
            detection_areas=detection_areas,
        )

    result_data = json.loads(json_path.read_text(encoding="utf-8"))

    st.subheader("Annotated output")
    st.image(str(annotated_path), use_container_width=True)

    st.metric("Total products", result_data["total_products"])

    st.subheader("Brand-wise count")
    brand_rows = [
        {"Brand": brand, "Count": count}
        for brand, count in sorted(result_data["brands"].items(), key=lambda item: (-item[1], item[0]))
    ]
    st.dataframe(brand_rows, use_container_width=True, hide_index=True)

    st.subheader("Shelf-space percentage")
    shelf_rows = [
        {"Brand": brand, "Shelf Space %": percent}
        for brand, percent in sorted(result_data["shelf_space_percent"].items(), key=lambda item: (-item[1], item[0]))
    ]
    st.dataframe(shelf_rows, use_container_width=True, hide_index=True)

    product_card_items = []
    for index, item in enumerate(result_data["detections"]):
        card_item = dict(item)
        if index < len(final_detections):
            card_item["crop_path"] = str(getattr(final_detections[index], "crop_path", ""))
        product_card_items.append(card_item)
    _render_product_cards(product_card_items)

    st.subheader("OCR labels / price tags")
    st.dataframe(
        [{"OCR Label": label} for label in result_data["ocr_labels"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Product table")
    product_rows = [
        {
            "Product": item["final_label"],
            "Price": item.get("linked_price") or "unknown",
            "Area": item["area"],
            "OCR Text": ", ".join(item["ocr_text"]) if item["ocr_text"] else "None",
            "BBox": item["bbox"],
        }
        for item in result_data["detections"]
    ]
    st.dataframe(product_rows, use_container_width=True, hide_index=True)

    st.subheader("JSON result")
    st.json(result_data)

    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Download JSON",
            data=json.dumps(result_data, indent=2).encode("utf-8"),
            file_name=json_path.name,
            mime="application/json",
        )
    with download_right:
        st.download_button(
            "Download annotated image",
            data=annotated_path.read_bytes(),
            file_name=annotated_path.name,
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
