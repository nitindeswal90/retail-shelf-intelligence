# retail-shelf-intelligence

Retail-shelf-intelligence is an interactive final assignment pipeline for analyzing retail
shelf images. It detects products with YOLO-World, reads product labels with
EasyOCR, extracts shelf price labels, estimates brand-wise shelf space, links
prices to products, and exports one annotated image plus assignment-style JSON.

## Objective

The project answers four practical shelf-analysis questions:

- How many products are visible?
- Which brands/products are present?
- How much shelf space does each brand occupy?
- Which visible shelf prices can be linked to products?

## Features

- YOLO-World product detection.
- EasyOCR product crop labeling.
- Brand mapping from OCR text.
- Full-image EasyOCR price tag detection.
- Product-price linking using text match first and nearest-below fallback second.
- Shelf-space estimation from product bounding-box area.
- One final annotated image per input.
- Assignment-style JSON output.
- Lightweight Streamlit GUI with compact product cards.

## Folder Structure

```text
object detection/
|-- app.py
|-- brand_mapper.py
|-- config.py
|-- detector.py
|-- main.py
|-- metrics.py
|-- ocr_service.py
|-- price_linker.py
|-- price_ocr.py
|-- utils.py
|-- docs/
|   `-- pipeline.md
|-- input_images/
|-- outputs/
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Add shelf images to:

```text
input_images/
```

## Run

Batch processing:

```bash
python main.py
```

Streamlit GUI:

```bash
streamlit run app.py
```

## Pipeline

```text
Image
  -> YOLO-World Detection
  -> Crop OCR with EasyOCR
  -> Brand Mapping
  -> Full-image Price OCR
  -> Bbox-area Shelf-Space Estimation
  -> Product-Price Linking
  -> Final JSON + Annotated Image + Streamlit
```

## Final Annotated Image

Each input creates one image:

```text
outputs/<image_name>_detected.jpg
```

The image includes:

- green product boxes with clean product/brand names,
- orange price tag boxes with clean labels such as `Rs 20`, `Rs 50`, or `Rs 99`,
- no segmentation mask overlay,
- no raw OCR text,
- no confidence values,
- no separate price overlay image.

## JSON Output

Each input creates:

```text
outputs/results/<image_name>.json
```

Format:

```json
{
  "image_name": "img_1.jpg",
  "total_products": 82,
  "brands": {
    "Coca-Cola": 5,
    "Pepsi": 4,
    "Other": 10
  },
  "shelf_space_percent": {
    "Coca-Cola": 24.5,
    "Pepsi": 18.2,
    "Other": 57.3
  },
  "ocr_labels": ["Rs 20", "Rs 50", "Rs 99"],
  "detections": [
    {
      "final_label": "Coca-Cola",
      "bbox": [100, 120, 180, 260],
      "ocr_text": ["Coke"],
      "linked_price": "Rs 50",
      "area": 12345,
      "area_source": "bbox"
    }
  ]
}
```

## Streamlit GUI

The GUI shows:

- original uploaded image,
- final annotated image,
- total product count,
- brand-wise counts,
- shelf-space percentage table,
- compact product cards with crop image, product name, and linked price,
- OCR labels / price tags,
- product table,
- JSON preview and download.

## Shelf-Space Estimation

Shelf-space estimation is calculated using product bounding-box area as a
lightweight product-region segmentation approximation.

For each detected product, the bbox area is assigned to its final brand label.
Each brand total is then divided by the total product bbox area:

```text
brand_area = sum(product bbox area for that brand)
shelf_space_percent = brand_area / total_product_bbox_area * 100
```

This assumption keeps the project practical and lightweight for assignment
submission while preserving a consistent brand-wise shelf-space metric.

## Assumptions

- YOLO-World detections are used as the product bounding boxes.
- Bounding-box area approximates the visible product region for shelf-space estimation.
- EasyOCR reads enough product text for brand mapping.
- Price labels are visible in the full shelf image.

## Limitations

- OCR quality depends on image sharpness, glare, and text size.
- Nearest-below price linking can be ambiguous on dense shelves.
- Brand mapping depends on aliases in `brand_mapper.py`.
